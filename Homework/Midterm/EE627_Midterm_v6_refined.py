"""
EE627A Midterm v6 - Refined Model
Jude Eschete

Building on v5's 0.878 score. Key changes inspired by HW6 (feature selection):

  1. Bigger MF (k=128) + longer training (80 epochs) for richer embeddings
  2. Feature selection: train once, rank features by importance, retrain
     with only the top features (HW6 lesson: fewer features = less overfit)
  3. Multiple MF ranks (k=32 + k=128) as separate features -- different
     factorization depths capture different patterns
  5. Neighbor-based CF: find similar users via MF cosine similarity,
     average their ratings for candidate's album/artist as new features
  7. Ensemble 3 models with different seeds, average scores before ranking

Uses RTX 5080 GPU.
"""

import os
import sys
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import defaultdict

# =====================================================================
# Configuration
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data", "music-recommender-2026s")

OUTPUT_DIR = SCRIPT_DIR
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.txt")
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MF Hyperparameters (#1, #3)
MF_CONFIGS = [
    {"factors": 32,  "epochs": 40, "lr": 0.01,  "label": "mf32"},
    {"factors": 128, "epochs": 80, "lr": 0.003, "label": "mf128"},
]
MF_BATCH = 65536

# Training
GROUPS_PER_USER = 30
HIDDEN_SIZES = (64, 32)   # back to simpler net (HW6 lesson)
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 2048
MAX_EPOCHS = 100
PATIENCE = 15
HARD_NEG_TEMP = 15.0

# Neighbor CF (#5)
N_NEIGHBORS = 20

# Ensemble (#7)
N_ENSEMBLE = 3


# =====================================================================
# Logging
# =====================================================================

class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


# =====================================================================
# 1. Data Parsing
# =====================================================================

def parse_training(path):
    print("[1/4] Loading training data ...")
    users = {}
    with open(path, "r") as f:
        uid = None
        for line in f:
            line = line.strip()
            if "|" in line:
                uid = int(line.split("|")[0])
                users[uid] = {}
            elif "\t" in line and uid is not None:
                parts = line.split("\t")
                users[uid][int(parts[0])] = int(parts[1])
    total = sum(len(r) for r in users.values())
    print(f"       {len(users):,} users, {total:,} ratings")
    return users


def parse_tracks(path):
    print("[2/4] Loading track metadata ...")
    tracks = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            tid = int(parts[0])
            alb = None if parts[1] == "None" else int(parts[1])
            art = None if parts[2] == "None" else int(parts[2])
            genres = [int(g) for g in parts[3:] if g.strip() and g != "None"]
            tracks[tid] = (alb, art, genres)
    print(f"       {len(tracks):,} tracks")
    return tracks


def parse_albums(path):
    print("[3/4] Loading album metadata ...")
    albums = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 1:
                continue
            alb_id = int(parts[0])
            art = None if len(parts) < 2 or parts[1] == "None" else int(parts[1])
            genres = []
            for g in parts[2:]:
                g = g.strip()
                if g and g != "None":
                    genres.append(int(g))
            albums[alb_id] = (art, genres)
    print(f"       {len(albums):,} albums")
    return albums


def parse_test(path):
    print("[4/4] Loading test data ...")
    test = []
    with open(path, "r") as f:
        uid = None
        tids = []
        for line in f:
            line = line.strip()
            if "|" in line:
                if uid is not None:
                    test.append((uid, tids))
                uid = int(line.split("|")[0])
                tids = []
            elif line:
                tids.append(int(line))
        if uid is not None:
            test.append((uid, tids))
    print(f"       {len(test):,} test users, "
          f"{sum(len(t) for _, t in test):,} candidates")
    return test


# =====================================================================
# 2. Album Enrichment
# =====================================================================

def enrich_tracks_with_albums(tracks, albums):
    enriched = {}
    stats = {"artist_filled": 0, "genres_added": 0}
    for tid, (alb, art, genres) in tracks.items():
        new_art = art
        new_genres = list(genres)
        if alb is not None and alb in albums:
            album_art, album_genres = albums[alb]
            if art is None and album_art is not None:
                new_art = album_art
                stats["artist_filled"] += 1
            track_genre_set = set(genres)
            for g in album_genres:
                if g not in track_genre_set:
                    new_genres.append(g)
                    track_genre_set.add(g)
                    stats["genres_added"] += 1
        enriched[tid] = (alb, new_art, new_genres)
    print(f"  Enrichment: {stats['artist_filled']:,} artists filled, "
          f"{stats['genres_added']:,} genres added")
    return enriched


# =====================================================================
# 3. Auxiliary Data Structures
# =====================================================================

def build_global_item_stats(user_ratings):
    print("  Building global item stats ...")
    item_sum = defaultdict(float)
    item_cnt = defaultdict(int)
    for ratings in user_ratings.values():
        for iid, r in ratings.items():
            item_sum[iid] += r
            item_cnt[iid] += 1
    stats = {iid: (item_sum[iid] / item_cnt[iid], item_cnt[iid])
             for iid in item_sum}
    print(f"       {len(stats):,} items with global stats")
    return stats


def build_track_mappings(track_meta):
    print("  Building track mappings ...")
    album_to_tracks = defaultdict(list)
    artist_to_tracks = defaultdict(list)
    for tid, (alb, art, genres) in track_meta.items():
        if alb is not None:
            album_to_tracks[alb].append(tid)
        if art is not None:
            artist_to_tracks[art].append(tid)
    print(f"       {len(album_to_tracks):,} albums, "
          f"{len(artist_to_tracks):,} artists mapped")
    return dict(album_to_tracks), dict(artist_to_tracks)


# =====================================================================
# 4. Multiple MF Models (#1, #3)
# =====================================================================

def train_mf(user_ratings, n_factors, n_epochs, lr, batch_size, label):
    """Train one MF model. Returns (uid_map, iid_map, user_factors, item_factors)."""
    print(f"\n  Training MF [{label}] (k={n_factors}, {n_epochs} epochs) ...")

    uid_to_idx = {}
    iid_to_idx = {}
    u_list, i_list, r_list = [], [], []

    for uid, ratings in user_ratings.items():
        if uid not in uid_to_idx:
            uid_to_idx[uid] = len(uid_to_idx)
        u_idx = uid_to_idx[uid]
        for iid, r in ratings.items():
            if iid not in iid_to_idx:
                iid_to_idx[iid] = len(iid_to_idx)
            u_list.append(u_idx)
            i_list.append(iid_to_idx[iid])
            r_list.append(float(r))

    n_users = len(uid_to_idx)
    n_items = len(iid_to_idx)

    users_t = torch.tensor(u_list, dtype=torch.long)
    items_t = torch.tensor(i_list, dtype=torch.long)
    ratings_t = torch.tensor(r_list, dtype=torch.float32)

    dataset = TensorDataset(users_t, items_t, ratings_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    user_emb = nn.Embedding(n_users, n_factors).to(DEVICE)
    item_emb = nn.Embedding(n_items, n_factors).to(DEVICE)
    nn.init.normal_(user_emb.weight, 0, 0.1)
    nn.init.normal_(item_emb.weight, 0, 0.1)

    optimizer = torch.optim.Adam(
        list(user_emb.parameters()) + list(item_emb.parameters()),
        lr=lr, weight_decay=1e-4)

    for epoch in range(n_epochs):
        total_loss = 0
        n_b = 0
        for u_b, i_b, r_b in loader:
            u_b, i_b, r_b = u_b.to(DEVICE), i_b.to(DEVICE), r_b.to(DEVICE)
            pred = (user_emb(u_b) * item_emb(i_b)).sum(dim=1)
            loss = ((pred - r_b) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    [{label}] Epoch {epoch+1:3d}/{n_epochs}  "
                  f"MSE={total_loss/n_b:.2f}")

    uf = user_emb.weight.detach().cpu().numpy()
    itf = item_emb.weight.detach().cpu().numpy()
    print(f"    [{label}] Done. User: {uf.shape}, Item: {itf.shape}")
    return uid_to_idx, iid_to_idx, uf, itf


# =====================================================================
# 5. Neighbor CF (#5)
# =====================================================================

def build_neighbor_index(mf_uid_map, mf_user_f, user_ratings, n_neighbors=20):
    """For each user, find N most similar users by MF cosine similarity.
    Returns {uid: [(neighbor_uid, similarity), ...]}."""
    print(f"\n  Building neighbor index (k={n_neighbors}) ...")
    t0 = time.time()

    # Normalize user factors for cosine similarity
    norms = np.linalg.norm(mf_user_f, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normed = mf_user_f / norms

    # Reverse map: idx -> uid
    idx_to_uid = {v: k for k, v in mf_uid_map.items()}

    # Only compute for users who appear in test or train
    uid_list = list(mf_uid_map.keys())
    neighbors = {}

    # Process in batches to avoid OOM
    batch_size = 1000
    for start in range(0, len(uid_list), batch_size):
        batch_uids = uid_list[start:start + batch_size]
        batch_idxs = [mf_uid_map[u] for u in batch_uids]
        batch_vecs = normed[batch_idxs]  # (batch, k)

        # Cosine sim against all users
        sims = batch_vecs @ normed.T  # (batch, n_users)

        for i, uid in enumerate(batch_uids):
            my_idx = batch_idxs[i]
            sim_row = sims[i]
            sim_row[my_idx] = -1.0  # exclude self

            top_idxs = np.argpartition(sim_row, -n_neighbors)[-n_neighbors:]
            top_idxs = top_idxs[np.argsort(sim_row[top_idxs])[::-1]]

            nbrs = []
            for ni in top_idxs:
                if ni in idx_to_uid:
                    nbrs.append((idx_to_uid[ni], float(sim_row[ni])))
            neighbors[uid] = nbrs

    t1 = time.time()
    print(f"       Built neighbors for {len(neighbors):,} users "
          f"in {t1-t0:.1f}s")
    return neighbors


def compute_neighbor_features(uid, tid, track_meta, user_ratings,
                               neighbor_index):
    """CF features from similar users' ratings (#5).
    Returns [nbr_album_score, nbr_artist_score, nbr_coverage]."""
    alb_id, art_id, _ = track_meta.get(tid, (None, None, []))
    neighbors = neighbor_index.get(uid, [])

    alb_scores = []
    art_scores = []

    for nbr_uid, sim in neighbors:
        nbr_ratings = user_ratings.get(nbr_uid, {})
        if alb_id is not None and alb_id in nbr_ratings:
            alb_scores.append(nbr_ratings[alb_id] * sim)
        if art_id is not None and art_id in nbr_ratings:
            art_scores.append(nbr_ratings[art_id] * sim)

    nbr_album = sum(alb_scores) / len(alb_scores) if alb_scores else 0.0
    nbr_artist = sum(art_scores) / len(art_scores) if art_scores else 0.0
    nbr_coverage = float(len(alb_scores) + len(art_scores))

    return [nbr_album, nbr_artist, nbr_coverage]


# =====================================================================
# 6. Feature Engineering
# =====================================================================

def compute_user_profile(ratings):
    vals = list(ratings.values())
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    return mean, var ** 0.5, n


def compute_base_features(ratings, track_meta, tid, u_mean, u_std, u_n,
                           global_stats, album_to_tracks, artist_to_tracks,
                           mf_models, uid, neighbor_index, user_ratings):
    """Base features per (user, track) pair.
    mf_models is a list of (uid_map, iid_map, user_f, item_f)."""
    alb_id, art_id, genre_ids = track_meta.get(tid, (None, None, []))

    # --- Album ---
    if alb_id is not None and alb_id in ratings:
        album_score, has_album = float(ratings[alb_id]), 1.0
    else:
        album_score, has_album = 0.0, 0.0

    # --- Artist ---
    if art_id is not None and art_id in ratings:
        artist_score, has_artist = float(ratings[art_id]), 1.0
    else:
        artist_score, has_artist = 0.0, 0.0

    # --- Genre stats ---
    genre_scores = [ratings[g] for g in genre_ids if g in ratings]
    gc = len(genre_scores)
    if gc > 0:
        g_max = float(max(genre_scores))
        g_min = float(min(genre_scores))
        g_mean = sum(genre_scores) / gc
        g_var = sum((s - g_mean) ** 2 for s in genre_scores) / gc
    else:
        g_max = g_min = g_mean = g_var = 0.0

    evidence_count = has_album + has_artist + (1.0 if gc > 0 else 0.0)

    # --- Interactions ---
    aa_int = album_score * artist_score / 100.0 if (has_album and has_artist) else 0.0
    ga_int = g_mean * artist_score / 100.0 if (has_artist and gc > 0) else 0.0
    ag_int = album_score * g_mean / 100.0 if (has_album and gc > 0) else 0.0

    # --- User-relative ---
    album_above = (album_score - u_mean) if has_album else 0.0
    artist_above = (artist_score - u_mean) if has_artist else 0.0
    genre_above = (g_mean - u_mean) if gc > 0 else 0.0

    # --- Popularity & global avg ---
    alb_gavg, alb_nrat = global_stats.get(alb_id, (0.0, 0)) if alb_id else (0.0, 0)
    art_gavg, art_nrat = global_stats.get(art_id, (0.0, 0)) if art_id else (0.0, 0)
    g_gavg_sum, g_nrat_sum, g_items = 0.0, 0.0, 0
    for gid in genre_ids:
        if gid in global_stats:
            gavg, nrat = global_stats[gid]
            g_gavg_sum += gavg
            g_nrat_sum += nrat
            g_items += 1
    avg_genre_gavg = g_gavg_sum / g_items if g_items > 0 else 0.0
    avg_genre_nrat = g_nrat_sum / g_items if g_items > 0 else 0.0

    # --- CF sibling features ---
    sibling_ratings = []
    if alb_id is not None:
        for sib in album_to_tracks.get(alb_id, []):
            if sib != tid and sib in ratings:
                sibling_ratings.append(ratings[sib])
    if art_id is not None:
        album_sibs = set(album_to_tracks.get(alb_id, [])) if alb_id else set()
        for sib in artist_to_tracks.get(art_id, []):
            if sib != tid and sib not in album_sibs and sib in ratings:
                sibling_ratings.append(ratings[sib])
    cf_score = sum(sibling_ratings) / len(sibling_ratings) if sibling_ratings else 0.0
    cf_count = len(sibling_ratings)

    # --- MF dot products (#3: multiple ranks) ---
    mf_scores = []
    for mf_uid_map, mf_iid_map, mf_user_f, mf_item_f in mf_models:
        if uid in mf_uid_map and tid in mf_iid_map:
            mf_scores.append(float(
                mf_user_f[mf_uid_map[uid]] @ mf_item_f[mf_iid_map[tid]]))
        else:
            mf_scores.append(0.0)

    # --- Neighbor CF (#5) ---
    nbr_feats = compute_neighbor_features(
        uid, tid, track_meta, user_ratings, neighbor_index)

    # --- v2 heuristic score ---
    v2_score = (has_album * album_score + has_artist * artist_score
                + gc * g_mean / 10.0)

    feats = [
        album_score, has_album,                         # 0-1
        artist_score, has_artist,                       # 2-3
        float(gc), g_max, g_min, g_mean, g_var,        # 4-8
        evidence_count,                                 # 9
        aa_int, ga_int, ag_int,                         # 10-12
        u_mean, u_std, math.log1p(u_n),                # 13-15
        album_above, artist_above, genre_above,         # 16-18
        alb_gavg, art_gavg, avg_genre_gavg,             # 19-21
        math.log1p(alb_nrat), math.log1p(art_nrat),    # 22-23
        math.log1p(avg_genre_nrat),                     # 24
        cf_score, math.log1p(cf_count),                 # 25-26
        v2_score,                                       # 27
    ]
    feats.extend(mf_scores)        # 28-29 (two MF scores)
    feats.extend(nbr_feats)        # 30-32 (neighbor CF)

    return feats


def compute_cross_features(track_meta, candidates):
    """3 cross-candidate features per candidate in a group of 6."""
    metas = [track_meta.get(tid, (None, None, [])) for tid in candidates]
    cross = []
    for i, (alb_i, art_i, genres_i) in enumerate(metas):
        n_same_art = 0
        n_same_alb = 0
        genre_overlap = 0
        gset_i = set(genres_i)
        for j, (alb_j, art_j, genres_j) in enumerate(metas):
            if i == j:
                continue
            if art_i and art_j and art_i == art_j:
                n_same_art += 1
            if alb_i and alb_j and alb_i == alb_j:
                n_same_alb += 1
            if gset_i:
                genre_overlap += len(gset_i & set(genres_j))
        cross.append([float(n_same_art), float(n_same_alb), float(genre_overlap)])
    return cross


# Total features: 33 base + 3 cross = 36
N_BASE_FEATURES = 33
N_CROSS_FEATURES = 3
N_FEATURES = N_BASE_FEATURES + N_CROSS_FEATURES


# =====================================================================
# 7. Training Data
# =====================================================================

def build_groups(user_ratings, track_meta, user_set, groups_per_user, seed):
    rng = random.Random(seed)
    track_ids = set(track_meta.keys())
    groups = []
    for uid in user_set:
        ratings = user_ratings[uid]
        user_tracks = [(iid, ratings[iid]) for iid in ratings if iid in track_ids]
        if len(user_tracks) < 6:
            continue
        for _ in range(groups_per_user):
            sample = rng.sample(user_tracks, 6)
            sample.sort(key=lambda x: x[1], reverse=True)
            tids = [t[0] for t in sample]
            rats = [float(t[1]) for t in sample]
            groups.append((uid, tids, rats))
    return groups


def featurize_groups(groups, user_ratings, track_meta, global_stats,
                     album_to_tracks, artist_to_tracks,
                     mf_models, neighbor_index):
    n = len(groups)
    X = np.zeros((n, 6, N_FEATURES), dtype=np.float32)
    R = np.zeros((n, 6), dtype=np.float32)

    for g, (uid, tids, rats) in enumerate(groups):
        ratings = user_ratings[uid]
        u_mean, u_std, u_n = compute_user_profile(ratings)

        for c, tid in enumerate(tids):
            base = compute_base_features(
                ratings, track_meta, tid, u_mean, u_std, u_n,
                global_stats, album_to_tracks, artist_to_tracks,
                mf_models, uid, neighbor_index, user_ratings)
            X[g, c, :N_BASE_FEATURES] = base
            R[g, c] = rats[c]

        cross = compute_cross_features(track_meta, tids)
        for c in range(6):
            X[g, c, N_BASE_FEATURES:] = cross[c]

        if (g + 1) % 50000 == 0:
            print(f"       Featurized {g+1:,}/{n:,} groups ...")

    print(f"       Featurized {n:,} groups -> X{X.shape}")
    return X, R


# =====================================================================
# 8. Feature Normalization
# =====================================================================

def compute_norm_stats(X):
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std


# =====================================================================
# 9. Feature Selection (#2)
# =====================================================================

def feature_importance(model, val_X, val_R):
    """Permutation importance: for each feature, shuffle it and
    measure AUC drop. Bigger drop = more important."""
    print("\n  Computing feature importance (permutation) ...")
    base_auc = compute_val_auc(model, val_X, val_R)
    print(f"    Base val AUC: {base_auc:.4f}")

    n_features = val_X.shape[2]
    importances = []

    for f in range(n_features):
        # Shuffle feature f across all candidates
        X_perm = val_X.copy()
        flat = X_perm.reshape(-1, n_features)
        rng = np.random.RandomState(SEED + f)
        rng.shuffle(flat[:, f])
        X_perm = flat.reshape(val_X.shape)

        perm_auc = compute_val_auc(model, X_perm, val_R)
        drop = base_auc - perm_auc
        importances.append((f, drop))

    importances.sort(key=lambda x: x[1], reverse=True)

    print("    Feature importance (top 15):")
    for rank, (f_idx, drop) in enumerate(importances[:15]):
        print(f"      #{rank+1:2d}  feature[{f_idx:2d}]  "
              f"AUC drop = {drop:+.4f}")

    print(f"    Bottom 5:")
    for f_idx, drop in importances[-5:]:
        print(f"          feature[{f_idx:2d}]  "
              f"AUC drop = {drop:+.4f}")

    return importances


def select_features(X, importances, min_drop=0.0001):
    """Keep only features with importance above threshold."""
    selected = [f_idx for f_idx, drop in importances if drop > min_drop]
    if len(selected) < 5:
        # Keep at least top 10
        selected = [f_idx for f_idx, _ in importances[:10]]
    selected.sort()
    return selected


def apply_feature_mask(X, feature_indices):
    """Reduce feature dimension to selected indices."""
    return X[:, :, feature_indices]


# =====================================================================
# 10. Model
# =====================================================================

class RankingNet(nn.Module):
    def __init__(self, n_features, hidden_sizes=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        layers.append(nn.BatchNorm1d(n_features))
        prev = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch, n_cand, n_feat = x.shape
        flat = x.reshape(batch * n_cand, n_feat)
        scores = self.net(flat)
        return scores.reshape(batch, n_cand)


# =====================================================================
# 11. Loss
# =====================================================================

def bpr_group_loss(scores, ratings, hard_neg_temp=15.0):
    pos_scores = scores[:, :3]
    neg_scores = scores[:, 3:]
    pos_ratings = ratings[:, :3]
    neg_ratings = ratings[:, 3:]

    pos_exp = pos_scores.unsqueeze(2)
    neg_exp = neg_scores.unsqueeze(1)
    diff = pos_exp - neg_exp

    pr_exp = pos_ratings.unsqueeze(2)
    nr_exp = neg_ratings.unsqueeze(1)
    rating_gap = (pr_exp - nr_exp).abs()
    weight = torch.exp(-rating_gap / hard_neg_temp)

    pair_loss = -torch.log(torch.sigmoid(diff) + 1e-8) * weight
    return pair_loss.mean()


# =====================================================================
# 12. Training
# =====================================================================

class GroupDataset(Dataset):
    def __init__(self, X, R):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.R = torch.tensor(R, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.R[idx]


def compute_val_auc(model, val_X, val_R):
    model.eval()
    dataset = GroupDataset(val_X, val_R)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, r_batch in loader:
            x_batch = x_batch.to(DEVICE)
            scores = model(x_batch).cpu().numpy()
            for i in range(scores.shape[0]):
                for p in range(3):
                    for n in range(3, 6):
                        if scores[i, p] > scores[i, n]:
                            correct += 1
                        elif scores[i, p] == scores[i, n]:
                            correct += 0.5
                        total += 1
    return correct / total if total > 0 else 0.0


def train_ranking_model(train_X, train_R, val_X, val_R, n_features,
                         seed_offset=0):
    """Train one model. seed_offset allows different seeds for ensemble."""
    torch.manual_seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)

    dataset = GroupDataset(train_X, train_R)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0,
                        pin_memory=(DEVICE.type == "cuda"))

    model = RankingNet(n_features, HIDDEN_SIZES, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, MAX_EPOCHS)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for x_batch, r_batch in loader:
            x_batch = x_batch.to(DEVICE)
            r_batch = r_batch.to(DEVICE)
            scores = model(x_batch)
            loss = bpr_group_loss(scores, r_batch, HARD_NEG_TEMP)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample_x = torch.tensor(
                    train_X[:2048], dtype=torch.float32, device=DEVICE)
                sample_scores = model(sample_x).cpu().numpy()
                train_correct = 0
                train_total = 0
                for i in range(sample_scores.shape[0]):
                    for p in range(3):
                        for n in range(3, 6):
                            if sample_scores[i, p] > sample_scores[i, n]:
                                train_correct += 1
                            elif sample_scores[i, p] == sample_scores[i, n]:
                                train_correct += 0.5
                            train_total += 1
                train_acc = train_correct / train_total

            val_auc = compute_val_auc(model, val_X, val_R)
            improved = ""
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_counter = 0
                improved = " *BEST*"
            else:
                patience_counter += 5

            print(f"    Epoch {epoch+1:3d}/{MAX_EPOCHS}  "
                  f"loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  "
                  f"val_AUC={val_auc:.4f}{improved}")

            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    print(f"    Best val AUC: {best_auc:.4f}")
    return model, best_auc


# =====================================================================
# 13. Ensemble Prediction (#7)
# =====================================================================

def ensemble_predict(models, user_ratings, track_meta, test_users,
                     feat_mean, feat_std, feature_indices,
                     global_stats, album_to_tracks, artist_to_tracks,
                     mf_models, neighbor_index,
                     out_path, model_name="ensemble"):
    print(f"\n>>> Predicting with {model_name} ({len(models)} models) ...")
    results = []

    with torch.no_grad():
        for uid, candidates in test_users:
            ratings = user_ratings.get(uid, {})
            u_mean, u_std, u_n = compute_user_profile(ratings)

            base_feats = []
            for tid in candidates:
                bf = compute_base_features(
                    ratings, track_meta, tid, u_mean, u_std, u_n,
                    global_stats, album_to_tracks, artist_to_tracks,
                    mf_models, uid, neighbor_index, user_ratings)
                base_feats.append(bf)

            cross = compute_cross_features(track_meta, candidates)

            X = np.zeros((1, 6, N_FEATURES), dtype=np.float32)
            for c in range(6):
                X[0, c, :N_BASE_FEATURES] = base_feats[c]
                X[0, c, N_BASE_FEATURES:] = cross[c]

            X = normalize(X, feat_mean, feat_std)
            if feature_indices is not None:
                X = apply_feature_mask(X, feature_indices)

            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)

            # Average scores across ensemble
            avg_scores = np.zeros(6)
            for model in models:
                model.eval()
                scores = model(X_t).cpu().numpy()[0]
                avg_scores += scores
            avg_scores /= len(models)

            ranked = sorted(zip(avg_scores, candidates), reverse=True)
            top3 = set(tid for _, tid in ranked[:3])
            for tid in candidates:
                results.append((f"{uid}_{tid}", 1 if tid in top3 else 0))

    with open(out_path, "w") as f:
        f.write("TrackID,Predictor\n")
        for key, label in results:
            f.write(f"{key},{label}\n")

    ones = sum(v for _, v in results)
    print(f"    {len(results):,} predictions ({ones:,} rec, "
          f"{len(results)-ones:,} don't)")
    print(f"    Saved -> {out_path}")
    return results


# =====================================================================
# 14. V2 Baseline
# =====================================================================

def v2_heuristic_val_auc(user_ratings, track_meta, val_groups):
    correct = 0
    total = 0
    for uid, tids, rats in val_groups:
        ratings = user_ratings[uid]
        scores = []
        for tid in tids:
            alb_id, art_id, genre_ids = track_meta.get(tid, (None, None, []))
            s = 0.0
            if alb_id is not None and alb_id in ratings:
                s += ratings[alb_id]
            if art_id is not None and art_id in ratings:
                s += ratings[art_id]
            gs = [ratings[g] for g in genre_ids if g in ratings]
            if gs:
                s += len(gs) * (sum(gs) / len(gs)) / 10.0
            scores.append(s)
        for p in range(3):
            for n in range(3, 6):
                if scores[p] > scores[n]:
                    correct += 1
                elif scores[p] == scores[n]:
                    correct += 0.5
                total += 1
    return correct / total if total > 0 else 0.0


# =====================================================================
# 15. Main
# =====================================================================

def main():
    tee = Tee(RESULTS_FILE)
    sys.stdout = tee
    t0 = time.time()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("  EE627 Midterm v6 - Refined Model")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- Load data ----
    user_ratings = parse_training(os.path.join(DATA_DIR, "trainItem2.txt"))
    track_meta = parse_tracks(os.path.join(DATA_DIR, "trackData2.txt"))
    album_meta = parse_albums(os.path.join(DATA_DIR, "albumData2.txt"))
    test_users = parse_test(os.path.join(DATA_DIR, "testItem2.txt"))

    # ---- Enrich ----
    print("\n" + "=" * 60)
    print("  ENRICHING TRACKS")
    print("=" * 60)
    enriched_meta = enrich_tracks_with_albums(track_meta, album_meta)

    # ---- Auxiliary structures ----
    print("\n" + "=" * 60)
    print("  BUILDING AUXILIARY STRUCTURES")
    print("=" * 60)
    global_stats = build_global_item_stats(user_ratings)
    album_to_tracks, artist_to_tracks = build_track_mappings(enriched_meta)

    # ---- (#1, #3) Multiple MF models ----
    print("\n" + "=" * 60)
    print("  TRAINING MF MODELS")
    print("=" * 60)
    mf_models = []
    for cfg in MF_CONFIGS:
        result = train_mf(user_ratings, cfg["factors"], cfg["epochs"],
                          cfg["lr"], MF_BATCH, cfg["label"])
        mf_models.append(result)

    # ---- (#5) Neighbor index from larger MF ----
    print("\n" + "=" * 60)
    print("  BUILDING NEIGHBOR INDEX")
    print("=" * 60)
    # Use the larger MF (mf128) for neighbor similarity
    mf128_uid_map, _, mf128_user_f, _ = mf_models[1]
    neighbor_index = build_neighbor_index(
        mf128_uid_map, mf128_user_f, user_ratings, N_NEIGHBORS)

    # ---- Split users ----
    print("\n" + "=" * 60)
    print("  BUILDING TRAINING & VALIDATION DATA")
    print("=" * 60)
    rng = random.Random(SEED)
    all_uids = list(user_ratings.keys())
    rng.shuffle(all_uids)
    split = int(len(all_uids) * 0.9)
    train_uids = set(all_uids[:split])
    val_uids = set(all_uids[split:])
    print(f"  Train users: {len(train_uids):,}, Val users: {len(val_uids):,}")

    t1 = time.time()
    train_groups = build_groups(user_ratings, enriched_meta, train_uids,
                                GROUPS_PER_USER, SEED)
    val_groups = build_groups(user_ratings, enriched_meta, val_uids,
                              10, SEED + 1)
    print(f"  Train groups: {len(train_groups):,}, "
          f"Val groups: {len(val_groups):,}")

    print("\n  Featurizing training groups ...")
    train_X, train_R = featurize_groups(
        train_groups, user_ratings, enriched_meta, global_stats,
        album_to_tracks, artist_to_tracks, mf_models, neighbor_index)

    print("  Featurizing validation groups ...")
    val_X, val_R = featurize_groups(
        val_groups, user_ratings, enriched_meta, global_stats,
        album_to_tracks, artist_to_tracks, mf_models, neighbor_index)

    t2 = time.time()
    print(f"  Feature computation: {t2-t1:.1f}s")

    # Normalize
    feat_mean, feat_std = compute_norm_stats(train_X)
    train_X_norm = normalize(train_X, feat_mean, feat_std)
    val_X_norm = normalize(val_X, feat_mean, feat_std)

    # ---- V2 baseline ----
    print("\n" + "=" * 60)
    print("  V2 HEURISTIC BASELINE")
    print("=" * 60)
    v2_auc = v2_heuristic_val_auc(user_ratings, enriched_meta, val_groups)
    print(f"  v2 heuristic val AUC: {v2_auc:.4f}  (Kaggle: 0.871)")

    # ---- Phase 1: Train initial model for feature importance ----
    print("\n" + "=" * 60)
    print("  PHASE 1: INITIAL MODEL (all {N_FEATURES} features)")
    print("=" * 60)
    model_full, full_auc = train_ranking_model(
        train_X_norm, train_R, val_X_norm, val_R, N_FEATURES)

    # ---- (#2) Feature selection ----
    print("\n" + "=" * 60)
    print("  FEATURE SELECTION")
    print("=" * 60)
    importances = feature_importance(model_full, val_X_norm, val_R)
    selected = select_features(None, importances)
    print(f"\n  Selected {len(selected)}/{N_FEATURES} features: {selected}")

    # Apply mask
    train_X_sel = apply_feature_mask(train_X_norm, selected)
    val_X_sel = apply_feature_mask(val_X_norm, selected)
    n_sel = len(selected)

    # ---- Phase 2: Retrain with selected features ----
    print("\n" + "=" * 60)
    print(f"  PHASE 2: RETRAIN WITH {n_sel} SELECTED FEATURES")
    print("=" * 60)
    model_sel, sel_auc = train_ranking_model(
        train_X_sel, train_R, val_X_sel, val_R, n_sel)

    print(f"\n  Full model AUC:     {full_auc:.4f}")
    print(f"  Selected model AUC: {sel_auc:.4f}")

    # Decide which features to use
    if sel_auc >= full_auc - 0.002:
        use_selected = True
        final_features = selected
        print(f"  Using selected features ({n_sel})")
    else:
        use_selected = False
        final_features = None
        print(f"  Keeping all features ({N_FEATURES})")

    # ---- (#7) Ensemble: train N models with different seeds ----
    print("\n" + "=" * 60)
    print(f"  ENSEMBLE: TRAINING {N_ENSEMBLE} MODELS")
    print("=" * 60)

    if use_selected:
        ens_train_X = train_X_sel
        ens_val_X = val_X_sel
        ens_n_feat = n_sel
    else:
        ens_train_X = train_X_norm
        ens_val_X = val_X_norm
        ens_n_feat = N_FEATURES

    ensemble_models = []
    ensemble_aucs = []
    for i in range(N_ENSEMBLE):
        print(f"\n  --- Ensemble model {i+1}/{N_ENSEMBLE} (seed offset={i*100}) ---")
        model_i, auc_i = train_ranking_model(
            ens_train_X, train_R, ens_val_X, val_R, ens_n_feat,
            seed_offset=i * 100)
        ensemble_models.append(model_i)
        ensemble_aucs.append(auc_i)
        print(f"    Model {i+1} val AUC: {auc_i:.4f}")

    print(f"\n  Ensemble individual AUCs: {ensemble_aucs}")
    print(f"  Mean: {np.mean(ensemble_aucs):.4f}")

    # ---- Generate submissions ----
    print("\n" + "=" * 60)
    print("  GENERATING SUBMISSIONS")
    print("=" * 60)

    v6_results = ensemble_predict(
        ensemble_models, user_ratings, enriched_meta, test_users,
        feat_mean, feat_std, final_features,
        global_stats, album_to_tracks, artist_to_tracks,
        mf_models, neighbor_index,
        os.path.join(OUTPUT_DIR, "submission_v6_ensemble.csv"),
        f"v6 Ensemble ({N_ENSEMBLE} models)")

    # ---- Compute v2 scores per user for hybrid submissions ----
    print("\n" + "=" * 60)
    print("  HYBRID SUBMISSIONS (multiple thresholds)")
    print("=" * 60)

    # Build v6 results lookup by user
    v6_by_user = {}
    idx = 0
    for uid, candidates in test_users:
        v6_by_user[uid] = {}
        for tid in candidates:
            v6_by_user[uid][tid] = v6_results[idx][1]
            idx += 1

    # Compute v2 scores and predictions per user
    v2_user_data = {}  # uid -> (v2_top3_set, confidence_gap, v2_results_list)
    v2_results = []
    for uid, candidates in test_users:
        ratings = user_ratings.get(uid, {})
        v2_scores = []
        for tid in candidates:
            alb_id, art_id, genre_ids = enriched_meta.get(
                tid, (None, None, []))
            s = 0.0
            if alb_id is not None and alb_id in ratings:
                s += ratings[alb_id]
            if art_id is not None and art_id in ratings:
                s += ratings[art_id]
            gs = [ratings[g] for g in genre_ids if g in ratings]
            if gs:
                s += len(gs) * (sum(gs) / len(gs)) / 10.0
            v2_scores.append(s)

        ranked = sorted(zip(v2_scores, candidates), reverse=True)
        v2_top3 = set(tid for _, tid in ranked[:3])
        sorted_scores = sorted(v2_scores, reverse=True)
        gap = sorted_scores[2] - sorted_scores[3]
        v2_user_data[uid] = (v2_top3, gap)

        for tid in candidates:
            v2_results.append((f"{uid}_{tid}", 1 if tid in v2_top3 else 0))

    # Pure v6 agreement with v2
    agree = sum(1 for (_, a), (_, b) in zip(v6_results, v2_results) if a == b)
    total_preds = len(v6_results)
    agree_pct = 100 * agree / total_preds
    print(f"\n  Pure v6 vs v2 agreement: {agree:,}/{total_preds:,} ({agree_pct:.1f}%)")

    # Generate hybrids at multiple thresholds
    # v5b hybrid with gap>5 scored 0.908, so test around that
    thresholds = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

    print(f"\n  {'Threshold':>10}  {'v2 users':>10}  {'v6 users':>10}  "
          f"{'v6 pct':>8}  {'Agreement':>10}  File")
    print("  " + "-" * 75)

    for thresh in thresholds:
        hybrid = []
        n_v2 = 0
        n_v6 = 0

        for uid, candidates in test_users:
            v2_top3, gap = v2_user_data[uid]

            if gap > thresh:
                # v2 confident -- keep v2
                n_v2 += 1
                for tid in candidates:
                    hybrid.append(
                        (f"{uid}_{tid}", 1 if tid in v2_top3 else 0))
            else:
                # v2 uncertain -- use v6 model
                n_v6 += 1
                for tid in candidates:
                    hybrid.append(
                        (f"{uid}_{tid}", v6_by_user[uid][tid]))

        # Write submission
        fname = f"submission_v6_hybrid_gap{thresh}.csv"
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w") as f:
            f.write("TrackID,Predictor\n")
            for key, label in hybrid:
                f.write(f"{key},{label}\n")

        # Agreement with v2
        h_agree = sum(1 for (_, a), (_, b) in zip(hybrid, v2_results)
                      if a == b)
        h_agree_pct = 100 * h_agree / total_preds
        v6_pct = 100 * n_v6 / len(test_users)

        print(f"  {thresh:>10}  {n_v2:>10,}  {n_v6:>10,}  "
              f"{v6_pct:>7.1f}%  {h_agree_pct:>9.1f}%  {fname}")

    # ---- Summary ----
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  v2 heuristic val AUC:  {v2_auc:.4f}  (Kaggle: 0.871)")
    print(f"  v5b hybrid (gap>5):    0.908    (Kaggle best so far)")
    print(f"  v6 full model AUC:     {full_auc:.4f}")
    print(f"  v6 selected model AUC: {sel_auc:.4f}")
    print(f"  v6 ensemble AUCs:      {ensemble_aucs}")
    print(f"  Features used:         {ens_n_feat} "
          f"({'selected' if use_selected else 'all'})")
    print(f"  Pure v6 agreement:     {agree_pct:.1f}%")
    print(f"  MF models:             {[c['label'] for c in MF_CONFIGS]}")
    print(f"  Neighbors:             {N_NEIGHBORS}")
    print(f"  Ensemble size:         {N_ENSEMBLE}")
    print()
    print("  Submissions generated:")
    print(f"    submission_v6_ensemble.csv         -- pure v6 model")
    for thresh in thresholds:
        print(f"    submission_v6_hybrid_gap{thresh}.csv"
              f"{' ':>{4-len(str(thresh))}}-- v2 if gap>{thresh}, else v6")
    print()
    print(f"  Best bet: start with gap5 (matched v5b's 0.908 threshold)")
    print(f"  Then try gap3 and gap8 to find the sweet spot")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print("\nDone.")

    tee.close()


if __name__ == "__main__":
    main()

"""
EE627A Midterm v5 - Full Model with All Improvements
Jude Eschete

10 improvements over v2 heuristic (0.871) and v4 collapsed model:

  1. Train on groups of 6 (mimics test scenario exactly)
  2. BPR pairwise loss (stable, no collapse like MarginRankingLoss)
  3. Heavy regularization (small net, dropout, weight decay, early stop)
  4. Feature normalization (StandardScaler on inputs)
  5. Collaborative filtering features (sibling track ratings)
  6. Popularity/rarity features (n_raters, global avg per item)
  7. Cross-candidate features (shared artist/album/genre within group)
  8. Global item statistics as features (not imputation)
  9. Hard negative weighting (close-rating pairs weighted more)
  10. Matrix factorization embeddings (SGD on GPU, dot product feature)

Architecture:
  - 32 features per (user, track, group) pair
  - Small MLP: 32 -> 64 -> 32 -> 1 with BatchNorm + Dropout
  - BPR loss on (top-3, bottom-3) pairs within each 6-candidate group
  - Early stopping on held-out validation AUC

Uses RTX 5080 GPU for MF training and ranking model training.
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

# Hyperparameters (v5b: tuned up from v5a which scored 0.878)
MF_FACTORS = 64         # was 32 -- richer embeddings
MF_EPOCHS = 40          # was 20 -- let MF converge more
MF_LR = 0.005           # was 0.01 -- slower for stability with more epochs
MF_BATCH = 65536

GROUPS_PER_USER = 30    # was 15 -- more training data
HIDDEN_SIZES = (128, 64, 32)  # was (64, 32) -- deeper network
DROPOUT = 0.3
LR = 5e-4               # was 1e-3 -- slower learning for deeper net
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 2048
MAX_EPOCHS = 150         # was 100 -- more room with early stopping
PATIENCE = 15            # was 10 -- more patience for deeper net
HARD_NEG_TEMP = 15.0     # was 20 -- focus more on hard pairs

N_BASE_FEATURES = 29   # per (user, track)
N_CROSS_FEATURES = 3   # per (user, track, group)
N_FEATURES = N_BASE_FEATURES + N_CROSS_FEATURES  # 32 total


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
    """Per-item: (mean_rating, n_raters)."""
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
    """album_id -> [track_ids], artist_id -> [track_ids]."""
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


def train_mf_embeddings(user_ratings, n_factors=32, n_epochs=20, lr=0.01,
                         batch_size=65536):
    """Train matrix factorization via SGD on GPU (#10)."""
    print(f"\n  Training MF embeddings (k={n_factors}) on {DEVICE} ...")

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
    print(f"       {n_users:,} users, {n_items:,} items, "
          f"{len(r_list):,} ratings")

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
        lr=lr, weight_decay=1e-4
    )

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
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    MF Epoch {epoch+1:3d}/{n_epochs}  "
                  f"MSE={total_loss/n_b:.2f}")

    uf = user_emb.weight.detach().cpu().numpy()
    itf = item_emb.weight.detach().cpu().numpy()
    print(f"       MF done. User factors: {uf.shape}, Item factors: {itf.shape}")
    return uid_to_idx, iid_to_idx, uf, itf


# =====================================================================
# 4. Feature Engineering
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
                           mf_uid_map, mf_iid_map, mf_user_f, mf_item_f,
                           uid):
    """29 base features per (user, track) pair."""
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

    # --- (#6) Popularity & global avg ---
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

    # --- (#5) CF sibling features ---
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

    # --- (#10) MF dot product ---
    mf_score = 0.0
    if uid in mf_uid_map and tid in mf_iid_map:
        mf_score = float(
            mf_user_f[mf_uid_map[uid]] @ mf_item_f[mf_iid_map[tid]])

    # --- v2 heuristic score as feature ---
    v2_score = (has_album * album_score + has_artist * artist_score
                + gc * g_mean / 10.0)

    return [
        album_score, has_album,                         # 0-1
        artist_score, has_artist,                       # 2-3
        float(gc), g_max, g_min, g_mean, g_var,        # 4-8
        evidence_count,                                 # 9
        aa_int, ga_int, ag_int,                         # 10-12
        u_mean, u_std, math.log1p(u_n),                # 13-15
        album_above, artist_above, genre_above,         # 16-18
        alb_gavg, art_gavg, avg_genre_gavg,             # 19-21  (#6)
        math.log1p(alb_nrat), math.log1p(art_nrat),    # 22-23  (#6)
        math.log1p(avg_genre_nrat),                     # 24     (#6)
        cf_score, math.log1p(cf_count),                 # 25-26  (#5)
        mf_score,                                       # 27     (#10)
        v2_score,                                       # 28     (baseline feature)
    ]


def compute_cross_features(track_meta, candidates):
    """3 cross-candidate features per candidate in a group of 6 (#7)."""
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


# =====================================================================
# 5. Training Data Construction
# =====================================================================

def build_groups(user_ratings, track_meta, user_set, groups_per_user, seed):
    """Build groups of 6 tracks per user, sorted by rating desc (#1).
    Returns [(uid, [6 tids], [6 ratings])] with top-3 first."""
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
                     mf_uid_map, mf_iid_map, mf_user_f, mf_item_f):
    """Compute feature matrices for groups.
    Returns X (n_groups, 6, N_FEATURES), ratings (n_groups, 6)."""
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
                mf_uid_map, mf_iid_map, mf_user_f, mf_item_f, uid)
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
# 6. Feature Normalization (#4)
# =====================================================================

def compute_norm_stats(X):
    """Compute per-feature mean and std from (n_groups, 6, n_feat)."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std


# =====================================================================
# 7. Model (#3)
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
        # x: (batch, 6, n_features) -> score each candidate
        batch, n_cand, n_feat = x.shape
        flat = x.reshape(batch * n_cand, n_feat)
        scores = self.net(flat)  # (batch*6, 1)
        return scores.reshape(batch, n_cand)


# =====================================================================
# 8. Loss Function (#2, #9)
# =====================================================================

def bpr_group_loss(scores, ratings, hard_neg_temp=20.0):
    """BPR loss over (top-3, bottom-3) pairs within each group.

    scores: (batch, 6) -- model predictions, positions 0-2 are top-3
    ratings: (batch, 6) -- true ratings for hard negative weighting
    """
    pos_scores = scores[:, :3]   # (batch, 3)
    neg_scores = scores[:, 3:]   # (batch, 3)
    pos_ratings = ratings[:, :3]
    neg_ratings = ratings[:, 3:]

    # All 9 pairs per group: pos_i vs neg_j
    pos_exp = pos_scores.unsqueeze(2)   # (batch, 3, 1)
    neg_exp = neg_scores.unsqueeze(1)   # (batch, 1, 3)
    diff = pos_exp - neg_exp            # (batch, 3, 3)

    # (#9) Hard negative weighting: pairs with close ratings get more weight
    pr_exp = pos_ratings.unsqueeze(2)
    nr_exp = neg_ratings.unsqueeze(1)
    rating_gap = (pr_exp - nr_exp).abs()
    weight = torch.exp(-rating_gap / hard_neg_temp)

    pair_loss = -torch.log(torch.sigmoid(diff) + 1e-8) * weight
    return pair_loss.mean()


# =====================================================================
# 9. Training Loop
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
    """AUC on validation groups: does model rank top-3 above bottom-3?"""
    model.eval()
    dataset = GroupDataset(val_X, val_R)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, r_batch in loader:
            x_batch = x_batch.to(DEVICE)
            scores = model(x_batch).cpu().numpy()  # (batch, 6)

            for i in range(scores.shape[0]):
                # Positions 0-2 are top-3 by rating, 3-5 are bottom-3
                for p in range(3):
                    for n in range(3, 6):
                        if scores[i, p] > scores[i, n]:
                            correct += 1
                        elif scores[i, p] == scores[i, n]:
                            correct += 0.5
                        total += 1

    return correct / total if total > 0 else 0.0


def train_ranking_model(train_X, train_R, val_X, val_R):
    """Train with BPR loss + early stopping on val AUC."""
    print(f"\n  Training ranking model on {DEVICE} ...")
    print(f"    Train: {train_X.shape[0]:,} groups, "
          f"Val: {val_X.shape[0]:,} groups")

    dataset = GroupDataset(train_X, train_R)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0,
                        pin_memory=(DEVICE.type == "cuda"))

    model = RankingNet(N_FEATURES, HIDDEN_SIZES, DROPOUT).to(DEVICE)
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

        # Evaluate every 5 epochs or first epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Training pair accuracy
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

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    print(f"    Best val AUC: {best_auc:.4f}")
    return model, best_auc


# =====================================================================
# 10. Prediction & Submission
# =====================================================================

def predict_test(model, user_ratings, track_meta, test_users,
                 feat_mean, feat_std,
                 global_stats, album_to_tracks, artist_to_tracks,
                 mf_uid_map, mf_iid_map, mf_user_f, mf_item_f,
                 out_path, model_name="model"):
    """Score test candidates and generate submission."""
    print(f"\n>>> Predicting with {model_name} ...")
    model.eval()
    results = []

    with torch.no_grad():
        for uid, candidates in test_users:
            ratings = user_ratings.get(uid, {})
            u_mean, u_std, u_n = compute_user_profile(ratings)

            # Base features
            base_feats = []
            for tid in candidates:
                bf = compute_base_features(
                    ratings, track_meta, tid, u_mean, u_std, u_n,
                    global_stats, album_to_tracks, artist_to_tracks,
                    mf_uid_map, mf_iid_map, mf_user_f, mf_item_f, uid)
                base_feats.append(bf)

            # Cross features
            cross = compute_cross_features(track_meta, candidates)

            # Combine
            X = np.zeros((1, 6, N_FEATURES), dtype=np.float32)
            for c in range(6):
                X[0, c, :N_BASE_FEATURES] = base_feats[c]
                X[0, c, N_BASE_FEATURES:] = cross[c]

            # Normalize
            X = normalize(X, feat_mean, feat_std)

            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            scores = model(X_t).cpu().numpy()[0]  # (6,)

            ranked = sorted(zip(scores, candidates), reverse=True)
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
# 11. V2 Heuristic Baseline (for comparison)
# =====================================================================

def v2_heuristic_val_auc(user_ratings, track_meta, val_groups):
    """Evaluate v2 heuristic on the same validation groups."""
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
# 12. Main
# =====================================================================

def main():
    tee = Tee(RESULTS_FILE)
    sys.stdout = tee
    t0 = time.time()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("  EE627 Midterm v5 - Full Model")
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

    # (#10) Matrix factorization
    mf_uid_map, mf_iid_map, mf_user_f, mf_item_f = train_mf_embeddings(
        user_ratings, MF_FACTORS, MF_EPOCHS, MF_LR, MF_BATCH)

    # ---- Split users for train/val ----
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

    # (#1) Build groups of 6
    t1 = time.time()
    train_groups = build_groups(user_ratings, enriched_meta, train_uids,
                                GROUPS_PER_USER, SEED)
    val_groups = build_groups(user_ratings, enriched_meta, val_uids,
                              10, SEED + 1)
    print(f"  Train groups: {len(train_groups):,}, "
          f"Val groups: {len(val_groups):,}")

    # Featurize
    print("\n  Featurizing training groups ...")
    train_X, train_R = featurize_groups(
        train_groups, user_ratings, enriched_meta, global_stats,
        album_to_tracks, artist_to_tracks,
        mf_uid_map, mf_iid_map, mf_user_f, mf_item_f)

    print("  Featurizing validation groups ...")
    val_X, val_R = featurize_groups(
        val_groups, user_ratings, enriched_meta, global_stats,
        album_to_tracks, artist_to_tracks,
        mf_uid_map, mf_iid_map, mf_user_f, mf_item_f)

    t2 = time.time()
    print(f"  Feature computation: {t2-t1:.1f}s")

    # (#4) Normalize
    feat_mean, feat_std = compute_norm_stats(train_X)
    train_X = normalize(train_X, feat_mean, feat_std)
    val_X = normalize(val_X, feat_mean, feat_std)

    # ---- V2 baseline on same val groups ----
    print("\n" + "=" * 60)
    print("  V2 HEURISTIC BASELINE")
    print("=" * 60)
    v2_auc = v2_heuristic_val_auc(user_ratings, enriched_meta, val_groups)
    print(f"  v2 heuristic val AUC: {v2_auc:.4f}  (Kaggle: 0.871)")

    # ---- Train ----
    print("\n" + "=" * 60)
    print("  TRAINING RANKING MODEL")
    print("=" * 60)
    model, best_auc = train_ranking_model(train_X, train_R, val_X, val_R)

    # ---- Generate submissions ----
    print("\n" + "=" * 60)
    print("  GENERATING SUBMISSIONS")
    print("=" * 60)

    v5_results = predict_test(
        model, user_ratings, enriched_meta, test_users,
        feat_mean, feat_std,
        global_stats, album_to_tracks, artist_to_tracks,
        mf_uid_map, mf_iid_map, mf_user_f, mf_item_f,
        os.path.join(OUTPUT_DIR, "submission_v5b_nn.csv"),
        "v5 Ranking NN")

    # ---- Generate v2 baseline predictions + confidence-gated hybrid ----
    print("\n" + "=" * 60)
    print("  V2 vs V5 AGREEMENT & HYBRID SUBMISSION")
    print("=" * 60)

    v2_results = []
    hybrid_results = []  # use v2 when confident, v5 when v2 is uncertain

    # Build v5 results lookup by user for hybrid
    v5_by_user = {}  # uid -> {tid: label}
    idx = 0
    for uid, candidates in test_users:
        v5_by_user[uid] = {}
        for tid in candidates:
            v5_by_user[uid][tid] = v5_results[idx][1]
            idx += 1

    for uid, candidates in test_users:
        ratings = user_ratings.get(uid, {})

        # Compute v2 scores for this user's 6 candidates
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

        # v2 predictions
        ranked = sorted(zip(v2_scores, candidates), reverse=True)
        v2_top3 = set(tid for _, tid in ranked[:3])
        for tid in candidates:
            v2_results.append((f"{uid}_{tid}", 1 if tid in v2_top3 else 0))

        # Confidence = gap between 3rd and 4th ranked candidate
        sorted_scores = sorted(v2_scores, reverse=True)
        confidence_gap = sorted_scores[2] - sorted_scores[3]

        # If v2 is confident (clear gap), keep v2. Otherwise use v5.
        # Threshold tuned: gap=0 means all tied, gap>10 means clear winner
        if confidence_gap > 5.0:
            # v2 is confident -- trust it
            for tid in candidates:
                hybrid_results.append(
                    (f"{uid}_{tid}", 1 if tid in v2_top3 else 0))
        else:
            # v2 is uncertain -- let v5 model decide
            for tid in candidates:
                hybrid_results.append(
                    (f"{uid}_{tid}", v5_by_user[uid][tid]))

    # Write hybrid submission
    hybrid_path = os.path.join(OUTPUT_DIR, "submission_v5b_hybrid.csv")
    with open(hybrid_path, "w") as f:
        f.write("TrackID,Predictor\n")
        for key, label in hybrid_results:
            f.write(f"{key},{label}\n")
    print(f"  Hybrid submission saved -> {hybrid_path}")

    # Count how many users fall into each bucket
    n_v2_confident = 0
    n_v5_override = 0
    for uid, candidates in test_users:
        v2_scores_quick = []
        ratings = user_ratings.get(uid, {})
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
            v2_scores_quick.append(s)
        ss = sorted(v2_scores_quick, reverse=True)
        if ss[2] - ss[3] > 5.0:
            n_v2_confident += 1
        else:
            n_v5_override += 1

    print(f"\n  Hybrid breakdown:")
    print(f"    v2 confident (gap>5):  {n_v2_confident:,} users "
          f"({100*n_v2_confident/len(test_users):.1f}%) -- kept v2")
    print(f"    v2 uncertain (gap<=5): {n_v5_override:,} users "
          f"({100*n_v5_override/len(test_users):.1f}%) -- used v5 model")

    # Compare v5 vs v2
    agree = sum(1 for (_, a), (_, b) in zip(v5_results, v2_results) if a == b)
    total_preds = len(v5_results)
    agree_pct = 100 * agree / total_preds

    v5_promotes = sum(1 for (_, v5), (_, v2) in zip(v5_results, v2_results)
                      if v5 == 1 and v2 == 0)
    v5_demotes = sum(1 for (_, v5), (_, v2) in zip(v5_results, v2_results)
                     if v5 == 0 and v2 == 1)

    # Compare hybrid vs v2
    h_agree = sum(1 for (_, a), (_, b) in zip(hybrid_results, v2_results)
                  if a == b)
    h_agree_pct = 100 * h_agree / total_preds

    print(f"\n  v5 vs v2:")
    print(f"    Agreement:   {agree:,}/{total_preds:,} ({agree_pct:.1f}%)")
    print(f"    Promotes:    {v5_promotes:,}  Demotes: {v5_demotes:,}")

    print(f"\n  hybrid vs v2:")
    print(f"    Agreement:   {h_agree:,}/{total_preds:,} ({h_agree_pct:.1f}%)")

    print()
    if agree_pct >= 95:
        print("  VERDICT v5: Very close to v2 -- safe to submit")
    elif agree_pct >= 90:
        print("  VERDICT v5: Moderate divergence -- could go either way")
    elif agree_pct >= 80:
        print("  VERDICT v5: Significant divergence -- risky")
    else:
        print("  VERDICT v5: Major divergence -- likely worse than v2")

    print(f"  VERDICT hybrid: {h_agree_pct:.1f}% agreement -- "
          f"safest bet, changes only uncertain users")

    # ---- Summary ----
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  v2 heuristic val AUC:  {v2_auc:.4f}  (Kaggle: 0.871)")
    print(f"  v5 ranking NN val AUC: {best_auc:.4f}")
    print(f"  Delta:                 {best_auc - v2_auc:+.4f}")
    print(f"  Agreement with v2:     {agree_pct:.1f}%")
    print()
    print(f"  Features: {N_FEATURES}")
    print(f"  Training groups: {len(train_groups):,}")
    print(f"  Validation groups: {len(val_groups):,}")
    print(f"  MF factors: {MF_FACTORS}")
    print()
    print(f"  v5 agreement with v2:  {agree_pct:.1f}%")
    print(f"  hybrid agreement:      {h_agree_pct:.1f}%")
    print()
    print(f"  Submissions:")
    print(f"    submission_v5b_nn.csv      -- pure model")
    print(f"    submission_v5b_hybrid.csv  -- v2 when confident, v5 when uncertain")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print("\nDone.")

    tee.close()


if __name__ == "__main__":
    main()

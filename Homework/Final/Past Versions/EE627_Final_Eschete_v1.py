"""
EE627A Final Project - Music Recommender (v5 BPR + MF Hybrid)
Jude Eschete

This is the production model that scored 0.911 on the midterm Kaggle.
Reused for the Final since the underlying competition is identical.

Approach (10 improvements over a v2 heuristic baseline):
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
  - MLP: 32 -> 128 -> 64 -> 32 -> 1 with BatchNorm + Dropout
  - BPR loss on (top-3, bottom-3) pairs within each 6-candidate group
  - Early stopping on held-out validation AUC
  - Confidence-gated hybrid: v2 heuristic when gap >= 10, neural net otherwise

Outputs:
  - submission_final_v5_nn.csv          (pure neural network)
  - submission_final_v2.csv             (v2 heuristic baseline)
  - submission_final_v5_hybrid_gap{N}.csv  (hybrid sweep)
    -> gap10 is the production submission (0.911 on midterm)

Uses CUDA GPU if available, falls back to CPU.
"""

import os
import sys
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# =====================================================================
# Configuration
# =====================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "Data", "music-recommender-2026s")
OUTPUT_DIR   = SCRIPT_DIR
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results_final.txt")
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Meta-learner inputs (optional -- loaded if present).
# We prefer the in-Final copy of hw9_probabilities.csv so the project folder is
# self-contained, but fall back to the original in Homework 9 if the local copy
# is missing.
LABEL_FILE       = os.path.join(SCRIPT_DIR, "test2_new.txt")
_HW9_LOCAL       = os.path.join(SCRIPT_DIR, "hw9_probabilities.csv")
_HW9_REMOTE      = os.path.join(SCRIPT_DIR, "..", "Homework 9",
                                 "hw9_probabilities.csv")
HW9_PROBS_FILE   = _HW9_LOCAL if os.path.exists(_HW9_LOCAL) else _HW9_REMOTE
PAST_PREDS_DIR   = os.path.join(SCRIPT_DIR, "past_predictions")

# Model hyperparameters (matching v5b which produced 0.911)
MF_FACTORS      = 64
MF_EPOCHS       = 40
MF_LR           = 0.005
MF_BATCH        = 65536

GROUPS_PER_USER = 30
HIDDEN_SIZES    = (128, 64, 32)
DROPOUT         = 0.3
LR              = 5e-4
WEIGHT_DECAY    = 1e-4
BATCH_SIZE      = 2048
MAX_EPOCHS      = 150
PATIENCE        = 15
HARD_NEG_TEMP   = 15.0

N_BASE_FEATURES  = 29
N_CROSS_FEATURES = 3
N_FEATURES       = N_BASE_FEATURES + N_CROSS_FEATURES   # 32 total

# Hybrid threshold sweep -- gap10 scored 0.911 on midterm
HYBRID_THRESHOLDS = [0, 3, 5, 7, 8, 10, 12, 15, 20]

# =====================================================================
# Cache paths -- delete these to force a rebuild of that stage.
# All caches sit in a 'cache/' subdir so they're easy to wipe in one go.
# =====================================================================
CACHE_DIR        = os.path.join(SCRIPT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
MF_CACHE         = os.path.join(CACHE_DIR,
                                  f"mf_k{MF_FACTORS}_e{MF_EPOCHS}.npz")
FEATURES_CACHE   = os.path.join(CACHE_DIR,
                                  f"features_g{GROUPS_PER_USER}"
                                  f"_seed{SEED}_f{N_FEATURES}.npz")
BPR_MODEL_CACHE  = os.path.join(CACHE_DIR,
                                  f"bpr_h{'-'.join(map(str, HIDDEN_SIZES))}"
                                  f"_d{DROPOUT}_seed{SEED}.pt")


# =====================================================================
# Logging: tee stdout to results file
# =====================================================================

class Tee:
    def __init__(self, filepath):
        self.file   = open(filepath, "w", encoding="utf-8")
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


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def elapsed(t0):
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}m"


# =====================================================================
# 1. Data Parsing
# =====================================================================

def parse_training(path):
    print("Loading training data ...")
    t0 = time.time()
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
    print(f"  {len(users):,} users  |  {total:,} ratings  [{elapsed(t0)}]")
    return users


def parse_tracks(path):
    print("Loading track metadata ...")
    t0 = time.time()
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
    print(f"  {len(tracks):,} tracks  [{elapsed(t0)}]")
    return tracks


def parse_albums(path):
    print("Loading album metadata ...")
    t0 = time.time()
    albums = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 1:
                continue
            alb_id = int(parts[0])
            art    = (None if len(parts) < 2 or parts[1] == "None"
                      else int(parts[1]))
            genres = [int(g) for g in parts[2:] if g.strip() and g != "None"]
            albums[alb_id] = (art, genres)
    print(f"  {len(albums):,} albums  [{elapsed(t0)}]")
    return albums


def parse_test(path):
    print("Loading test candidates ...")
    t0 = time.time()
    test = []
    with open(path, "r") as f:
        uid  = None
        tids = []
        for line in f:
            line = line.strip()
            if "|" in line:
                if uid is not None:
                    test.append((uid, tids))
                uid  = int(line.split("|")[0])
                tids = []
            elif line:
                tids.append(int(line))
        if uid is not None:
            test.append((uid, tids))
    n_cand = sum(len(t) for _, t in test)
    print(f"  {len(test):,} test users  |  {n_cand:,} candidates  [{elapsed(t0)}]")
    return test


# =====================================================================
# 2. Album Enrichment -- fill missing artist/genre via album lookup
# =====================================================================

def enrich_tracks_with_albums(tracks, albums):
    print("Enriching tracks with album metadata ...")
    t0 = time.time()
    enriched = {}
    n_artist_filled = 0
    n_genres_added  = 0
    for tid, (alb, art, genres) in tracks.items():
        new_art    = art
        new_genres = list(genres)
        if alb is not None and alb in albums:
            album_art, album_genres = albums[alb]
            if art is None and album_art is not None:
                new_art = album_art
                n_artist_filled += 1
            track_genre_set = set(genres)
            for g in album_genres:
                if g not in track_genre_set:
                    new_genres.append(g)
                    track_genre_set.add(g)
                    n_genres_added += 1
        enriched[tid] = (alb, new_art, new_genres)
    print(f"  Artists filled: {n_artist_filled:,}  |  "
          f"genres added: {n_genres_added:,}  [{elapsed(t0)}]")
    return enriched


# =====================================================================
# 3. Auxiliary Data Structures
# =====================================================================

def build_global_item_stats(user_ratings):
    """Per-item: (mean_rating, n_raters)."""
    print("Building global item stats ...")
    t0 = time.time()
    item_sum = defaultdict(float)
    item_cnt = defaultdict(int)
    for ratings in user_ratings.values():
        for iid, r in ratings.items():
            item_sum[iid] += r
            item_cnt[iid] += 1
    stats = {iid: (item_sum[iid] / item_cnt[iid], item_cnt[iid])
             for iid in item_sum}
    print(f"  {len(stats):,} items with global stats  [{elapsed(t0)}]")
    return stats


def build_track_mappings(track_meta):
    """album_id -> [track_ids], artist_id -> [track_ids]."""
    print("Building track -> album/artist mappings ...")
    t0 = time.time()
    album_to_tracks  = defaultdict(list)
    artist_to_tracks = defaultdict(list)
    for tid, (alb, art, _) in track_meta.items():
        if alb is not None:
            album_to_tracks[alb].append(tid)
        if art is not None:
            artist_to_tracks[art].append(tid)
    print(f"  {len(album_to_tracks):,} albums  |  "
          f"{len(artist_to_tracks):,} artists  [{elapsed(t0)}]")
    return dict(album_to_tracks), dict(artist_to_tracks)


def train_mf_embeddings(user_ratings, n_factors=64, n_epochs=40, lr=0.005,
                         batch_size=65536):
    """Train matrix factorization via SGD on GPU (improvement #10).
    Caches the four output arrays to MF_CACHE; skips retraining on rerun."""
    print(f"\nTraining MF embeddings (k={n_factors}) on {DEVICE} ...")
    t0 = time.time()

    if os.path.exists(MF_CACHE):
        print(f"  Loading cached MF embeddings from {os.path.basename(MF_CACHE)} ...")
        z = np.load(MF_CACHE, allow_pickle=True)
        uid_to_idx = z["uid_to_idx"].item()
        iid_to_idx = z["iid_to_idx"].item()
        uf  = z["user_factors"]
        itf = z["item_factors"]
        print(f"  Loaded.  User factors: {uf.shape}  |  Item factors: {itf.shape}"
              f"  [{elapsed(t0)}]")
        return uid_to_idx, iid_to_idx, uf, itf

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
    print(f"  {n_users:,} users  |  {n_items:,} items  |  "
          f"{len(r_list):,} ratings")

    # Move the entire ratings matrix to GPU once. Shuffling and batch
    # slicing are done on-GPU per epoch -- no DataLoader, no per-batch
    # CPU->GPU transfer, no Python overhead.
    users_t   = torch.tensor(u_list, dtype=torch.long,    device=DEVICE)
    items_t   = torch.tensor(i_list, dtype=torch.long,    device=DEVICE)
    ratings_t = torch.tensor(r_list, dtype=torch.float32, device=DEVICE)
    n_total   = users_t.shape[0]

    user_emb = nn.Embedding(n_users, n_factors).to(DEVICE)
    item_emb = nn.Embedding(n_items, n_factors).to(DEVICE)
    nn.init.normal_(user_emb.weight, 0, 0.1)
    nn.init.normal_(item_emb.weight, 0, 0.1)

    optimizer = torch.optim.Adam(
        list(user_emb.parameters()) + list(item_emb.parameters()),
        lr=lr, weight_decay=1e-4
    )

    for epoch in range(n_epochs):
        # Per-epoch on-GPU shuffle (one tensor permutation, vs. 189 host
        # round-trips per epoch through DataLoader).
        perm = torch.randperm(n_total, device=DEVICE)
        total_loss = 0.0
        n_b = 0
        for start in range(0, n_total, batch_size):
            idx  = perm[start:start + batch_size]
            u_b  = users_t[idx]
            i_b  = items_t[idx]
            r_b  = ratings_t[idx]
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

    uf  = user_emb.weight.detach().cpu().numpy()
    itf = item_emb.weight.detach().cpu().numpy()
    print(f"  MF done in {elapsed(t0)}.  "
          f"User factors: {uf.shape}  |  Item factors: {itf.shape}")

    np.savez(MF_CACHE,
             uid_to_idx=np.array(uid_to_idx, dtype=object),
             iid_to_idx=np.array(iid_to_idx, dtype=object),
             user_factors=uf,
             item_factors=itf)
    print(f"  Cached -> {os.path.basename(MF_CACHE)}")
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
    var  = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    return mean, var ** 0.5, n


def compute_base_features(ratings, track_meta, tid, u_mean, u_std, u_n,
                           global_stats, album_to_tracks, artist_to_tracks,
                           mf_uid_map, mf_iid_map, mf_user_f, mf_item_f,
                           uid):
    """29 base features per (user, track) pair."""
    alb_id, art_id, genre_ids = track_meta.get(tid, (None, None, []))

    # --- Album signal ---
    if alb_id is not None and alb_id in ratings:
        album_score, has_album = float(ratings[alb_id]), 1.0
    else:
        album_score, has_album = 0.0, 0.0

    # --- Artist signal ---
    if art_id is not None and art_id in ratings:
        artist_score, has_artist = float(ratings[art_id]), 1.0
    else:
        artist_score, has_artist = 0.0, 0.0

    # --- Genre statistics ---
    genre_scores = [ratings[g] for g in genre_ids if g in ratings]
    gc = len(genre_scores)
    if gc > 0:
        g_max  = float(max(genre_scores))
        g_min  = float(min(genre_scores))
        g_mean = sum(genre_scores) / gc
        g_var  = sum((s - g_mean) ** 2 for s in genre_scores) / gc
    else:
        g_max = g_min = g_mean = g_var = 0.0

    evidence_count = has_album + has_artist + (1.0 if gc > 0 else 0.0)

    # --- Interaction terms ---
    aa_int = (album_score * artist_score / 100.0
              if (has_album and has_artist) else 0.0)
    ga_int = (g_mean * artist_score / 100.0
              if (has_artist and gc > 0) else 0.0)
    ag_int = (album_score * g_mean / 100.0
              if (has_album and gc > 0) else 0.0)

    # --- User-relative offsets ---
    album_above  = (album_score  - u_mean) if has_album  else 0.0
    artist_above = (artist_score - u_mean) if has_artist else 0.0
    genre_above  = (g_mean       - u_mean) if gc > 0     else 0.0

    # --- Global popularity (#6, #8) ---
    alb_gavg, alb_nrat = (global_stats.get(alb_id, (0.0, 0))
                          if alb_id else (0.0, 0))
    art_gavg, art_nrat = (global_stats.get(art_id, (0.0, 0))
                          if art_id else (0.0, 0))
    g_gavg_sum, g_nrat_sum, g_items = 0.0, 0.0, 0
    for gid in genre_ids:
        if gid in global_stats:
            gavg, nrat = global_stats[gid]
            g_gavg_sum += gavg
            g_nrat_sum += nrat
            g_items    += 1
    avg_genre_gavg = g_gavg_sum / g_items if g_items > 0 else 0.0
    avg_genre_nrat = g_nrat_sum / g_items if g_items > 0 else 0.0

    # --- CF sibling features (#5) ---
    sibling_ratings = []
    if alb_id is not None:
        for sib in album_to_tracks.get(alb_id, []):
            if sib != tid and sib in ratings:
                sibling_ratings.append(ratings[sib])
    if art_id is not None:
        album_sibs = (set(album_to_tracks.get(alb_id, []))
                      if alb_id else set())
        for sib in artist_to_tracks.get(art_id, []):
            if sib != tid and sib not in album_sibs and sib in ratings:
                sibling_ratings.append(ratings[sib])
    cf_score = (sum(sibling_ratings) / len(sibling_ratings)
                if sibling_ratings else 0.0)
    cf_count = len(sibling_ratings)

    # --- MF dot product (#10) ---
    mf_score = 0.0
    if uid in mf_uid_map and tid in mf_iid_map:
        mf_score = float(mf_user_f[mf_uid_map[uid]]
                         @ mf_item_f[mf_iid_map[tid]])

    # --- v2 heuristic score (kept as a feature) ---
    v2_score = (has_album * album_score
                + has_artist * artist_score
                + gc * g_mean / 10.0)

    return [
        album_score,   has_album,                        # 0-1
        artist_score,  has_artist,                       # 2-3
        float(gc), g_max, g_min, g_mean, g_var,          # 4-8
        evidence_count,                                  # 9
        aa_int, ga_int, ag_int,                          # 10-12
        u_mean, u_std, math.log1p(u_n),                  # 13-15
        album_above, artist_above, genre_above,          # 16-18
        alb_gavg,  art_gavg,  avg_genre_gavg,            # 19-21
        math.log1p(alb_nrat), math.log1p(art_nrat),      # 22-23
        math.log1p(avg_genre_nrat),                      # 24
        cf_score, math.log1p(cf_count),                  # 25-26
        mf_score,                                        # 27
        v2_score,                                        # 28
    ]


def compute_cross_features(track_meta, candidates):
    """3 cross-candidate features per candidate in a 6-group (#7)."""
    metas = [track_meta.get(tid, (None, None, [])) for tid in candidates]
    cross = []
    for i, (alb_i, art_i, genres_i) in enumerate(metas):
        n_same_art    = 0
        n_same_alb    = 0
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
        cross.append([float(n_same_art), float(n_same_alb),
                      float(genre_overlap)])
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
        user_tracks = [(iid, ratings[iid]) for iid in ratings
                       if iid in track_ids]
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
    """Compute (n_groups, 6, N_FEATURES) feature tensor + ratings."""
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
            print(f"    Featurized {g+1:,}/{n:,} groups ...")

    print(f"  Featurized {n:,} groups -> X{X.shape}")
    return X, R


# =====================================================================
# 6. Feature Normalization (#4)
# =====================================================================

def compute_norm_stats(X):
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std


# =====================================================================
# 7. Model (#3) -- small MLP with BN+Dropout
# =====================================================================

class RankingNet(nn.Module):
    def __init__(self, n_features, hidden_sizes=(64, 32), dropout=0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(n_features)]
        prev = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 6, n_features) -> score for each of 6 candidates
        batch, n_cand, n_feat = x.shape
        flat   = x.reshape(batch * n_cand, n_feat)
        scores = self.net(flat)
        return scores.reshape(batch, n_cand)


# =====================================================================
# 8. BPR Loss (#2, #9)
# =====================================================================

def bpr_group_loss(scores, ratings, hard_neg_temp=20.0):
    """BPR over (top-3, bottom-3) pairs within each group of 6.

    scores  : (batch, 6)  -- model predictions, positions 0-2 are top-3
    ratings : (batch, 6)  -- true ratings used for hard-neg weighting
    """
    pos_scores  = scores[:, :3]
    neg_scores  = scores[:, 3:]
    pos_ratings = ratings[:, :3]
    neg_ratings = ratings[:, 3:]

    # All 9 pairs per group: pos_i vs neg_j
    pos_exp = pos_scores.unsqueeze(2)
    neg_exp = neg_scores.unsqueeze(1)
    diff    = pos_exp - neg_exp

    # (#9) Hard negative weighting: close-rating pairs get more weight
    rating_gap = (pos_ratings.unsqueeze(2) - neg_ratings.unsqueeze(1)).abs()
    weight     = torch.exp(-rating_gap / hard_neg_temp)

    pair_loss = -torch.log(torch.sigmoid(diff) + 1e-8) * weight
    return pair_loss.mean()


# =====================================================================
# 9. Training Loop
# =====================================================================

def compute_val_auc(model, val_X, val_R):
    """AUC on val groups: does model rank top-3 above bottom-3?

    Vectorized: pushes the whole val tensor to GPU, scores it in chunks,
    then computes pair-AUC with broadcasting instead of Python loops.
    """
    model.eval()
    if isinstance(val_X, np.ndarray):
        val_X_gpu = torch.tensor(val_X, dtype=torch.float32, device=DEVICE)
    else:
        val_X_gpu = val_X.to(DEVICE)

    n = val_X_gpu.shape[0]
    chunk = 8192
    all_scores = []
    with torch.no_grad():
        for start in range(0, n, chunk):
            all_scores.append(model(val_X_gpu[start:start + chunk]))
    scores = torch.cat(all_scores, dim=0)        # (n, 6)

    # Vectorized pair-AUC: for each group, compare each top-3 vs each
    # bottom-3 score.  ties contribute 0.5.
    pos = scores[:, :3].unsqueeze(2)              # (n, 3, 1)
    neg = scores[:, 3:].unsqueeze(1)              # (n, 1, 3)
    diff = pos - neg
    correct = (diff > 0).float().sum() + 0.5 * (diff == 0).float().sum()
    total   = float(n * 9)
    return float(correct / total) if total > 0 else 0.0


def train_ranking_model(train_X, train_R, val_X, val_R):
    """BPR training with cosine LR schedule + early stopping.
    Caches the trained model to BPR_MODEL_CACHE; skips retraining on rerun."""
    print(f"\nTraining ranking model on {DEVICE} ...")
    print(f"  Train: {train_X.shape[0]:,} groups  |  "
          f"Val: {val_X.shape[0]:,} groups")

    if os.path.exists(BPR_MODEL_CACHE):
        print(f"  Loading cached BPR model from "
              f"{os.path.basename(BPR_MODEL_CACHE)} ...")
        ckpt = torch.load(BPR_MODEL_CACHE, map_location=DEVICE,
                          weights_only=False)
        model = RankingNet(N_FEATURES, HIDDEN_SIZES, DROPOUT).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        best_auc = float(ckpt.get("best_auc", 0.0))
        print(f"  Loaded.  Cached val AUC: {best_auc:.4f}")
        return model, best_auc

    # Move the entire training tensor to GPU once. Shuffling and batch
    # slicing are done on-GPU per epoch -- no DataLoader, no per-batch
    # CPU->GPU transfer, no Python overhead.
    train_X_gpu = torch.tensor(train_X, dtype=torch.float32, device=DEVICE)
    train_R_gpu = torch.tensor(train_R, dtype=torch.float32, device=DEVICE)
    n_train     = train_X_gpu.shape[0]

    model     = RankingNet(N_FEATURES, HIDDEN_SIZES, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, MAX_EPOCHS)

    best_auc          = 0.0
    best_state        = None
    patience_counter  = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx     = perm[start:start + BATCH_SIZE]
            x_batch = train_X_gpu[idx]
            r_batch = train_R_gpu[idx]
            scores  = model(x_batch)
            loss    = bpr_group_loss(scores, r_batch, HARD_NEG_TEMP)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        avg_loss = total_loss / n_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample_x = torch.tensor(train_X[:2048], dtype=torch.float32,
                                        device=DEVICE)
                sample_scores = model(sample_x).cpu().numpy()
                tc, tt = 0, 0
                for i in range(sample_scores.shape[0]):
                    for p in range(3):
                        for n in range(3, 6):
                            if   sample_scores[i, p] >  sample_scores[i, n]:
                                tc += 1
                            elif sample_scores[i, p] == sample_scores[i, n]:
                                tc += 0.5
                            tt += 1
                train_acc = tc / tt

            val_auc = compute_val_auc(model, val_X, val_R)
            improved = ""
            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_counter = 0
                improved = " *BEST*"
            else:
                patience_counter += 5

            print(f"  Epoch {epoch+1:3d}/{MAX_EPOCHS}  "
                  f"loss={avg_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  "
                  f"val_AUC={val_auc:.4f}{improved}")

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    print(f"  Best val AUC: {best_auc:.4f}")

    torch.save({
        "state_dict":   model.state_dict(),
        "best_auc":     best_auc,
        "n_features":   N_FEATURES,
        "hidden_sizes": HIDDEN_SIZES,
        "dropout":      DROPOUT,
    }, BPR_MODEL_CACHE)
    print(f"  Cached -> {os.path.basename(BPR_MODEL_CACHE)}")
    return model, best_auc


# =====================================================================
# 10. Prediction & Submission
# =====================================================================

def predict_test(model, user_ratings, track_meta, test_users,
                 feat_mean, feat_std,
                 global_stats, album_to_tracks, artist_to_tracks,
                 mf_uid_map, mf_iid_map, mf_user_f, mf_item_f,
                 out_path, model_name="model"):
    """Score test candidates -> top-3 per user -> Kaggle CSV.

    Returns:
      results       : [(key, label)]   binary 0/1 for the CSV
      raw_scores    : {(uid, tid): float}   continuous NN score for stacking
    """
    print(f"\nPredicting with {model_name} ...")
    t0 = time.time()
    model.eval()
    results    = []
    raw_scores = {}

    with torch.no_grad():
        for uid, candidates in test_users:
            ratings = user_ratings.get(uid, {})
            u_mean, u_std, u_n = compute_user_profile(ratings)

            base_feats = []
            for tid in candidates:
                bf = compute_base_features(
                    ratings, track_meta, tid, u_mean, u_std, u_n,
                    global_stats, album_to_tracks, artist_to_tracks,
                    mf_uid_map, mf_iid_map, mf_user_f, mf_item_f, uid)
                base_feats.append(bf)

            cross = compute_cross_features(track_meta, candidates)

            X = np.zeros((1, 6, N_FEATURES), dtype=np.float32)
            for c in range(6):
                X[0, c, :N_BASE_FEATURES] = base_feats[c]
                X[0, c, N_BASE_FEATURES:] = cross[c]
            X = normalize(X, feat_mean, feat_std)

            X_t    = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            scores = model(X_t).cpu().numpy()[0]  # shape (6,)

            for c, tid in enumerate(candidates):
                raw_scores[(uid, tid)] = float(scores[c])

            ranked = sorted(zip(scores, candidates), reverse=True)
            top3   = set(tid for _, tid in ranked[:3])
            for tid in candidates:
                results.append((f"{uid}_{tid}", 1 if tid in top3 else 0))

    with open(out_path, "w") as f:
        f.write("TrackID,Predictor\n")
        for key, label in results:
            f.write(f"{key},{label}\n")

    n_ones = sum(v for _, v in results)
    print(f"  {len(results):,} predictions  "
          f"({n_ones:,} rec / {len(results)-n_ones:,} not rec)  "
          f"[{elapsed(t0)}]")
    print(f"  Saved -> {os.path.basename(out_path)}")
    return results, raw_scores


# =====================================================================
# 11. V2 Heuristic Baseline
# =====================================================================

def compute_v2_scores(uid, candidates, user_ratings, track_meta):
    """Per-candidate v2 score = album + artist + n_genre*mean_genre/10."""
    ratings = user_ratings.get(uid, {})
    scores  = []
    for tid in candidates:
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
    return scores


def predict_v2_baseline(user_ratings, track_meta, test_users, out_path):
    """Pure v2 heuristic submission (no neural network)."""
    print(f"\nGenerating v2 heuristic baseline ...")
    t0 = time.time()
    results = []
    for uid, candidates in test_users:
        v2_scores = compute_v2_scores(uid, candidates, user_ratings,
                                       track_meta)
        ranked = sorted(zip(v2_scores, candidates), reverse=True)
        top3   = set(tid for _, tid in ranked[:3])
        for tid in candidates:
            results.append((f"{uid}_{tid}", 1 if tid in top3 else 0))

    with open(out_path, "w") as f:
        f.write("TrackID,Predictor\n")
        for key, label in results:
            f.write(f"{key},{label}\n")
    n_ones = sum(v for _, v in results)
    print(f"  {len(results):,} predictions  "
          f"({n_ones:,} rec)  [{elapsed(t0)}]")
    print(f"  Saved -> {os.path.basename(out_path)}")
    return results


def v2_heuristic_val_auc(user_ratings, track_meta, val_groups):
    """Pair-AUC for v2 heuristic on the same val groups as the NN."""
    correct, total = 0, 0
    for uid, tids, _ in val_groups:
        scores = compute_v2_scores(uid, tids, user_ratings, track_meta)
        for p in range(3):
            for n in range(3, 6):
                if   scores[p] >  scores[n]: correct += 1
                elif scores[p] == scores[n]: correct += 0.5
                total += 1
    return correct / total if total > 0 else 0.0


# =====================================================================
# 11b. PySpark ALS (Final spec's expected method, used as one
#      meta-learner input column).
#
# Per Wang's supplemental "PySpark Recommendation Code for the Final
# Project.py", the ALS configuration is rank=5/maxIter=5 by default; HW8
# established that rank=20, maxIter=20 generalizes better on this dataset.
# We train ALS on the full ratings matrix and score every (uid, tid) pair
# we'll need: the 6,000 labeled rows AND the 120,000 Kaggle test rows.
# =====================================================================

def train_als_and_score(user_ratings, test_users, label_keys,
                          out_path_submission):
    """Train PySpark ALS on the full rating matrix, then score:
       (a) every (uid, tid) appearing in test2_new.txt   -- for stacking
       (b) every (uid, tid) in the Kaggle test set       -- for stacking
                                                            + ALS-only sub
    Returns:
       als_scores : {(uid, tid): float}   for ALL keys we care about
       None on failure (e.g., PySpark not installed)
    """
    section("PYSPARK ALS")

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.types import IntegerType, FloatType, StructType, \
            StructField
        from pyspark.ml.recommendation import ALS
    except ImportError:
        print("  PySpark not installed -- skipping ALS column.")
        print("  pip install pyspark  to enable.")
        return None

    t0 = time.time()

    # Pin Spark's Python workers to the SAME interpreter that's running this
    # script. Otherwise Spark may spawn Python 3.14 workers from PATH, and
    # 3.14's socket finalization breaks the Spark<->Python protocol on
    # Windows (WinError 10038 / "An operation was attempted on something
    # that is not a socket").
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    print(f"  Pinning Spark workers to: {sys.executable}")

    # JVM thread stack bumped to 32M -- same fix as HW9 to avoid
    # StackOverflowError when serializing ALS factor matrices on Windows.
    spark = (SparkSession.builder
             .appName("EE627-Final-ALS")
             .config("spark.driver.memory",            "6g")
             .config("spark.driver.extraJavaOptions",  "-Xss32m")
             .config("spark.executor.extraJavaOptions","-Xss32m")
             .config("spark.pyspark.python",           sys.executable)
             .config("spark.pyspark.driver.python",    sys.executable)
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Build training DataFrame from user_ratings dict
    print("  Assembling rating tuples ...")
    rows = []
    for uid, ratings in user_ratings.items():
        for iid, r in ratings.items():
            rows.append((int(uid), int(iid), float(r)))
    schema = StructType([
        StructField("userID", IntegerType()),
        StructField("itemID", IntegerType()),
        StructField("rating", FloatType()),
    ])
    train_df = spark.createDataFrame(rows, schema=schema)
    n_train  = train_df.count()
    print(f"  Training rows: {n_train:,}")

    print("  Training ALS (rank=20, maxIter=20, regParam=0.05) ...")
    als = ALS(
        rank=20,
        maxIter=20,
        regParam=0.05,
        userCol="userID",
        itemCol="itemID",
        ratingCol="rating",
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop",
        seed=42,
    )
    model = als.fit(train_df)
    print(f"  ALS trained  [{elapsed(t0)}]")

    # Build the union of all (uid, tid) we need to score:
    #   - test2_new.txt label rows  (for meta-learner training)
    #   - all 120K Kaggle test rows (for meta-learner inference + ALS sub)
    print("  Building score request set ...")
    all_keys = set(label_keys)
    for uid, candidates in test_users:
        for tid in candidates:
            all_keys.add((int(uid), int(tid)))

    score_rows = [(uid, tid) for uid, tid in all_keys]
    score_schema = StructType([
        StructField("userID", IntegerType()),
        StructField("itemID", IntegerType()),
    ])
    score_df = spark.createDataFrame(score_rows, schema=score_schema)

    print(f"  Scoring {len(score_rows):,} (uid, tid) pairs ...")
    pred_df = model.transform(score_df)
    pred_rows = pred_df.select("userID", "itemID", "prediction").collect()

    als_scores = {}
    n_drops = 0
    for r in pred_rows:
        uid = int(r["userID"])
        tid = int(r["itemID"])
        p   = r["prediction"]
        if p is None:        # cold-start -> coldStartStrategy='drop' filtered
            n_drops += 1
            continue
        als_scores[(uid, tid)] = float(p)

    # Backfill cold-start rows with the global mean rating
    if n_drops or len(als_scores) < len(score_rows):
        global_mean_row = train_df.selectExpr("avg(rating) as m").collect()[0]
        global_mean = float(global_mean_row["m"])
        n_filled = 0
        for k in score_rows:
            if k not in als_scores:
                als_scores[k] = global_mean
                n_filled += 1
        print(f"  Cold-start: {n_filled:,} pairs backfilled with global "
              f"mean ({global_mean:.2f})")

    # Write an ALS-only submission for diagnostic value
    if out_path_submission is not None:
        n_ones, n_total = 0, 0
        with open(out_path_submission, "w") as f:
            f.write("TrackID,Predictor\n")
            for uid, candidates in test_users:
                pairs = [(tid, als_scores[(uid, tid)]) for tid in candidates]
                top3  = set(t for t, _ in sorted(pairs, key=lambda x: x[1],
                                                  reverse=True)[:3])
                for tid, _ in pairs:
                    pred = 1 if tid in top3 else 0
                    f.write(f"{uid}_{tid},{pred}\n")
                    n_ones  += pred
                    n_total += 1
        print(f"  ALS-only submission: {n_ones:,} rec / "
              f"{n_total-n_ones:,} not rec  -> "
              f"{os.path.basename(out_path_submission)}")

    spark.stop()
    print(f"  ALS pipeline complete  [{elapsed(t0)}]")
    return als_scores


# =====================================================================
# 12. Confidence-Gated Hybrid (the actual 0.911 producer)
# =====================================================================

def generate_hybrid_submissions(test_users, user_ratings, track_meta,
                                  v5_results, thresholds, output_dir):
    """For each gap threshold, build a hybrid submission:
        if v2_gap > threshold:  use v2 top-3
        else:                   defer to neural net (v5)
    Writes one CSV per threshold."""
    section("CONFIDENCE-GATED HYBRID SWEEP")

    # Build v5 lookup by (uid, tid)
    v5_by_key = {key: label for key, label in v5_results}

    # Pre-compute v2 top-3 + gap per user
    user_data = []
    for uid, candidates in test_users:
        v2_scores = compute_v2_scores(uid, candidates, user_ratings,
                                       track_meta)
        ranked    = sorted(zip(v2_scores, candidates), reverse=True)
        v2_top3   = set(tid for _, tid in ranked[:3])
        ss        = sorted(v2_scores, reverse=True)
        gap       = ss[2] - ss[3]
        user_data.append((uid, candidates, v2_top3, gap))

    print(f"  {'Thresh':>6}  {'v2 users':>10}  {'v5 users':>10}  "
          f"{'v5 pct':>8}  File")
    print("  " + "-" * 70)

    for thresh in thresholds:
        rows = []
        n_v2, n_v5 = 0, 0
        for uid, candidates, v2_top3, gap in user_data:
            if gap > thresh:
                n_v2 += 1
                for tid in candidates:
                    key = f"{uid}_{tid}"
                    rows.append((key, 1 if tid in v2_top3 else 0))
            else:
                n_v5 += 1
                for tid in candidates:
                    key = f"{uid}_{tid}"
                    rows.append((key, v5_by_key.get(key, 0)))

        fname = f"submission_final_v5_hybrid_gap{thresh}.csv"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w") as f:
            f.write("TrackID,Predictor\n")
            for key, label in rows:
                f.write(f"{key},{label}\n")

        v5_pct = 100 * n_v5 / len(user_data)
        marker = "  <- production (0.911 on midterm)" if thresh == 10 else ""
        print(f"  {thresh:>6}  {n_v2:>10,}  {n_v5:>10,}  "
              f"{v5_pct:>7.1f}%  {fname}{marker}")


# =====================================================================
# 12b. Stacked Meta-Learner
# =====================================================================
#
# Trains a logistic regression on test2_new.txt labels using whatever
# score signals are available:
#   - v5 NN raw score        (this script's output)
#   - v2 heuristic score     (this script's output)
#   - HW9 PySpark probabilities  (loaded from ../Homework 9/hw9_probabilities.csv
#                                 if present)
#
# Falls back to (v5, v2) only if HW9 probs are missing.  Falls back to
# hand-coded LR if sklearn is missing.

def build_v2_score_dict(test_users, user_ratings, track_meta):
    """{(uid, tid): v2_score} for every test pair."""
    print("\nBuilding v2 score lookup for stacking ...")
    t0 = time.time()
    out = {}
    for uid, candidates in test_users:
        scores = compute_v2_scores(uid, candidates, user_ratings, track_meta)
        for tid, s in zip(candidates, scores):
            out[(uid, tid)] = float(s)
    print(f"  {len(out):,} (uid, tid) v2 scores  [{elapsed(t0)}]")
    return out


def load_hw9_probabilities(path):
    """Load HW9's per-(uid, tid) probability matrix if it exists.
    Returns dict {(uid, tid): {model_name: prob}} or None."""
    if not os.path.exists(path):
        return None
    print(f"\nLoading HW9 probability matrix ...")
    t0 = time.time()
    import csv as _csv
    out = {}
    model_names = None
    with open(path, "r", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        model_names = header[2:]   # skip user_id, track_id columns
        for row in reader:
            uid = int(row[0])
            tid = int(row[1])
            out[(uid, tid)] = {n: float(v)
                               for n, v in zip(model_names, row[2:])}
    print(f"  {len(out):,} rows x {len(model_names)} models loaded "
          f"[{elapsed(t0)}]")
    print(f"  Models: {', '.join(model_names)}")
    return out, model_names


def load_test2_labels(path):
    """test2_new.txt -> {(uid, tid): 0_or_1}."""
    labels = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                labels[(int(parts[0]), int(parts[1]))] = int(parts[2])
    return labels


def load_past_predictions(directory):
    """Load every submission CSV in `directory` as a binary 0/1 column.

    Filename convention: <name>_<score>.csv  (e.g., v5b_hybrid_gap10_0911.csv)
    The trailing 4 digits encode the Kaggle score (0911 -> 0.911).  This is
    used purely for logging / sorting -- the meta-learner sees only the
    binary predictions.

    Returns:
      preds : {(uid, tid): {col_name: 0_or_1}}
      cols  : ordered list of column names (sorted by Kaggle score desc)
    """
    if not os.path.isdir(directory):
        return None, []

    import csv as _csv
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not files:
        return None, []

    print(f"\nLoading past predictions from {os.path.basename(directory)} ...")
    t0 = time.time()

    # Parse score from filename if present
    def parse_score(fname):
        stem = fname[:-4]   # drop .csv
        if "_" in stem:
            tail = stem.rsplit("_", 1)[1]
            if tail.isdigit() and len(tail) == 4:
                return int(tail) / 1000.0
        return None

    files_with_scores = [(f, parse_score(f)) for f in files]
    # Sort: known scores desc, unknown last
    files_with_scores.sort(
        key=lambda x: (-x[1] if x[1] is not None else 1.0, x[0]))

    cols  = []
    preds = defaultdict(dict)
    for fname, score in files_with_scores:
        col_name = fname[:-4]
        cols.append(col_name)
        path = os.path.join(directory, fname)
        with open(path, "r", newline="") as f:
            reader = _csv.reader(f)
            next(reader)   # header
            for row in reader:
                key  = row[0]
                pred = int(row[1])
                # key format: "uid_tid"
                uid_str, tid_str = key.split("_")
                preds[(int(uid_str), int(tid_str))][col_name] = pred

    print(f"  Loaded {len(cols)} submissions  |  "
          f"{len(preds):,} (uid, tid) pairs  [{elapsed(t0)}]")
    print(f"  Score range:")
    for fname, score in files_with_scores:
        score_str = f"{score:.3f}" if score is not None else "  ?  "
        print(f"    {score_str}   {fname}")
    return preds, cols


def fit_logistic_regression(X, y, n_iter=2000, lr=0.05, l2=0.01):
    """Hand-coded LR with L2.  Used if sklearn is unavailable.
    X: (n, d), y: (n,) in {0, 1}.  Features assumed roughly z-scored."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        z   = X @ w + b
        p   = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = p - y
        gw  = (X.T @ err) / n + l2 * w
        gb  = err.mean()
        w  -= lr * gw
        b  -= lr * gb
    return w, b


def generate_stacked_submission(test_users, v5_raw, v2_raw,
                                 label_path, hw9_probs_path,
                                 past_preds_dir, out_path,
                                 als_scores=None):
    """Train a meta-learner on test2_new.txt labels using all available
    score columns, then apply to all 120K test rows and write submission.

    Returns the output path on success, None if labels are missing.
    """
    section("STACKED META-LEARNER")

    if not os.path.exists(label_path):
        print(f"  test2_new.txt not found at {label_path} -- skipping stacking")
        return None

    labels = load_test2_labels(label_path)
    print(f"  Loaded {len(labels):,} labels  "
          f"({sum(labels.values()):,} positives)")

    # Optional: load HW9 PySpark probabilities
    hw9 = load_hw9_probabilities(hw9_probs_path)
    if hw9 is not None:
        hw9_probs, hw9_models = hw9
    else:
        hw9_probs, hw9_models = None, []
        print(f"\n  hw9_probabilities.csv not found at:")
        print(f"    {hw9_probs_path}")
        print(f"  Stacking on (v5, v2) only from this run -- run HW9 first")
        print(f"  to add 7 more PySpark columns.")

    # Optional: load past Kaggle submissions as binary feature columns
    past_preds, past_cols = load_past_predictions(past_preds_dir)
    if past_preds is None:
        print(f"\n  No past predictions in {past_preds_dir}")

    # Build feature matrix for the LABELED set
    has_als      = als_scores is not None
    als_col      = ["als_score"] if has_als else []
    feature_cols = (["v5_score", "v2_score"] + als_col
                    + list(hw9_models) + list(past_cols))
    print(f"\n  Stacking features ({len(feature_cols)} total): "
          f"v5+v2 + {len(als_col)} ALS + {len(hw9_models)} HW9 "
          f"+ {len(past_cols)} past")

    X_rows, y_rows, missing = [], [], 0
    for (uid, tid), y in labels.items():
        v5 = v5_raw.get((uid, tid))
        v2 = v2_raw.get((uid, tid))
        if v5 is None or v2 is None:
            missing += 1
            continue
        row = [v5, v2]
        if has_als:
            row.append(als_scores.get((uid, tid), 0.0))
        if hw9_probs is not None:
            hw9_row = hw9_probs.get((uid, tid))
            if hw9_row is None:
                missing += 1
                continue
            row += [hw9_row[m] for m in hw9_models]
        if past_preds is not None:
            past_row = past_preds.get((uid, tid), {})
            row += [float(past_row.get(c, 0)) for c in past_cols]
        X_rows.append(row)
        y_rows.append(y)

    if missing:
        print(f"  Skipped {missing:,} labeled rows (missing scores)")

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.float64)
    print(f"  Training matrix: {X.shape}  "
          f"({int(y.sum()):,} pos / {int(len(y)-y.sum()):,} neg)")

    # Standardize for numerical stability of the LR
    feat_mean = X.mean(axis=0)
    feat_std  = X.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    Xn = (X - feat_mean) / feat_std

    # Fit LR -- prefer sklearn if present
    weights = None
    bias    = None
    try:
        from sklearn.linear_model import LogisticRegressionCV
        clf = LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc",
                                    max_iter=2000, n_jobs=-1,
                                    random_state=SEED)
        clf.fit(Xn, y)
        weights = clf.coef_[0]
        bias    = float(clf.intercept_[0])
        cv_auc  = float(clf.scores_[1.0].mean(axis=0).max())
        print(f"\n  sklearn LogisticRegressionCV  "
              f"(best C={clf.C_[0]:.4f}, 5-fold CV AUC={cv_auc:.4f})")
    except ImportError:
        print(f"\n  sklearn not available -- using hand-coded LR")
        w, b   = fit_logistic_regression(Xn, y, n_iter=3000, lr=0.05, l2=0.01)
        weights = w
        bias    = b

    # Print learned weights -- which signals matter
    print(f"\n  Meta-learner weights (on standardized features):")
    print(f"  {'Feature':<20} {'Weight':>10}")
    print(f"  {'-'*20} {'-'*10}")
    ranked = sorted(enumerate(weights), key=lambda x: abs(x[1]), reverse=True)
    for idx, w_val in ranked:
        print(f"  {feature_cols[idx]:<20} {w_val:>+10.4f}")
    print(f"  {'(intercept)':<20} {bias:>+10.4f}")

    # Score every test row, group by user, pick top-3
    print(f"\n  Scoring all 120K test rows ...")
    by_user = defaultdict(list)
    for uid, candidates in test_users:
        for tid in candidates:
            row = [v5_raw.get((uid, tid), 0.0),
                   v2_raw.get((uid, tid), 0.0)]
            if has_als:
                row.append(als_scores.get((uid, tid), 0.0))
            if hw9_probs is not None:
                hw9_row = hw9_probs.get((uid, tid))
                if hw9_row is None:
                    row += [0.5] * len(hw9_models)   # neutral fallback
                else:
                    row += [hw9_row[m] for m in hw9_models]
            if past_preds is not None:
                past_row = past_preds.get((uid, tid), {})
                row += [float(past_row.get(c, 0)) for c in past_cols]
            row_n = (np.array(row) - feat_mean) / feat_std
            score = float(row_n @ weights + bias)
            by_user[uid].append((tid, score))

    # Top-3 per user
    n_ones, n_total = 0, 0
    with open(out_path, "w") as f:
        f.write("TrackID,Predictor\n")
        for uid, items in by_user.items():
            top3 = set(t for t, _ in sorted(items, key=lambda x: x[1],
                                            reverse=True)[:3])
            for tid, _ in items:
                pred = 1 if tid in top3 else 0
                f.write(f"{uid}_{tid},{pred}\n")
                n_ones  += pred
                n_total += 1

    print(f"  {n_total:,} predictions  "
          f"({n_ones:,} rec / {n_total-n_ones:,} not rec)")
    print(f"  Saved -> {os.path.basename(out_path)}")
    return out_path


# =====================================================================
# 13. Main
# =====================================================================

def main():
    tee = Tee(RESULTS_FILE)
    sys.stdout = tee
    t_total = time.time()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    section("EE627 FINAL PROJECT - Music Recommender")
    print(f"  Log file : {RESULTS_FILE}")
    print(f"  Started  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device   : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")

    # ---- Load data ----
    section("LOADING RAW DATA")
    user_ratings = parse_training(os.path.join(DATA_DIR, "trainItem2.txt"))
    track_meta   = parse_tracks  (os.path.join(DATA_DIR, "trackData2.txt"))
    album_meta   = parse_albums  (os.path.join(DATA_DIR, "albumData2.txt"))
    test_users   = parse_test    (os.path.join(DATA_DIR, "testItem2.txt"))

    # ---- Enrich + auxiliary structures ----
    section("ENRICHMENT & AUXILIARY STRUCTURES")
    enriched      = enrich_tracks_with_albums(track_meta, album_meta)
    global_stats  = build_global_item_stats(user_ratings)
    album_to_tracks, artist_to_tracks = build_track_mappings(enriched)

    # ---- Matrix factorization (#10) ----
    section("MATRIX FACTORIZATION (GPU)")
    mf_uid_map, mf_iid_map, mf_user_f, mf_item_f = train_mf_embeddings(
        user_ratings, MF_FACTORS, MF_EPOCHS, MF_LR, MF_BATCH)

    # ---- User split for validation ----
    section("BUILDING TRAINING & VALIDATION DATA")
    rng = random.Random(SEED)
    all_uids = list(user_ratings.keys())
    rng.shuffle(all_uids)
    split_pt    = int(len(all_uids) * 0.9)
    train_uids  = set(all_uids[:split_pt])
    val_uids    = set(all_uids[split_pt:])
    print(f"  Train users: {len(train_uids):,}  |  "
          f"Val users: {len(val_uids):,}")

    if os.path.exists(FEATURES_CACHE):
        print(f"\n  Loading cached features from "
              f"{os.path.basename(FEATURES_CACHE)} ...")
        t_feat = time.time()
        z = np.load(FEATURES_CACHE, allow_pickle=True)
        train_X    = z["train_X"]
        train_R    = z["train_R"]
        val_X      = z["val_X"]
        val_R      = z["val_R"]
        feat_mean  = z["feat_mean"]
        feat_std   = z["feat_std"]
        val_groups = z["val_groups"].tolist()
        print(f"  Train X: {train_X.shape}  |  Val X: {val_X.shape}  "
              f"[{elapsed(t_feat)}]")
    else:
        train_groups = build_groups(user_ratings, enriched, train_uids,
                                     GROUPS_PER_USER, SEED)
        val_groups   = build_groups(user_ratings, enriched, val_uids,
                                     10, SEED + 1)
        print(f"  Train groups: {len(train_groups):,}  |  "
              f"Val groups: {len(val_groups):,}")

        print("\n  Featurizing training groups ...")
        train_X, train_R = featurize_groups(
            train_groups, user_ratings, enriched, global_stats,
            album_to_tracks, artist_to_tracks,
            mf_uid_map, mf_iid_map, mf_user_f, mf_item_f)

        print("  Featurizing validation groups ...")
        val_X, val_R = featurize_groups(
            val_groups, user_ratings, enriched, global_stats,
            album_to_tracks, artist_to_tracks,
            mf_uid_map, mf_iid_map, mf_user_f, mf_item_f)

        # ---- Normalize (#4) ----
        feat_mean, feat_std = compute_norm_stats(train_X)
        train_X = normalize(train_X, feat_mean, feat_std)
        val_X   = normalize(val_X,   feat_mean, feat_std)

        np.savez(FEATURES_CACHE,
                 train_X=train_X, train_R=train_R,
                 val_X=val_X, val_R=val_R,
                 feat_mean=feat_mean, feat_std=feat_std,
                 val_groups=np.array(val_groups, dtype=object))
        print(f"  Cached -> {os.path.basename(FEATURES_CACHE)}")

    # ---- v2 baseline AUC (sanity check) ----
    section("V2 HEURISTIC BASELINE")
    v2_auc = v2_heuristic_val_auc(user_ratings, enriched, val_groups)
    print(f"  v2 heuristic val AUC: {v2_auc:.4f}  (expected Kaggle ~0.871)")

    # ---- Train neural ranker ----
    section("TRAINING RANKING MODEL (BPR)")
    model, best_auc = train_ranking_model(train_X, train_R, val_X, val_R)

    # ---- Submissions ----
    section("GENERATING SUBMISSIONS")

    # 1. Pure v5 neural network  (returns binary results + raw scores)
    v5_results, v5_raw_scores = predict_test(
        model, user_ratings, enriched, test_users,
        feat_mean, feat_std,
        global_stats, album_to_tracks, artist_to_tracks,
        mf_uid_map, mf_iid_map, mf_user_f, mf_item_f,
        os.path.join(OUTPUT_DIR, "submission_final_v5_nn.csv"),
        "v5 BPR Neural Net")

    # 2. v2 heuristic baseline
    predict_v2_baseline(
        user_ratings, enriched, test_users,
        os.path.join(OUTPUT_DIR, "submission_final_v2.csv"))

    # Build v2 score lookup for stacking
    v2_raw_scores = build_v2_score_dict(test_users, user_ratings, enriched)

    # 3. Confidence-gated hybrid sweep -- gap10 = production submission
    generate_hybrid_submissions(
        test_users, user_ratings, enriched,
        v5_results, HYBRID_THRESHOLDS, OUTPUT_DIR)

    # 4. PySpark ALS -- another meta-learner input column.
    #    Pre-load the labeled-set keys so ALS knows which (uid, tid) pairs
    #    we'll need scored for the meta-learner training matrix.
    als_label_keys = []
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 3:
                    als_label_keys.append((int(parts[0]), int(parts[1])))
    als_scores = train_als_and_score(
        user_ratings, test_users, als_label_keys,
        os.path.join(OUTPUT_DIR, "submission_final_als.csv"))

    # 5. Stacked meta-learner submission
    stacked_path = generate_stacked_submission(
        test_users, v5_raw_scores, v2_raw_scores,
        LABEL_FILE, HW9_PROBS_FILE, PAST_PREDS_DIR,
        os.path.join(OUTPUT_DIR, "submission_final_stacked.csv"),
        als_scores=als_scores)

    # ---- Summary ----
    section("FINAL SUMMARY")
    print(f"  v2 heuristic val AUC : {v2_auc:.4f}  (Kaggle baseline 0.871)")
    print(f"  v5 BPR-NN  val AUC   : {best_auc:.4f}")
    print(f"  Delta                : {best_auc - v2_auc:+.4f}")
    print()
    print(f"  Submissions written:")
    print(f"    submission_final_v2.csv                  v2 heuristic alone")
    print(f"    submission_final_v5_nn.csv               pure neural net")
    print(f"    submission_final_v5_hybrid_gap{{N}}.csv    hybrid sweep")
    if stacked_path is not None:
        print(f"    submission_final_stacked.csv             stacked meta-learner")
    print()
    print(f"  Recommended Kaggle submissions:")
    print(f"    submission_final_v5_hybrid_gap10.csv     "
          f"(scored 0.911 on midterm Kaggle)")
    if stacked_path is not None:
        print(f"    submission_final_stacked.csv            "
              f"(meta-learner over all signals)")
    print()
    print(f"  Total wall time : {elapsed(t_total)}")

    tee.close()


if __name__ == "__main__":
    main()

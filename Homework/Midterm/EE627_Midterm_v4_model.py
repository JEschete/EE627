"""
EE627A Midterm v4 - PyTorch Pairwise Ranking Model (GPU)
Jude Eschete

Key insight from failed attempts:
  - Heuristic weight tuning hit ceiling at 0.871
  - Pointwise regression (predict rating) doesn't optimize ranking (0.821-0.852)
  - Pairwise ranking loss directly optimizes "which candidate ranks higher"
    which is exactly what AUC measures

Approach:
  1. Build (user, track) feature vectors from album/artist/genre signals
  2. Construct PAIRS: (preferred_track, non-preferred_track) from user history
  3. Train a neural network with pairwise margin ranking loss:
     loss = max(0, margin - (score(preferred) - score(non-preferred)))
  4. The model learns a scoring function that produces correct relative
     rankings, not absolute ratings
  5. At test time, score 6 candidates, pick top 3

Uses RTX 5080 for fast training.
"""

import os
import sys
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# Configuration
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "Homework 4", "Data")

TRAIN_FILE = os.path.join(DATA_DIR, "trainItem2.txt")
TEST_FILE = os.path.join(DATA_DIR, "testItem2.txt")
TRACK_FILE = os.path.join(DATA_DIR, "trackData2.txt")
ALBUM_FILE = os.path.join(DATA_DIR, "albumData2.txt")

OUTPUT_DIR = SCRIPT_DIR
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results.txt")
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
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
# 3. Feature Engineering
# =====================================================================

N_FEATURES = 19

def compute_user_profile(ratings):
    vals = list(ratings.values())
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    return mean, var ** 0.5, n


def compute_feature_vec(ratings, track_meta, tid, u_mean, u_std, u_n):
    """19-dimensional feature vector."""
    alb_id, art_id, genre_ids = track_meta.get(tid, (None, None, []))

    if alb_id is not None and alb_id in ratings:
        album_score, has_album = float(ratings[alb_id]), 1.0
    else:
        album_score, has_album = 0.0, 0.0

    if art_id is not None and art_id in ratings:
        artist_score, has_artist = float(ratings[art_id]), 1.0
    else:
        artist_score, has_artist = 0.0, 0.0

    genre_scores = [ratings[g] for g in genre_ids if g in ratings]
    gc = len(genre_scores)
    if gc > 0:
        g_max = float(max(genre_scores))
        g_min = float(min(genre_scores))
        g_mean = sum(genre_scores) / gc
        g_var = sum((s - g_mean) ** 2 for s in genre_scores) / gc
    else:
        g_max = g_min = 0.0
        g_mean = g_var = 0.0

    evidence_count = has_album + has_artist + (1.0 if gc > 0 else 0.0)

    aa_int = album_score * artist_score / 100.0 if (has_album and has_artist) else 0.0
    ga_int = g_mean * artist_score / 100.0 if (has_artist and gc > 0) else 0.0
    ag_int = album_score * g_mean / 100.0 if (has_album and gc > 0) else 0.0

    album_above = (album_score - u_mean) if has_album else 0.0
    artist_above = (artist_score - u_mean) if has_artist else 0.0
    genre_above = (g_mean - u_mean) if gc > 0 else 0.0

    return [
        album_score, has_album,
        artist_score, has_artist,
        float(gc), g_max, g_min, g_mean, g_var,
        evidence_count,
        aa_int, ga_int, ag_int,
        u_mean, u_std, float(u_n),
        album_above, artist_above, genre_above,
    ]


# =====================================================================
# 4. Pairwise Training Data
# =====================================================================

class PairwiseDataset(Dataset):
    """Dataset of (preferred_features, non_preferred_features) pairs."""
    def __init__(self, X_pos, X_neg):
        self.X_pos = torch.tensor(X_pos, dtype=torch.float32)
        self.X_neg = torch.tensor(X_neg, dtype=torch.float32)

    def __len__(self):
        return len(self.X_pos)

    def __getitem__(self, idx):
        return self.X_pos[idx], self.X_neg[idx]


def build_pairwise_data(user_ratings, track_meta, pairs_per_user=100,
                        seed=42):
    """Build pairwise training data from user track ratings.

    For each user, sample pairs of tracks where one was rated higher.
    The model learns: score(higher_rated) > score(lower_rated).
    """
    print("\n  Building pairwise training data ...")
    rng = random.Random(seed)
    track_ids = set(track_meta.keys())

    X_pos_rows = []
    X_neg_rows = []
    users_used = 0

    for uid, ratings in user_ratings.items():
        user_tracks = [(iid, ratings[iid]) for iid in ratings if iid in track_ids]
        if len(user_tracks) < 2:
            continue

        users_used += 1
        u_mean, u_std, u_n = compute_user_profile(ratings)

        # Cache features for this user's tracks
        track_feats = {}
        for tid, _ in user_tracks:
            track_feats[tid] = compute_feature_vec(
                ratings, track_meta, tid, u_mean, u_std, u_n)

        # Sample pairs: pick two tracks, higher-rated = positive
        n_pairs = min(pairs_per_user, len(user_tracks) * (len(user_tracks) - 1) // 2)
        for _ in range(n_pairs):
            t1, t2 = rng.sample(user_tracks, 2)
            tid1, r1 = t1
            tid2, r2 = t2
            if r1 == r2:
                continue  # Skip ties
            if r1 > r2:
                X_pos_rows.append(track_feats[tid1])
                X_neg_rows.append(track_feats[tid2])
            else:
                X_pos_rows.append(track_feats[tid2])
                X_neg_rows.append(track_feats[tid1])

    X_pos = np.array(X_pos_rows, dtype=np.float32)
    X_neg = np.array(X_neg_rows, dtype=np.float32)
    print(f"       {users_used:,} users, {len(X_pos):,} pairs")
    return X_pos, X_neg


# =====================================================================
# 5. Neural Network
# =====================================================================

class RankingNet(nn.Module):
    """MLP that scores a (user, track) feature vector.
    Trained with pairwise margin loss so it learns relative rankings."""

    def __init__(self, n_features, hidden_sizes=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(X_pos, X_neg, n_epochs=30, batch_size=4096, lr=1e-3,
                margin=1.0):
    """Train with MarginRankingLoss on GPU."""
    print(f"\n  Training on {DEVICE} ...")

    dataset = PairwiseDataset(X_pos, X_neg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    model = RankingNet(N_FEATURES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.MarginRankingLoss(margin=margin)

    target = torch.ones(batch_size, device=DEVICE)  # pos should rank higher

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for x_pos, x_neg in loader:
            x_pos = x_pos.to(DEVICE)
            x_neg = x_neg.to(DEVICE)
            bs = x_pos.size(0)

            s_pos = model(x_pos)
            s_neg = model(x_neg)

            t = target[:bs]
            loss = criterion(s_pos, s_neg, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Compute accuracy: how often does model rank positive > negative
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_pos, x_neg in loader:
                    x_pos = x_pos.to(DEVICE)
                    x_neg = x_neg.to(DEVICE)
                    s_pos = model(x_pos)
                    s_neg = model(x_neg)
                    correct += (s_pos > s_neg).sum().item()
                    total += x_pos.size(0)
            acc = correct / total
            print(f"    Epoch {epoch+1:3d}/{n_epochs}  "
                  f"loss={avg_loss:.4f}  pair_acc={acc:.4f}")

    return model


# =====================================================================
# 6. Validation
# =====================================================================

def validate(model, user_ratings, track_meta, fraction=0.1, seed=42):
    """AUC validation: can model rank top-3 tracks above bottom-3?"""
    print("\n  Validating ...")
    model.eval()
    rng = random.Random(seed)
    track_ids = set(track_meta.keys())

    all_uids = list(user_ratings.keys())
    rng.shuffle(all_uids)
    n_val = int(len(all_uids) * fraction)

    correct = 0
    total = 0
    val_users = 0

    with torch.no_grad():
        for uid in all_uids[:n_val]:
            ratings = user_ratings[uid]
            user_tracks = [iid for iid in ratings if iid in track_ids]
            if len(user_tracks) < 6:
                continue

            sample = rng.sample(user_tracks, 6)
            true_ratings = {tid: ratings[tid] for tid in sample}
            u_mean, u_std, u_n = compute_user_profile(ratings)

            X = torch.tensor(
                [compute_feature_vec(ratings, track_meta, tid, u_mean, u_std, u_n)
                 for tid in sample],
                dtype=torch.float32, device=DEVICE
            )
            scores = model(X).cpu().numpy()

            rated = sorted(zip([true_ratings[t] for t in sample], sample),
                           reverse=True)
            positives = set(tid for _, tid in rated[:3])
            negatives = set(tid for _, tid in rated[3:])
            score_map = dict(zip(sample, scores))

            for p in positives:
                for n in negatives:
                    if score_map[p] > score_map[n]:
                        correct += 1
                    elif score_map[p] == score_map[n]:
                        correct += 0.5
                    total += 1
            val_users += 1

    auc = correct / total if total > 0 else 0.0
    print(f"       {val_users:,} val users, AUC = {auc:.4f}")
    return auc


def validate_v2(user_ratings, track_meta, fraction=0.1, seed=42):
    """v2 heuristic baseline for comparison."""
    print("\n  Validating v2 heuristic ...")
    rng = random.Random(seed)
    track_ids = set(track_meta.keys())
    all_uids = list(user_ratings.keys())
    rng.shuffle(all_uids)
    n_val = int(len(all_uids) * fraction)

    correct = 0
    total = 0
    val_users = 0

    for uid in all_uids[:n_val]:
        ratings = user_ratings[uid]
        user_tracks = [iid for iid in ratings if iid in track_ids]
        if len(user_tracks) < 6:
            continue
        sample = rng.sample(user_tracks, 6)
        true_ratings = {tid: ratings[tid] for tid in sample}
        scores = []
        for tid in sample:
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
        rated = sorted(zip([true_ratings[t] for t in sample], sample),
                       reverse=True)
        positives = set(tid for _, tid in rated[:3])
        negatives = set(tid for _, tid in rated[3:])
        score_map = dict(zip(sample, scores))
        for p in positives:
            for n in negatives:
                if score_map[p] > score_map[n]:
                    correct += 1
                elif score_map[p] == score_map[n]:
                    correct += 0.5
                total += 1
        val_users += 1

    auc = correct / total if total > 0 else 0.0
    print(f"       {val_users:,} val users, AUC = {auc:.4f}")
    return auc


# =====================================================================
# 7. Prediction
# =====================================================================

def predict_and_submit(model, user_ratings, track_meta, test_users,
                       out_path, model_name="model"):
    print(f"\n>>> Predicting with {model_name} ...")
    model.eval()
    results = []

    with torch.no_grad():
        for uid, candidates in test_users:
            ratings = user_ratings.get(uid, {})
            u_mean, u_std, u_n = compute_user_profile(ratings)

            X = torch.tensor(
                [compute_feature_vec(ratings, track_meta, tid, u_mean, u_std, u_n)
                 for tid in candidates],
                dtype=torch.float32, device=DEVICE
            )
            scores = model(X).cpu().numpy()

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


# =====================================================================
# 8. Main
# =====================================================================

def main():
    tee = Tee(RESULTS_FILE)
    sys.stdout = tee
    start_time = time.time()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    user_ratings = parse_training(TRAIN_FILE)
    track_meta = parse_tracks(TRACK_FILE)
    album_meta = parse_albums(ALBUM_FILE)
    test_users = parse_test(TEST_FILE)

    print("\n" + "=" * 60)
    print("  ENRICHING TRACKS")
    print("=" * 60)
    enriched_meta = enrich_tracks_with_albums(track_meta, album_meta)

    # ---- Build pairwise training data ----
    print("\n" + "=" * 60)
    print("  BUILDING PAIRWISE TRAINING DATA")
    print("=" * 60)
    X_pos, X_neg = build_pairwise_data(user_ratings, enriched_meta,
                                        pairs_per_user=100)

    # ---- Train model ----
    print("\n" + "=" * 60)
    print("  TRAINING RANKING MODEL")
    print("=" * 60)
    model = train_model(X_pos, X_neg, n_epochs=30, batch_size=4096,
                        lr=1e-3, margin=1.0)

    # ---- Validation ----
    print("\n" + "=" * 60)
    print("  VALIDATION")
    print("=" * 60)
    v2_auc = validate_v2(user_ratings, enriched_meta)
    nn_auc = validate(model, user_ratings, enriched_meta)

    print(f"\n  v2 heuristic:    {v2_auc:.4f}  (Kaggle: 0.871)")
    print(f"  Pairwise NN:     {nn_auc:.4f}")

    # ---- Also train a wider/deeper variant ----
    print("\n" + "=" * 60)
    print("  TRAINING LARGER MODEL")
    print("=" * 60)
    model_large = train_model(X_pos, X_neg, n_epochs=50, batch_size=8192,
                              lr=5e-4, margin=0.5)
    nn_large_auc = validate(model_large, user_ratings, enriched_meta)
    print(f"  Larger model:    {nn_large_auc:.4f}")

    # ---- Submissions ----
    print("\n" + "=" * 60)
    print("  GENERATING SUBMISSIONS")
    print("=" * 60)

    predict_and_submit(model, user_ratings, enriched_meta, test_users,
                       os.path.join(OUTPUT_DIR, "submission_v4_pairwise.csv"),
                       "Pairwise NN")

    predict_and_submit(model_large, user_ratings, enriched_meta, test_users,
                       os.path.join(OUTPUT_DIR, "submission_v4_pairwise_lg.csv"),
                       "Pairwise NN (large)")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  v2 heuristic val AUC:     {v2_auc:.4f}  (Kaggle: 0.871)")
    print(f"  Pairwise NN val AUC:      {nn_auc:.4f}")
    print(f"  Pairwise NN (lg) val AUC: {nn_large_auc:.4f}")
    print()
    print("  Files:")
    print("    submission_v4_pairwise.csv     - Pairwise ranking NN")
    print("    submission_v4_pairwise_lg.csv  - Larger pairwise NN")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print("\nDone.")

    tee.close()


if __name__ == "__main__":
    main()

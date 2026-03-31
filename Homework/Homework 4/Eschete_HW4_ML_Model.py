"""
EE627 Homework 4 - Yahoo Music Recommendation (Scikit-learn)
Jude Eschete

Trains a Scikit-learn gradient boosted classifier on hand-crafted features
and writes `eschete_submission.csv` in Kaggle format.

Why this version:
- Uses Scikit-learn end-to-end (no PyTorch/CUDA dependency)
- Prints frequent progress updates during long steps
- Supports quick smoke tests and full runs from the terminal

Examples:
  python HW4_ML_Model.py
  python HW4_ML_Model.py --quick
  python HW4_ML_Model.py --continuous
"""

import argparse
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "eschete_submission.csv")

# --- Defaults ---
RANDOM_SEED = 42
DEFAULT_N_USERS = 15000
DEFAULT_NEG_PER_POS = 1
DEFAULT_MAX_ITER = 350

# CF helpers
MAX_TOP_TRACKS_FOR_CF = 30
MAX_SIMILAR_USERS = 50000

FEATURE_NAMES = [
    "track_log_pop",
    "track_norm_pop",
    "track_avg_rating",
    "track_raw_pop",
    "track_num_genres",
    "artist_popularity",
    "album_popularity",
    "user_avg_rating",
    "user_log_num_ratings",
    "artist_match",
    "album_match",
    "genre_pref_sum",
    "genre_pref_max",
    "genre_overlap_count",
    "genre_overlap_ratio",
    "genre_pref_frac",
    "user_num_artists_log",
    "user_tracks_by_artist_ratio",
    "cf_overlap_count_log",
    "cf_overlap_ratio",
    "cf_coverage",
]


def log(msg):
    print(msg, flush=True)


def report_progress(label, current, total, start_time):
    elapsed = time.time() - start_time
    pct = (100.0 * current / total) if total else 0.0
    rate = current / elapsed if elapsed > 0 else 0.0
    eta = (total - current) / rate if rate > 0 else float("inf")
    eta_s = f"{eta:.1f}s" if math.isfinite(eta) else "n/a"
    log(
        f"  {label}: {current:,}/{total:,} ({pct:5.1f}%) | "
        f"{elapsed:.1f}s elapsed | ETA {eta_s}"
    )


# ===================== Data Parsing =====================


def parse_training(path, report_every_users=5000):
    log("Loading training data...")
    users = {}
    total_ratings = 0
    t0 = time.time()

    with open(path, "r") as f:
        uid = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                uid = int(line.split("|")[0])
                users[uid] = {}
                if len(users) % report_every_users == 0:
                    log(
                        f"  parsed {len(users):,} users | "
                        f"{total_ratings:,} ratings | {time.time() - t0:.1f}s"
                    )
            elif "\t" in line and uid is not None:
                tid, rating = line.split("\t")
                users[uid][int(tid)] = int(rating)
                total_ratings += 1

    log(
        f"  done: {len(users):,} users, {total_ratings:,} ratings "
        f"({time.time() - t0:.1f}s)"
    )
    return users


def parse_tracks(path, report_every_tracks=50000):
    log("Loading track metadata...")
    tracks = {}
    t0 = time.time()

    with open(path, "r") as f:
        for idx, line in enumerate(f, start=1):
            p = line.strip().split("|")
            if len(p) < 3:
                continue
            tid = int(p[0])
            alb = None if p[1] == "None" else int(p[1])
            art = None if p[2] == "None" else int(p[2])
            genres = frozenset(int(g) for g in p[3:] if g != "None" and g.strip())
            tracks[tid] = (alb, art, genres)
            if idx % report_every_tracks == 0:
                log(f"  parsed {idx:,} tracks ({time.time() - t0:.1f}s)")

    log(f"  done: {len(tracks):,} tracks ({time.time() - t0:.1f}s)")
    return tracks


def parse_test(path, report_every_users=2500):
    log("Loading test data...")
    test = []
    t0 = time.time()

    with open(path, "r") as f:
        uid = None
        tids = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                if uid is not None:
                    test.append((uid, tids))
                    if len(test) % report_every_users == 0:
                        log(
                            f"  parsed {len(test):,} test users ({time.time() - t0:.1f}s)"
                        )
                uid = int(line.split("|")[0])
                tids = []
            else:
                tids.append(int(line))

    if uid is not None:
        test.append((uid, tids))

    total_candidates = sum(len(t) for _, t in test)
    log(
        f"  done: {len(test):,} test users, {total_candidates:,} candidates "
        f"({time.time() - t0:.1f}s)"
    )
    return test


# ===================== Stats + Features =====================


def build_stats(user_ratings, track_meta):
    log("\nBuilding global statistics...")
    t0 = time.time()

    track_count = defaultdict(int)
    track_rsum = defaultdict(float)
    track_raters = defaultdict(set)

    user_genres = {}
    user_artists = {}
    user_albums = {}
    user_avg_rating = {}
    user_num_ratings = {}

    total_users = len(user_ratings)
    for idx, (uid, ratings) in enumerate(user_ratings.items(), start=1):
        user_num_ratings[uid] = len(ratings)
        user_avg_rating[uid] = sum(ratings.values()) / (100.0 * len(ratings))

        gp = defaultdict(float)
        arts, albs = set(), set()

        for tid, r in ratings.items():
            track_count[tid] += 1
            track_rsum[tid] += r
            track_raters[tid].add(uid)

            if tid in track_meta:
                alb, art, genres = track_meta[tid]
                if art is not None:
                    arts.add(art)
                if alb is not None:
                    albs.add(alb)
                w = r / 100.0
                for g in genres:
                    gp[g] += w

        user_genres[uid] = dict(gp)
        user_artists[uid] = arts
        user_albums[uid] = albs

        if idx % 5000 == 0 or idx == total_users:
            report_progress("users profiled", idx, total_users, t0)

    artist_pop = defaultdict(int)
    album_pop = defaultdict(int)

    t1 = time.time()
    total_tracks = len(track_meta)
    for idx, (tid, (alb, art, _)) in enumerate(track_meta.items(), start=1):
        pop = track_count.get(tid, 0)
        if pop > 0:
            if art is not None:
                artist_pop[art] += pop
            if alb is not None:
                album_pop[alb] += pop

        if idx % 50000 == 0 or idx == total_tracks:
            report_progress("artist/album popularity", idx, total_tracks, t1)

    max_log_pop = math.log1p(max(track_count.values())) if track_count else 1.0

    log(f"  done building stats in {time.time() - t0:.1f}s")

    return {
        "track_count": track_count,
        "track_rsum": track_rsum,
        "max_log_pop": max_log_pop,
        "user_genres": user_genres,
        "user_artists": user_artists,
        "user_albums": user_albums,
        "user_avg_rating": user_avg_rating,
        "user_num_ratings": user_num_ratings,
        "track_raters": track_raters,
        "artist_pop": artist_pop,
        "album_pop": album_pop,
        "track_meta": track_meta,
        "user_ratings": user_ratings,
    }


def get_similar_users(uid, stats, sim_cache):
    cached = sim_cache.get(uid)
    if cached is not None:
        return cached

    ur = stats["user_ratings"]
    t_raters = stats["track_raters"]
    similar_users = set()

    if uid in ur:
        top_tracks = sorted(ur[uid].items(), key=lambda x: x[1], reverse=True)[
            :MAX_TOP_TRACKS_FOR_CF
        ]
        for tid, _ in top_tracks:
            raters = t_raters.get(tid)
            if raters:
                similar_users.update(raters)
            if len(similar_users) > MAX_SIMILAR_USERS:
                break
        similar_users.discard(uid)

    sim_cache[uid] = similar_users
    return similar_users


def extract_features(uid, tid, stats, sim_cache, similar_users=None):
    tc = stats["track_count"]
    tr = stats["track_rsum"]
    mlp = stats["max_log_pop"]
    ug = stats["user_genres"]
    ua = stats["user_artists"]
    ual = stats["user_albums"]
    uar = stats["user_avg_rating"]
    unr = stats["user_num_ratings"]
    t_raters = stats["track_raters"]
    ap = stats["artist_pop"]
    abp = stats["album_pop"]
    tm = stats["track_meta"]
    ur = stats["user_ratings"]

    pop = tc.get(tid, 0)
    log_pop = math.log1p(pop)
    avg_r = (tr[tid] / pop / 100.0) if pop > 0 else 0.0

    alb, art, genres = tm.get(tid, (None, None, frozenset()))

    gp = ug.get(uid, {})
    arts = ua.get(uid, set())
    albs = ual.get(uid, set())

    feats = [
        log_pop,
        log_pop / mlp if mlp > 0 else 0.0,
        avg_r,
        float(pop),
        float(len(genres)),
        math.log1p(ap.get(art, 0)) if art is not None else 0.0,
        math.log1p(abp.get(alb, 0)) if alb is not None else 0.0,
        uar.get(uid, 0.0),
        math.log1p(unr.get(uid, 0)),
        1.0 if (art is not None and art in arts) else 0.0,
        1.0 if (alb is not None and alb in albs) else 0.0,
    ]

    if genres and gp:
        genre_sum = sum(gp.get(g, 0.0) for g in genres)
        genre_max = max((gp.get(g, 0.0) for g in genres), default=0.0)
        genre_count = sum(1 for g in genres if g in gp)
        total_gp = sum(gp.values())
        feats.extend(
            [
                genre_sum,
                genre_max,
                float(genre_count),
                genre_count / len(genres),
                (genre_sum / total_gp) if total_gp > 0 else 0.0,
            ]
        )
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    feats.append(math.log1p(len(arts)))

    if art is not None and uid in ur:
        artist_track_count = sum(
            1 for t in ur[uid] if tm.get(t, (None, None, frozenset()))[1] == art
        )
        feats.append(artist_track_count / max(1, unr.get(uid, 1)))
    else:
        feats.append(0.0)

    if similar_users is None:
        similar_users = get_similar_users(uid, stats, sim_cache)

    if similar_users and tid in t_raters:
        overlap = len(t_raters[tid] & similar_users)
        feats.extend(
            [
                math.log1p(overlap),
                overlap / len(similar_users),
                overlap / len(t_raters[tid]),
            ]
        )
    else:
        feats.extend([0.0, 0.0, 0.0])

    return feats


# ===================== Dataset Build =====================


def generate_training_samples(user_ratings, stats, n_users, neg_per_pos, seed):
    log("\nGenerating training samples...")
    t0 = time.time()

    rng = random.Random(seed)
    np.random.seed(seed)

    all_track_ids = list(stats["track_meta"].keys())
    eligible = [uid for uid, ratings in user_ratings.items() if len(ratings) >= 8]
    rng.shuffle(eligible)

    n_target = min(n_users, len(eligible))
    sample_uids = eligible[:n_target]
    log(f"  selected {n_target:,} users out of {len(eligible):,} eligible users")

    X, y = [], []
    sim_cache = {}
    skipped = 0

    for idx, uid in enumerate(sample_uids, start=1):
        ratings = user_ratings[uid]
        user_tracks = set(ratings.keys())
        sorted_t = sorted(ratings.items(), key=lambda x: (-x[1], rng.random()))

        positives = [t for t, _ in sorted_t[:3]]
        n_neg = max(3, len(positives) * neg_per_pos)

        negatives = []
        attempts = 0
        while len(negatives) < n_neg and attempts < 5000:
            cand = all_track_ids[rng.randrange(len(all_track_ids))]
            if cand not in user_tracks and cand not in negatives:
                negatives.append(cand)
            attempts += 1

        if len(negatives) < n_neg:
            skipped += 1
            continue

        similar_users = get_similar_users(uid, stats, sim_cache)

        for tid in positives:
            X.append(
                extract_features(
                    uid, tid, stats, sim_cache, similar_users=similar_users
                )
            )
            y.append(1)

        for tid in negatives:
            X.append(
                extract_features(
                    uid, tid, stats, sim_cache, similar_users=similar_users
                )
            )
            y.append(0)

        if idx % 1000 == 0 or idx == n_target:
            report_progress("users sampled", idx, n_target, t0)
            log(f"    samples so far: {len(y):,}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    n_pos = int(y.sum())
    log(
        f"  done: {len(y):,} samples ({n_pos:,} pos, {len(y) - n_pos:,} neg), "
        f"skipped={skipped:,}, time={time.time() - t0:.1f}s"
    )

    return X, y


# ===================== Training =====================


def train_model(X, y, max_iter, seed):
    log("\nTraining HistGradientBoostingClassifier (Scikit-learn)...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    log(f"  train size={len(y_train):,}, val size={len(y_val):,}")

    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0) + 1e-8

    X_train_s = (X_train - feat_mean) / feat_std
    X_val_s = (X_val - feat_mean) / feat_std

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=max_iter,
        max_depth=8,
        min_samples_leaf=40,
        l2_regularization=0.2,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=seed,
        verbose=1,
    )

    t0 = time.time()
    model.fit(X_train_s, y_train)
    fit_time = time.time() - t0

    val_probs = model.predict_proba(X_val_s)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    n_iter = getattr(model, "n_iter_", max_iter)
    log(f"  fit complete in {fit_time:.1f}s | iterations={n_iter}")
    log(f"  validation AUC={val_auc:.4f}")

    return model, feat_mean.astype(np.float32), feat_std.astype(np.float32)


# ===================== Prediction =====================


def score_test_users(model, test_users, stats, feat_mean, feat_std, continuous):
    log("\nScoring test users...")
    t0 = time.time()

    results = []
    sim_cache = {}
    total_users = len(test_users)

    for idx, (uid, candidates) in enumerate(test_users, start=1):
        similar_users = get_similar_users(uid, stats, sim_cache)
        feats = [
            extract_features(uid, tid, stats, sim_cache, similar_users=similar_users)
            for tid in candidates
        ]

        X_user = np.asarray(feats, dtype=np.float32)
        X_user = (X_user - feat_mean) / feat_std
        probs = model.predict_proba(X_user)[:, 1]

        if continuous:
            for tid, prob in zip(candidates, probs):
                results.append((f"{uid}_{tid}", float(prob)))
        else:
            order = np.argsort(-probs)
            top3_idx = set(order[:3].tolist())
            for j, tid in enumerate(candidates):
                results.append((f"{uid}_{tid}", 1 if j in top3_idx else 0))

        if idx % 2000 == 0 or idx == total_users:
            report_progress("test users scored", idx, total_users, t0)

    return results


def write_submission(results, continuous):
    log(f"\nWriting {len(results):,} predictions to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", newline="") as f:
        f.write("TrackID,Predictor\n")
        for key, pred in results:
            if continuous:
                f.write(f"{key},{pred:.6f}\n")
            else:
                f.write(f"{key},{int(pred)}\n")

    if continuous:
        vals = [v for _, v in results]
        log(f"  wrote continuous scores | min={min(vals):.6f}, max={max(vals):.6f}")
    else:
        ones = sum(1 for _, v in results if v == 1)
        zeros = len(results) - ones
        log(f"  wrote binary labels | ones={ones:,}, zeros={zeros:,}")

    log("Done! Upload eschete_submission.csv to Kaggle.")


# ===================== CLI =====================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and predict with Scikit-learn for HW4"
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=DEFAULT_N_USERS,
        help="Number of users to sample for training data generation",
    )
    parser.add_argument(
        "--neg-per-pos",
        type=int,
        default=DEFAULT_NEG_PER_POS,
        help="Negative samples per positive sample",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Max boosting iterations for HistGradientBoosting",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Write continuous probabilities instead of top-3 binary labels",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test mode (smaller training sample and fewer iterations)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    n_users = args.n_users
    max_iter = args.max_iter
    if args.quick:
        n_users = min(n_users, 2500)
        max_iter = min(max_iter, 120)

    random.seed(args.seed)
    np.random.seed(args.seed)

    log("=" * 72)
    log("EE627 HW4 - Scikit-learn Model")
    log(f"  n_users={n_users:,}")
    log(f"  neg_per_pos={args.neg_per_pos}")
    log(f"  max_iter={max_iter}")
    log(f"  continuous_output={args.continuous}")
    log(f"  quick_mode={args.quick}")
    log("=" * 72)

    user_ratings = parse_training(os.path.join(DATA_DIR, "trainItem2.txt"))
    track_meta = parse_tracks(os.path.join(DATA_DIR, "trackData2.txt"))
    test_users = parse_test(os.path.join(DATA_DIR, "testItem2.txt"))

    stats = build_stats(user_ratings, track_meta)
    X, y = generate_training_samples(
        user_ratings,
        stats,
        n_users=n_users,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
    )

    log(f"\nFeature count: {len(FEATURE_NAMES)}")
    log("  " + ", ".join(FEATURE_NAMES))

    model, feat_mean, feat_std = train_model(X, y, max_iter=max_iter, seed=args.seed)
    results = score_test_users(
        model,
        test_users,
        stats,
        feat_mean,
        feat_std,
        continuous=args.continuous,
    )
    write_submission(results, continuous=args.continuous)


if __name__ == "__main__":
    main()

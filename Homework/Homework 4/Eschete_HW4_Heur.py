"""
EE627 Homework 4 - Yahoo Music Recommendation
Jude Eschete

Usage:
  python Eschete_HW4.py --validate   # Tune hyperparameters on held-out data
  python Eschete_HW4.py              # Generate eschete_submission.csv for Kaggle

Approach: Multi-signal scoring combining content-based filtering
(genre, artist, album preferences), collaborative filtering (similar-user
overlap), track popularity, and optionally SVD matrix factorization.
"""

import math
import os
import random
import sys
from collections import defaultdict

# --- File paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "eschete_submission.csv")


# ===================== Data Parsing =====================


def parse_training(path):
    """Parse training file. Format: UserID|NumTracks header, then TrackID\\tRating lines."""
    print("Loading training data...")
    users = {}
    with open(path, "r") as f:
        uid = None
        for line in f:
            line = line.strip()
            if "|" in line:
                uid = int(line.split("|")[0])
                users[uid] = {}
            elif "\t" in line and uid is not None:
                tid, rating = line.split("\t")
                users[uid][int(tid)] = int(rating)
    total = sum(len(r) for r in users.values())
    print(f"  {len(users)} users, {total} ratings")
    return users


def parse_tracks(path):
    """Parse track metadata: TrackId|AlbumId|ArtistId|GenreId_1|...|GenreId_k"""
    print("Loading track metadata...")
    tracks = {}
    with open(path, "r") as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) < 3:
                continue
            tid = int(p[0])
            alb = None if p[1] == "None" else int(p[1])
            art = None if p[2] == "None" else int(p[2])
            genres = frozenset(int(g) for g in p[3:] if g != "None" and g.strip())
            tracks[tid] = (alb, art, genres)
    print(f"  {len(tracks)} tracks")
    return tracks


def parse_test(path):
    """Parse test file. Format: UserID|NumTracks header, then TrackID lines."""
    print("Loading test data...")
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
    print(f"  {len(test)} test users, {sum(len(t) for _, t in test)} candidates")
    return test


# ===================== Model Building =====================


def build_model(user_ratings, track_meta, extra_tids=None, svd_k=50, use_svd=True):
    """Build all model components from training data.

    Returns a dict with all precomputed structures needed for scoring.
    """
    # ---- Track popularity ----
    print("  Computing track statistics...")
    track_count = defaultdict(int)
    track_rsum = defaultdict(float)
    for ratings in user_ratings.values():
        for tid, r in ratings.items():
            track_count[tid] += 1
            track_rsum[tid] += r
    max_log_pop = math.log1p(max(track_count.values())) if track_count else 1.0

    # ---- SVD ----
    U_sigma = None
    Vt = None
    uid_map = {}
    tid_map = {}
    user_means = None

    if use_svd:
        try:
            import numpy as np
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import svds
        except ImportError as exc:
            raise ImportError(
                "SVD mode requires numpy and scipy. Install them or set use_svd=False."
            ) from exc

        user_list = sorted(user_ratings.keys())
        uid_map = {u: i for i, u in enumerate(user_list)}

        all_tids = set()
        for ratings in user_ratings.values():
            all_tids.update(ratings.keys())
        if extra_tids:
            all_tids.update(extra_tids)
        track_list = sorted(all_tids)
        tid_map = {t: j for j, t in enumerate(track_list)}

        n_users, n_items = len(user_list), len(track_list)
        print(f"  Matrix: {n_users} x {n_items}, SVD k={svd_k}")

        rows, cols, vals = [], [], []
        user_means = np.zeros(n_users, dtype=np.float32)
        for uid, ratings in user_ratings.items():
            ui = uid_map[uid]
            mean_r = sum(ratings.values()) / len(ratings)
            user_means[ui] = mean_r
            for tid, r in ratings.items():
                rows.append(ui)
                cols.append(tid_map[tid])
                vals.append(r - mean_r)

        R = csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(n_users, n_items),
        )
        print("  Running SVD...")
        U, sigma, Vt = svds(R, k=svd_k)
        idx = np.argsort(-sigma)
        U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]
        U_sigma = U * sigma[np.newaxis, :]
        print("  SVD complete.")

    # ---- User content profiles ----
    print("  Building content profiles...")
    user_genres = {}
    user_artists = {}
    user_albums = {}
    for uid, ratings in user_ratings.items():
        gp = defaultdict(float)
        arts, albs = set(), set()
        for tid, r in ratings.items():
            if tid in track_meta:
                alb, art, genres = track_meta[tid]
                w = r / 100.0
                for g in genres:
                    gp[g] += w
                if art is not None:
                    arts.add(art)
                if alb is not None:
                    albs.add(alb)
        user_genres[uid] = dict(gp)
        user_artists[uid] = arts
        user_albums[uid] = albs

    # ---- Inverted index for CF ----
    print("  Building CF index...")
    track_raters = defaultdict(set)
    for uid, ratings in user_ratings.items():
        for tid in ratings:
            track_raters[tid].add(uid)

    return {
        "track_count": track_count,
        "track_rsum": track_rsum,
        "max_log_pop": max_log_pop,
        "uid_map": uid_map,
        "tid_map": tid_map,
        "user_means": user_means,
        "U_sigma": U_sigma,
        "Vt": Vt,
        "user_genres": user_genres,
        "user_artists": user_artists,
        "user_albums": user_albums,
        "track_raters": track_raters,
        "user_ratings": user_ratings,
        "track_meta": track_meta,
    }


# ===================== Scoring =====================


def score_users(model, test_users, weights, output_continuous=False):
    """Score test users' candidates. Returns list of (key, label_or_score).

    weights = (w_svd, w_content, w_cf)
    If output_continuous=True, outputs raw combined score instead of binary 0/1.
    """
    w_svd, w_cnt, w_cf = weights

    tc = model["track_count"]
    mlp = model["max_log_pop"]
    uid_map = model["uid_map"]
    tid_map = model["tid_map"]
    um_arr = model["user_means"]
    U_sigma = model["U_sigma"]
    Vt = model["Vt"]
    ug = model["user_genres"]
    ua = model["user_artists"]
    ual = model["user_albums"]
    t_raters = model["track_raters"]
    u_ratings = model["user_ratings"]
    tm = model["track_meta"]
    has_svd = U_sigma is not None

    results = []

    for uid, candidates in test_users:
        gp = ug.get(uid, {})
        arts = ua.get(uid, set())
        albs = ual.get(uid, set())

        # --- SVD signal ---
        svd_scores = [0.0] * len(candidates)
        if has_svd and uid in uid_map:
            ui = uid_map[uid]
            um = float(um_arr[ui])
            for j, tid in enumerate(candidates):
                if tid in tid_map:
                    ji = tid_map[tid]
                    svd_scores[j] = float(um + U_sigma[ui] @ Vt[:, ji])
                else:
                    svd_scores[j] = um

        # --- Content signal ---
        content_scores = []
        for tid in candidates:
            cs = 0.0
            # Popularity
            if tid in tc:
                cs += 0.2 * math.log1p(tc[tid]) / mlp
            # Metadata match
            if tid in tm:
                alb, art, genres = tm[tid]
                if genres and gp:
                    cs += sum(gp.get(g, 0.0) for g in genres)
                if art is not None and art in arts:
                    cs += 5.0
                if alb is not None and alb in albs:
                    cs += 3.0
            content_scores.append(cs)

        # --- CF signal ---
        similar_users = set()
        if uid in u_ratings:
            top_tracks = sorted(
                u_ratings[uid].items(), key=lambda x: x[1], reverse=True
            )[:30]
            for t, _ in top_tracks:
                if t in t_raters:
                    similar_users.update(t_raters[t])
                if len(similar_users) > 50000:
                    break
            similar_users.discard(uid)

        cf_scores = []
        for tid in candidates:
            cf = 0.0
            if similar_users and tid in t_raters:
                cf = len(t_raters[tid] & similar_users) / len(similar_users)
            cf_scores.append(cf)

        # --- Normalize each to [0,1] within this user's candidates ---
        def norm01(vals):
            mn, mx = min(vals), max(vals)
            if mx > mn:
                return [(v - mn) / (mx - mn) for v in vals]
            return [0.5] * len(vals)

        svd_n = norm01(svd_scores)
        cnt_n = norm01(content_scores)
        cf_n = norm01(cf_scores)

        # --- Combine ---
        combined = []
        for j in range(len(candidates)):
            s = w_svd * svd_n[j] + w_cnt * cnt_n[j] + w_cf * cf_n[j]
            combined.append((candidates[j], s))

        # Sort and assign
        combined.sort(key=lambda x: x[1], reverse=True)
        top3 = set(c[0] for c in combined[:3])

        if output_continuous:
            for tid in candidates:
                score = next(s for t, s in combined if t == tid)
                results.append((f"{uid}_{tid}", score))
        else:
            for tid in candidates:
                results.append((f"{uid}_{tid}", 1 if tid in top3 else 0))

    return results


# ===================== Validation =====================


def run_validation(all_ratings, track_meta):
    """Hold out users with negative sampling to simulate Kaggle test scenario.

    For each validation user:
      - 3 positives: tracks the user rated (held out from training)
      - 3 negatives: random tracks the user NEVER rated (likely not interested)
    This matches Kaggle's setup where candidates mix relevant and irrelevant tracks.
    """
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError("Validation mode requires numpy and scikit-learn.") from exc

    random.seed(42)
    np.random.seed(42)

    # Build pool of all track IDs for negative sampling
    all_track_ids = list(
        set(tid for ratings in all_ratings.values() for tid in ratings)
    )

    # Find users with enough ratings
    eligible = [uid for uid, r in all_ratings.items() if len(r) >= 10]
    random.shuffle(eligible)
    n_val = 5000
    val_uids = set(eligible[:n_val])

    # Create simulated test set with negative sampling
    val_test = []  # [(uid, [candidate_tids])]
    val_truth = {}  # "uid_tid" -> ground truth label
    train_ratings = {}

    for uid, ratings in all_ratings.items():
        if uid in val_uids:
            user_tracks = set(ratings.keys())
            sorted_t = sorted(ratings.items(), key=lambda x: (-x[1], random.random()))

            # 3 positives: hold out top-rated tracks
            positives = [t for t, r in sorted_t[:3]]

            # 3 negatives: random tracks this user never rated
            negatives = []
            attempts = 0
            while len(negatives) < 3 and attempts < 1000:
                candidate = random.choice(all_track_ids)
                if candidate not in user_tracks and candidate not in negatives:
                    negatives.append(candidate)
                attempts += 1

            if len(negatives) < 3:
                # Fallback: skip this user
                train_ratings[uid] = ratings
                continue

            cands = positives + negatives
            labels = [1] * 3 + [0] * 3

            # Shuffle
            combined = list(zip(cands, labels))
            random.shuffle(combined)
            cands = [c for c, _ in combined]
            labels = [la for _, la in combined]

            val_test.append((uid, cands))
            for t, la in zip(cands, labels):
                val_truth[f"{uid}_{t}"] = la

            # Remove only the 3 positive tracks from training (negatives were never there)
            held_out = set(positives)
            remaining = {t: r for t, r in ratings.items() if t not in held_out}
            if remaining:
                train_ratings[uid] = remaining
        else:
            train_ratings[uid] = ratings

    print(f"Validation: {len(val_test)} users with negative sampling")
    print(f"Training: {len(train_ratings)} users retained\n")

    # Collect all candidate tids for the SVD matrix index
    extra_tids = set()
    for _, cands in val_test:
        extra_tids.update(cands)

    # Build models
    print("Building model with SVD (k=50)...")
    model_svd50 = build_model(
        train_ratings, track_meta, extra_tids=extra_tids, svd_k=50, use_svd=True
    )
    print("\nBuilding model with SVD (k=100)...")
    model_svd100 = build_model(
        train_ratings, track_meta, extra_tids=extra_tids, svd_k=100, use_svd=True
    )
    print("\nBuilding model without SVD...")
    model_nosvd = build_model(train_ratings, track_meta, use_svd=False)

    # ---- Test configurations ----
    configs = [
        # Content-based approaches (what got 0.798 on Kaggle)
        (model_nosvd, (0.00, 0.60, 0.40), False, "Cnt=0.60+CF=0.40 (binary)"),
        (model_nosvd, (0.00, 0.60, 0.40), True, "Cnt=0.60+CF=0.40 (continuous)"),
        (model_nosvd, (0.00, 0.70, 0.30), False, "Cnt=0.70+CF=0.30 (binary)"),
        (model_nosvd, (0.00, 0.70, 0.30), True, "Cnt=0.70+CF=0.30 (continuous)"),
        (model_nosvd, (0.00, 0.80, 0.20), True, "Cnt=0.80+CF=0.20 (continuous)"),
        (model_nosvd, (0.00, 1.00, 0.00), False, "Content only (binary)"),
        (model_nosvd, (0.00, 1.00, 0.00), True, "Content only (continuous)"),
        (model_nosvd, (0.00, 0.00, 1.00), True, "CF only (continuous)"),
        # SVD k=50 blends
        (model_svd50, (1.00, 0.00, 0.00), False, "SVD50 only (binary)"),
        (model_svd50, (1.00, 0.00, 0.00), True, "SVD50 only (continuous)"),
        (model_svd50, (0.20, 0.50, 0.30), True, "SVD50=0.20+Cnt=0.50+CF=0.30 (cont)"),
        (model_svd50, (0.30, 0.45, 0.25), True, "SVD50=0.30+Cnt=0.45+CF=0.25 (cont)"),
        (model_svd50, (0.40, 0.40, 0.20), True, "SVD50=0.40+Cnt=0.40+CF=0.20 (cont)"),
        (model_svd50, (0.50, 0.30, 0.20), True, "SVD50=0.50+Cnt=0.30+CF=0.20 (cont)"),
        # SVD k=100 blends
        (model_svd100, (1.00, 0.00, 0.00), True, "SVD100 only (continuous)"),
        (model_svd100, (0.20, 0.50, 0.30), True, "SVD100=0.20+Cnt=0.50+CF=0.30 (cont)"),
        (model_svd100, (0.30, 0.45, 0.25), True, "SVD100=0.30+Cnt=0.45+CF=0.25 (cont)"),
        (model_svd100, (0.40, 0.40, 0.20), True, "SVD100=0.40+Cnt=0.40+CF=0.20 (cont)"),
        (model_svd100, (0.50, 0.30, 0.20), True, "SVD100=0.50+Cnt=0.30+CF=0.20 (cont)"),
    ]

    print("\n" + "=" * 65)
    print(f"{'Configuration':<48} {'AUC-ROC':>8}")
    print("=" * 65)

    best_auc = 0
    best_label = ""
    results_table = []

    for mdl, weights, continuous, label in configs:
        results = score_users(mdl, val_test, weights, output_continuous=continuous)
        y_true = [val_truth[key] for key, _ in results]
        y_score = [score for _, score in results]
        auc = roc_auc_score(y_true, y_score)

        marker = " ***" if auc > best_auc else ""
        print(f"  {label:<46} {auc:.4f}{marker}")
        results_table.append((label, auc))

        if auc > best_auc:
            best_auc = auc
            best_label = label

    print("=" * 65)
    print(f"\nBest: {best_label} -> AUC = {best_auc:.4f}")
    print(
        "\nUpdate WEIGHTS, USE_SVD, and CONTINUOUS in the script, then run without --validate."
    )


# ===================== Prediction =====================


def run_prediction(all_ratings, track_meta, test_users, weights, use_svd, continuous):
    """Generate full Kaggle submission."""
    extra_tids = set()
    for _, cands in test_users:
        extra_tids.update(cands)

    print("Building model...")
    model = build_model(
        all_ratings,
        track_meta,
        extra_tids=extra_tids,
        svd_k=50,
        use_svd=use_svd,
    )

    print("Scoring candidates...")
    results = score_users(model, test_users, weights, output_continuous=continuous)

    print(f"Writing {len(results)} predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="") as f:
        f.write("TrackID,Predictor\n")
        for key, val in results:
            f.write(f"{key},{val}\n")

    if not continuous:
        ones = sum(1 for _, v in results if v == 1)
        zeros = sum(1 for _, v in results if v == 0)
        print(f"\nDone! {len(results)} predictions ({ones} recommend, {zeros} don't)")
    else:
        print(f"\nDone! {len(results)} continuous-score predictions")
    print("Saved to eschete_submission.csv")


# ===================== Main =====================

if __name__ == "__main__":
    all_ratings = parse_training(os.path.join(DATA_DIR, "trainItem2.txt"))
    track_meta = parse_tracks(os.path.join(DATA_DIR, "trackData2.txt"))

    if "--validate" in sys.argv:
        run_validation(all_ratings, track_meta)
    else:
        test_users = parse_test(os.path.join(DATA_DIR, "testItem2.txt"))

        # ============================================================
        # CONFIGURATION
        # Safer default: blended content + CF with binary top-3 assignment.
        # This aligns with the assignment rule (3 ones, 3 zeros per user)
        # and avoids overfitting to synthetic validation artifacts.
        # ============================================================
        WEIGHTS = (0.00, 0.60, 0.40)  # (SVD, Content, CF)
        USE_SVD = False
        CONTINUOUS = False
        # ============================================================

        run_prediction(
            all_ratings,
            track_meta,
            test_users,
            weights=WEIGHTS,
            use_svd=USE_SVD,
            continuous=CONTINUOUS,
        )

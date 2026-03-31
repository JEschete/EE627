"""
EE627A Midterm - Part 1: Heuristic-Based Music Recommendation
Jude Eschete

Develops a rule-based recommendation system to predict user preferences
for six candidate tracks. For each test user, exactly three tracks are
labeled '1' (Recommend) and three tracks '0' (Do Not Recommend).

Usage:
    python EE627_Midterm.py
"""

import os

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


# =====================================================================
# 1. Data Parsing
# =====================================================================


def parse_training(path):
    """Parse training file into {user_id: {item_id: rating}}.

    Training data format:
        UserID|NumItems   (header line)
        ItemID<TAB>Rating (one per rated item)

    Items can be tracks, albums, artists, or genres --- all share a single
    global ID space.  Ratings are integers in [0, 100].
    """
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
    """Parse trackData2.txt -> {track_id: (album_id, artist_id, [genre_ids])}.

    Format: TrackId|AlbumId|ArtistId|GenreId_1|...|GenreId_k
    """
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
    """Parse albumData2.txt -> {album_id: (artist_id, [genre_ids])}.

    Format: AlbumId|ArtistId|GenreId_1|...|GenreId_k
    """
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
    """Parse testItem2.txt -> [(user_id, [track_id, ...])].

    Format: UserID|6 (header), then 6 TrackID lines.
    """
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
# 2. Feature Engineering  (Part 1.a)
# =====================================================================


def compute_feature_vector(user_ratings, track_meta, uid, tid):
    """Build the feature vector for one (user, candidate-track) pair.

    For each candidate track we look up the track's hierarchy (Album,
    Artist, Genre list) from trackData2, then search the user's training
    history for direct ratings of those hierarchy IDs.

    Returns dict:
        album_score   - user's rating for the track's album   (0 if absent)
        artist_score  - user's rating for the track's artist  (0 if absent)
        has_album     - 1 if user rated this album, else 0
        has_artist    - 1 if user rated this artist, else 0
        genre_count   - number of the track's genres rated by the user
        genre_max     - highest rating among matched genres
        genre_min     - lowest  rating among matched genres
        genre_mean    - mean rating across matched genres
        genre_var     - variance of ratings across matched genres
        genre_sum     - sum of ratings across matched genres
    """
    ratings = user_ratings.get(uid, {})
    alb_id, art_id, genre_ids = track_meta.get(tid, (None, None, []))

    # --- Album score ---
    if alb_id is not None and alb_id in ratings:
        album_score, has_album = ratings[alb_id], 1
    else:
        album_score, has_album = 0, 0

    # --- Artist score ---
    if art_id is not None and art_id in ratings:
        artist_score, has_artist = ratings[art_id], 1
    else:
        artist_score, has_artist = 0, 0

    # --- Genre scores (statistical aggregation) ---
    genre_scores = [ratings[g] for g in genre_ids if g in ratings]
    n = len(genre_scores)

    if n > 0:
        g_max  = max(genre_scores)
        g_min  = min(genre_scores)
        g_mean = sum(genre_scores) / n
        g_var  = sum((s - g_mean) ** 2 for s in genre_scores) / n
        g_sum  = sum(genre_scores)
    else:
        g_max = g_min = g_sum = 0
        g_mean = g_var = 0.0

    return {
        "album_score":  album_score,
        "artist_score": artist_score,
        "has_album":    has_album,
        "has_artist":   has_artist,
        "genre_count":  n,
        "genre_max":    g_max,
        "genre_min":    g_min,
        "genre_mean":   g_mean,
        "genre_var":    g_var,
        "genre_sum":    g_sum,
    }


# =====================================================================
# 3. Heuristic Strategies  (Part 1.b)
# =====================================================================


def strategy_max_genre(features_list):
    """Strategy 1 -- Maximum Genre Score.

    Composite = 0.30 * album + 0.20 * artist + 0.50 * genre_max

    Rationale
    ---------
    Genre preference is a fundamental taste indicator.  If a user gives a
    high rating to *any* genre associated with a track, the track is likely
    to resonate with that user.  Album and artist scores add specificity.
    The maximum genre score captures the "best-case" genre alignment.
    """
    return [
        0.30 * f["album_score"]
        + 0.20 * f["artist_score"]
        + 0.50 * f["genre_max"]
        for f in features_list
    ]


def strategy_weighted_avg(features_list):
    """Strategy 2 -- Weighted Average (Album + Artist + Mean-Genre).

    Composite = 0.40 * album + 0.35 * artist + 0.25 * genre_mean

    Rationale
    ---------
    Balances the three hierarchy levels.  Album score receives the highest
    weight because it is the most granular personalized signal (same album
    implies high content similarity).  Artist affinity is next, and average
    genre preference provides a broad taste signal.
    """
    return [
        0.40 * f["album_score"]
        + 0.35 * f["artist_score"]
        + 0.25 * f["genre_mean"]
        for f in features_list
    ]


def strategy_evidence_weighted(features_list):
    """Strategy 3 -- Evidence-Weighted Composite.

    Composite = has_album * album + has_artist * artist
                + genre_count * genre_mean / 10

    Rationale
    ---------
    Addresses data sparsity.  A zero album score for an un-rated album is
    absence-of-evidence, not evidence-of-absence.  This strategy only
    counts a score when the user *actually* rated that hierarchy item.
    Genre breadth (count * mean) rewards tracks with broad genre overlap.
    """
    return [
        f["has_album"]  * f["album_score"]
        + f["has_artist"] * f["artist_score"]
        + f["genre_count"] * f["genre_mean"] / 10.0
        for f in features_list
    ]


# =====================================================================
# 4. Ranking & Submission Output
# =====================================================================


def rank_top3(scores, track_ids):
    """Return {tid: 1|0} picking the top-3 scored tracks as '1'."""
    paired = sorted(zip(scores, track_ids), key=lambda x: x[0], reverse=True)
    top3 = set(tid for _, tid in paired[:3])
    return {tid: (1 if tid in top3 else 0) for tid in track_ids}


def write_submission(results, path):
    """Write Kaggle-format CSV: TrackID,Predictor."""
    with open(path, "w") as f:
        f.write("TrackID,Predictor\n")
        for key, label in results:
            f.write(f"{key},{label}\n")


# =====================================================================
# 5. Pipeline
# =====================================================================


def run_strategy(user_ratings, track_meta, test_users, strategy_fn, out_path):
    """Run one heuristic strategy end-to-end.

    Returns (results_list, first_user_example).
    """
    name = strategy_fn.__name__
    print(f"\n>>> Running {name} ...")

    results = []
    first_example = None

    for idx, (uid, candidates) in enumerate(test_users):
        feats  = [compute_feature_vector(user_ratings, track_meta, uid, tid)
                   for tid in candidates]
        scores = strategy_fn(feats)
        recs   = rank_top3(scores, candidates)

        if idx == 0:
            first_example = (uid, candidates, feats, scores)

        for tid in candidates:
            results.append((f"{uid}_{tid}", recs[tid]))

    write_submission(results, out_path)
    ones  = sum(v for _, v in results)
    zeros = len(results) - ones
    print(f"    {len(results):,} predictions  "
          f"({ones:,} recommend, {zeros:,} don't)")
    print(f"    Saved -> {out_path}")
    return results, first_example


def print_feature_table(example, track_meta):
    """Pretty-print feature vectors for the first test user."""
    if example is None:
        return
    uid, cands, feats, scores = example

    print(f"\n{'='*94}")
    print(f"  Feature Vectors  --  User {uid}")
    print(f"{'='*94}")
    print(f"{'TrackID':>8}  {'AlbumID':>8}  {'ArtistID':>8}  "
          f"{'AlbScr':>6}  {'ArtScr':>6}  "
          f"{'G_Cnt':>5}  {'G_Max':>5}  {'G_Min':>5}  "
          f"{'G_Mean':>6}  {'G_Var':>7}  {'Score':>7}")
    print("-" * 94)
    for tid, feat, sc in zip(cands, feats, scores):
        alb, art, _ = track_meta.get(tid, (None, None, []))
        print(f"{tid:>8}  {str(alb):>8}  {str(art):>8}  "
              f"{feat['album_score']:>6}  {feat['artist_score']:>6}  "
              f"{feat['genre_count']:>5}  {feat['genre_max']:>5}  "
              f"{feat['genre_min']:>5}  {feat['genre_mean']:>6.1f}  "
              f"{feat['genre_var']:>7.1f}  {sc:>7.1f}")
    print()


# =====================================================================
# 6. Main
# =====================================================================


def main():
    # ---- Load all data ----
    user_ratings = parse_training(TRAIN_FILE)
    track_meta   = parse_tracks(TRACK_FILE)
    parse_albums(ALBUM_FILE)     # loaded for completeness / Part 2
    test_users   = parse_test(TEST_FILE)

    # ---- Strategy 1: Max Genre Score ----
    res1, ex1 = run_strategy(
        user_ratings, track_meta, test_users,
        strategy_max_genre,
        os.path.join(OUTPUT_DIR, "submission_strategy1_maxgenre.csv"),
    )
    print_feature_table(ex1, track_meta)

    # ---- Strategy 2: Weighted Average ----
    res2, _ = run_strategy(
        user_ratings, track_meta, test_users,
        strategy_weighted_avg,
        os.path.join(OUTPUT_DIR, "submission_strategy2_weightedavg.csv"),
    )

    # ---- Strategy 3: Evidence-Weighted ----
    res3, _ = run_strategy(
        user_ratings, track_meta, test_users,
        strategy_evidence_weighted,
        os.path.join(OUTPUT_DIR, "submission_strategy3_evidence.csv"),
    )

    # ---- Cross-strategy comparison ----
    print(f"\n{'='*60}")
    print("  Cross-Strategy Agreement")
    print(f"{'='*60}")
    total = len(res1)
    for na, nb, ra, rb in [
        ("MaxGenre",    "WeightedAvg", res1, res2),
        ("MaxGenre",    "Evidence",    res1, res3),
        ("WeightedAvg", "Evidence",    res2, res3),
    ]:
        agree = sum(1 for (_, a), (_, b) in zip(ra, rb) if a == b)
        print(f"  {na:>12} vs {nb:<12}:  "
              f"{agree:,}/{total:,}  ({100*agree/total:.1f}%)")

    # ---- Data-sparsity analysis ----
    print(f"\n{'='*60}")
    print("  Data-Sparsity Analysis (all test candidates)")
    print(f"{'='*60}")
    n_total = n_album = n_artist = n_genre = 0
    gc_list = []

    for uid, candidates in test_users:
        for tid in candidates:
            feat = compute_feature_vector(user_ratings, track_meta, uid, tid)
            n_total  += 1
            n_album  += feat["has_album"]
            n_artist += feat["has_artist"]
            if feat["genre_count"] > 0:
                n_genre += 1
            gc_list.append(feat["genre_count"])

    print(f"  Total candidates:      {n_total:,}")
    print(f"  Have album rating:     {n_album:,}  "
          f"({100*n_album/n_total:.1f}%)")
    print(f"  Have artist rating:    {n_artist:,}  "
          f"({100*n_artist/n_total:.1f}%)")
    print(f"  Have >=1 genre rating: {n_genre:,}  "
          f"({100*n_genre/n_total:.1f}%)")
    print(f"  Avg genre matches:     {sum(gc_list)/len(gc_list):.2f}")

    print("\nDone. Submission files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""
EE627A Midterm - Heuristic-Based Music Recommendation
Jude Eschete

Part 1: Rule-based recommendation via feature engineering on the
        Album / Artist / Genre hierarchy.
Part 2: Cold-start handling with Global Fallback and Dig-Deeper
        intra-album search logic.

Usage:
    python EE627_Midterm.py
"""

import os
from collections import defaultdict

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
# 6. Part 2 — Cold Start: Global Fallback & Dig Deeper
# =====================================================================


def build_album_to_tracks(track_meta):
    """Build {album_id: [track_ids]} from track metadata.

    Used by the "Dig Deeper" fallback to find sibling tracks that share
    the same album as a candidate track.
    """
    album_tracks = defaultdict(list)
    for tid, (alb, _, _) in track_meta.items():
        if alb is not None:
            album_tracks[alb].append(tid)
    print(f"       Album->Tracks index: {len(album_tracks):,} albums")
    return dict(album_tracks)


def build_global_stats(user_ratings):
    """Compute global popularity: mean rating & rater count per item.

    Iterates every (user, item, rating) triple in the training set and
    aggregates by item_id.  Works for tracks, albums, artists, and genres
    since they share one ID space.

    Returns {item_id: (mean_rating, num_raters)}.
    """
    item_sum = defaultdict(float)
    item_cnt = defaultdict(int)
    for _, ratings in user_ratings.items():
        for item_id, rating in ratings.items():
            item_sum[item_id] += rating
            item_cnt[item_id] += 1
    stats = {iid: (item_sum[iid] / item_cnt[iid], item_cnt[iid])
             for iid in item_sum}
    print(f"       Global stats for {len(stats):,} items")
    return stats


def compute_feature_vector_v2(user_ratings, track_meta, uid, tid,
                              album_to_tracks, global_stats,
                              use_dig_deeper=True, use_global=True):
    """Enhanced feature vector with hierarchical cold-start fallback.

    Album-score priority:
        1. Primary   — direct album rating from user's history
        2. Dig Deeper — mean rating of sibling tracks in same album
        3. Global    — global average rating for this album

    Artist and genre scores follow a simpler two-level fallback:
        1. Primary   — direct rating
        2. Global    — global average

    Returns the same dict as compute_feature_vector() plus:
        album_source  : 'direct' | 'dig_deeper' | 'global' | 'none'
        artist_source : 'direct' | 'global' | 'none'
    """
    ratings = user_ratings.get(uid, {})
    alb_id, art_id, genre_ids = track_meta.get(tid, (None, None, []))

    # --- Album score (hierarchical fallback) ---
    album_score, has_album, album_source = 0, 0, "none"

    if alb_id is not None:
        if alb_id in ratings:
            # 1. Primary: direct album rating
            album_score = ratings[alb_id]
            has_album = 1
            album_source = "direct"
        elif use_dig_deeper:
            # 2. Dig Deeper: sibling tracks in the same album
            siblings = album_to_tracks.get(alb_id, [])
            sib_ratings = [ratings[st] for st in siblings
                           if st in ratings and st != tid]
            if sib_ratings:
                album_score = sum(sib_ratings) / len(sib_ratings)
                has_album = 1
                album_source = "dig_deeper"
            elif use_global and alb_id in global_stats:
                # 3. Global: average across all users
                album_score = global_stats[alb_id][0]
                has_album = 1
                album_source = "global"
        elif use_global and alb_id in global_stats:
            # Global only (Dig Deeper disabled)
            album_score = global_stats[alb_id][0]
            has_album = 1
            album_source = "global"

    # --- Artist score (direct → global) ---
    artist_score, has_artist, artist_source = 0, 0, "none"

    if art_id is not None:
        if art_id in ratings:
            artist_score = ratings[art_id]
            has_artist = 1
            artist_source = "direct"
        elif use_global and art_id in global_stats:
            artist_score = global_stats[art_id][0]
            has_artist = 1
            artist_source = "global"

    # --- Genre scores (direct → global per genre) ---
    genre_scores = []
    for gid in genre_ids:
        if gid in ratings:
            genre_scores.append(ratings[gid])
        elif use_global and gid in global_stats:
            genre_scores.append(global_stats[gid][0])

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
        "album_source": album_source,
        "artist_source": artist_source,
    }


# =====================================================================
# 7. Part 2 Pipeline
# =====================================================================


def run_strategy_v2(user_ratings, track_meta, test_users, strategy_fn,
                    album_to_tracks, global_stats, out_path,
                    use_dig_deeper=True, use_global=True):
    """Run a strategy with the Part 2 enhanced feature vector.

    Returns (results, source_counts) where source_counts tracks how
    often each fallback level was triggered for album scores.
    """
    name = strategy_fn.__name__
    tag = []
    if use_dig_deeper:
        tag.append("DigDeeper")
    if use_global:
        tag.append("Global")
    tag_str = "+".join(tag) if tag else "NoFallback"
    print(f"\n>>> Running {name} [{tag_str}] ...")

    results = []
    src_album = defaultdict(int)
    src_artist = defaultdict(int)

    for uid, candidates in test_users:
        feats = [compute_feature_vector_v2(
                     user_ratings, track_meta, uid, tid,
                     album_to_tracks, global_stats,
                     use_dig_deeper=use_dig_deeper,
                     use_global=use_global)
                 for tid in candidates]
        scores = strategy_fn(feats)
        recs = rank_top3(scores, candidates)

        for feat in feats:
            src_album[feat["album_source"]] += 1
            src_artist[feat["artist_source"]] += 1

        for tid in candidates:
            results.append((f"{uid}_{tid}", recs[tid]))

    write_submission(results, out_path)
    ones = sum(v for _, v in results)
    zeros = len(results) - ones
    print(f"    {len(results):,} predictions  "
          f"({ones:,} recommend, {zeros:,} don't)")
    print(f"    Saved -> {out_path}")

    total = sum(src_album.values())
    print(f"    Album source breakdown:")
    for src in ["direct", "dig_deeper", "global", "none"]:
        c = src_album.get(src, 0)
        print(f"      {src:>11}: {c:>7,}  ({100*c/total:.1f}%)")
    print(f"    Artist source breakdown:")
    for src in ["direct", "global", "none"]:
        c = src_artist.get(src, 0)
        print(f"      {src:>11}: {c:>7,}  ({100*c/total:.1f}%)")

    return results, dict(src_album)


# =====================================================================
# 8. Main
# =====================================================================


def main():
    # ================================================================
    # Load all data
    # ================================================================
    user_ratings = parse_training(TRAIN_FILE)
    track_meta   = parse_tracks(TRACK_FILE)
    parse_albums(ALBUM_FILE)
    test_users   = parse_test(TEST_FILE)

    # ================================================================
    # PART 1 — Heuristic Strategies (no cold-start handling)
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 1: Heuristic-Based Recommendation")
    print("=" * 60)

    res1, ex1 = run_strategy(
        user_ratings, track_meta, test_users,
        strategy_max_genre,
        os.path.join(OUTPUT_DIR, "submission_strategy1_maxgenre.csv"),
    )
    print_feature_table(ex1, track_meta)

    res2, _ = run_strategy(
        user_ratings, track_meta, test_users,
        strategy_weighted_avg,
        os.path.join(OUTPUT_DIR, "submission_strategy2_weightedavg.csv"),
    )

    res3, _ = run_strategy(
        user_ratings, track_meta, test_users,
        strategy_evidence_weighted,
        os.path.join(OUTPUT_DIR, "submission_strategy3_evidence.csv"),
    )

    # Cross-strategy agreement
    print(f"\n{'='*60}")
    print("  Part 1 Cross-Strategy Agreement")
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

    # Data-sparsity analysis
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

    # ================================================================
    # PART 2 — Cold Start: Global Fallback & Dig Deeper
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 2: Cold Start — Building Fallback Structures")
    print("=" * 60)

    album_to_tracks = build_album_to_tracks(track_meta)
    global_stats    = build_global_stats(user_ratings)

    # --- 2a. Global Fallback Only (no Dig Deeper) ---
    print("\n" + "-" * 60)
    print("  Part 2.a: Global Fallback Only")
    print("-" * 60)

    res_global_wt, _ = run_strategy_v2(
        user_ratings, track_meta, test_users,
        strategy_weighted_avg,
        album_to_tracks, global_stats,
        os.path.join(OUTPUT_DIR, "submission_p2a_global_weightedavg.csv"),
        use_dig_deeper=False, use_global=True,
    )

    res_global_mg, _ = run_strategy_v2(
        user_ratings, track_meta, test_users,
        strategy_max_genre,
        album_to_tracks, global_stats,
        os.path.join(OUTPUT_DIR, "submission_p2a_global_maxgenre.csv"),
        use_dig_deeper=False, use_global=True,
    )

    # --- 2b. Full Hierarchical: Direct → Dig Deeper → Global ---
    print("\n" + "-" * 60)
    print("  Part 2.b: Full Hierarchical (Dig Deeper + Global)")
    print("-" * 60)

    res_hier_wt, src_hier_wt = run_strategy_v2(
        user_ratings, track_meta, test_users,
        strategy_weighted_avg,
        album_to_tracks, global_stats,
        os.path.join(OUTPUT_DIR, "submission_p2b_hier_weightedavg.csv"),
        use_dig_deeper=True, use_global=True,
    )

    res_hier_mg, src_hier_mg = run_strategy_v2(
        user_ratings, track_meta, test_users,
        strategy_max_genre,
        album_to_tracks, global_stats,
        os.path.join(OUTPUT_DIR, "submission_p2b_hier_maxgenre.csv"),
        use_dig_deeper=True, use_global=True,
    )

    run_strategy_v2(
        user_ratings, track_meta, test_users,
        strategy_evidence_weighted,
        album_to_tracks, global_stats,
        os.path.join(OUTPUT_DIR, "submission_p2b_hier_evidence.csv"),
        use_dig_deeper=True, use_global=True,
    )

    # ================================================================
    # Impact Analysis
    # ================================================================
    print("\n" + "=" * 60)
    print("  IMPACT ANALYSIS")
    print("=" * 60)

    # Compare Part 1 baseline vs Part 2 variants (WeightedAvg strategy)
    print("\n  WeightedAvg Strategy — Prediction Changes:")
    for label, res_new in [
        ("Global Only  ", res_global_wt),
        ("Hierarchical ", res_hier_wt),
    ]:
        changed = sum(1 for (_, v1), (_, v2) in zip(res2, res_new) if v1 != v2)
        agree   = total - changed
        print(f"    Part1 vs {label}:  "
              f"{changed:,} changed ({100*changed/total:.1f}%), "
              f"{agree:,} same ({100*agree/total:.1f}%)")

    # Compare Global-only vs Hierarchical
    changed_g_h = sum(1 for (_, a), (_, b)
                      in zip(res_global_wt, res_hier_wt) if a != b)
    print(f"    Global vs Hierarchical:  "
          f"{changed_g_h:,} changed ({100*changed_g_h/total:.1f}%)")

    # Dig Deeper trigger rate
    print(f"\n  Dig-Deeper Trigger Rate (Hierarchical, WeightedAvg):")
    dd_count = src_hier_wt.get("dig_deeper", 0)
    print(f"    Triggered:  {dd_count:,} / {total:,}  "
          f"({100*dd_count/total:.1f}%)")

    # Same for MaxGenre
    print(f"\n  MaxGenre Strategy — Prediction Changes:")
    for label, res_new in [
        ("Global Only  ", res_global_mg),
        ("Hierarchical ", res_hier_mg),
    ]:
        changed = sum(1 for (_, v1), (_, v2) in zip(res1, res_new) if v1 != v2)
        print(f"    Part1 vs {label}:  "
              f"{changed:,} changed ({100*changed/total:.1f}%)")

    dd_mg = src_hier_mg.get("dig_deeper", 0)
    print(f"    Dig-Deeper triggered: {dd_mg:,} / {total:,}  "
          f"({100*dd_mg/total:.1f}%)")

    # ---- Final summary of all submission files ----
    print(f"\n{'='*60}")
    print("  ALL SUBMISSION FILES")
    print(f"{'='*60}")
    subs = [
        ("Part 1 — S1 MaxGenre",            "submission_strategy1_maxgenre.csv"),
        ("Part 1 — S2 WeightedAvg",         "submission_strategy2_weightedavg.csv"),
        ("Part 1 — S3 Evidence",            "submission_strategy3_evidence.csv"),
        ("Part 2a — Global+WeightedAvg",    "submission_p2a_global_weightedavg.csv"),
        ("Part 2a — Global+MaxGenre",       "submission_p2a_global_maxgenre.csv"),
        ("Part 2b — Hier+WeightedAvg",      "submission_p2b_hier_weightedavg.csv"),
        ("Part 2b — Hier+MaxGenre",         "submission_p2b_hier_maxgenre.csv"),
        ("Part 2b — Hier+Evidence",         "submission_p2b_hier_evidence.csv"),
    ]
    for label, fname in subs:
        print(f"  {label:<35} {fname}")

    print("\nDone. All files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

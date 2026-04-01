"""
EE627A Midterm Part 2 v2 - Cold Start with Album-Enriched Metadata
Jude Eschete

Builds on the Part 1 v2 album-enrichment approach:
    1. Enriches track metadata using albumData2.txt (fill missing
       artists, union genre lists).
    2. Applies cold-start fallback strategies on the enriched data:
       - Part 2a: Global Fallback only
       - Part 2b: Hierarchical (Direct -> Dig Deeper -> Global)
    3. Impact analysis comparing Part 1 baseline vs Part 2 variants.

Usage:
    python EE627_Midterm_part2_v2.py
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
    """Parse training file into {user_id: {item_id: rating}}."""
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
    """Parse trackData2.txt -> {track_id: (album_id, artist_id, [genre_ids])}."""
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
    """Parse albumData2.txt -> {album_id: (artist_id, [genre_ids])}."""
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
    """Parse testItem2.txt -> [(user_id, [track_id, ...])]."""
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
# 2. Album-Based Track Enrichment
# =====================================================================


def enrich_tracks_with_albums(tracks, albums):
    """Use album metadata to fill missing artist/genre info on tracks.

    For each track:
      - If the track has no ArtistID but its album does, inherit the
        album's artist.
      - Union the track's genre list with the album's genre list so
        that tracks with sparse genre data gain additional genre IDs.

    Returns a new enriched track dict (original is not modified).
    """
    enriched = {}
    stats = {"artist_filled": 0, "genres_added": 0, "tracks_enriched": 0}

    for tid, (alb, art, genres) in tracks.items():
        new_art = art
        new_genres = list(genres)

        if alb is not None and alb in albums:
            album_art, album_genres = albums[alb]

            # Fill missing artist from album
            if art is None and album_art is not None:
                new_art = album_art
                stats["artist_filled"] += 1

            # Union genres: add album genres not already in track genres
            track_genre_set = set(genres)
            added = 0
            for g in album_genres:
                if g not in track_genre_set:
                    new_genres.append(g)
                    track_genre_set.add(g)
                    added += 1
            if added > 0:
                stats["genres_added"] += added
                stats["tracks_enriched"] += 1

        enriched[tid] = (alb, new_art, new_genres)

    print(f"\n  Album Enrichment Results:")
    print(f"    Artists filled from album:  {stats['artist_filled']:,}")
    print(f"    Tracks with new genres:     {stats['tracks_enriched']:,}")
    print(f"    Total genre IDs added:      {stats['genres_added']:,}")

    return enriched


# =====================================================================
# 3. Cold-Start Structures
# =====================================================================


def build_album_to_tracks(track_meta):
    """Build {album_id: [track_ids]} from track metadata.

    Used by the "Dig Deeper" fallback to find sibling tracks that
    share the same album as a candidate track.
    """
    album_tracks = defaultdict(list)
    for tid, (alb, _, _) in track_meta.items():
        if alb is not None:
            album_tracks[alb].append(tid)
    print(f"       Album->Tracks index: {len(album_tracks):,} albums")
    return dict(album_tracks)


def build_global_stats(user_ratings):
    """Compute global popularity: mean rating & rater count per item.

    Iterates every (user, item, rating) triple in the training set
    and aggregates by item_id.

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


# =====================================================================
# 5. Enhanced Feature Vector with Hierarchical Fallback
# =====================================================================


def compute_feature_vector_v2(user_ratings, track_meta, uid, tid,
                              album_to_tracks, global_stats,
                              use_dig_deeper=True, use_global=True):
    """Enhanced feature vector with hierarchical cold-start fallback.

    Album-score priority:
        1. Primary   -- direct album rating from user's history
        2. Dig Deeper -- mean rating of sibling tracks in same album
        3. Global    -- global average rating for this album

    Artist and genre scores follow a simpler two-level fallback:
        1. Primary   -- direct rating
        2. Global    -- global average

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
            album_score = ratings[alb_id]
            has_album = 1
            album_source = "direct"
        elif use_dig_deeper:
            siblings = album_to_tracks.get(alb_id, [])
            sib_ratings = [ratings[st] for st in siblings
                           if st in ratings and st != tid]
            if sib_ratings:
                album_score = sum(sib_ratings) / len(sib_ratings)
                has_album = 1
                album_source = "dig_deeper"
            elif use_global and alb_id in global_stats:
                album_score = global_stats[alb_id][0]
                has_album = 1
                album_source = "global"
        elif use_global and alb_id in global_stats:
            album_score = global_stats[alb_id][0]
            has_album = 1
            album_source = "global"

    # --- Artist score (direct -> global) ---
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

    # --- Genre scores (direct -> global per genre) ---
    genre_scores = []
    for gid in genre_ids:
        if gid in ratings:
            genre_scores.append(ratings[gid])
        elif use_global and gid in global_stats:
            genre_scores.append(global_stats[gid][0])

    n = len(genre_scores)
    if n > 0:
        g_max = max(genre_scores)
        g_min = min(genre_scores)
        g_mean = sum(genre_scores) / n
        g_var = sum((s - g_mean) ** 2 for s in genre_scores) / n
        g_sum = sum(genre_scores)
    else:
        g_max = g_min = g_sum = 0
        g_mean = g_var = 0.0

    return {
        "album_score": album_score,
        "artist_score": artist_score,
        "has_album": has_album,
        "has_artist": has_artist,
        "genre_count": n,
        "genre_max": g_max,
        "genre_min": g_min,
        "genre_mean": g_mean,
        "genre_var": g_var,
        "genre_sum": g_sum,
        "album_source": album_source,
        "artist_source": artist_source,
    }


# =====================================================================
# 6. Heuristic Strategies
# =====================================================================


def strategy_max_genre(features_list):
    """Strategy 1 -- Maximum Genre Score.

    Composite = 0.30 * album + 0.20 * artist + 0.50 * genre_max
    """
    return [
        0.30 * f["album_score"]
        + 0.20 * f["artist_score"]
        + 0.50 * f["genre_max"]
        for f in features_list
    ]


def strategy_weighted_avg(features_list):
    """Strategy 2 -- Weighted Average.

    Composite = 0.40 * album + 0.35 * artist + 0.25 * genre_mean
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
    """
    return [
        f["has_album"] * f["album_score"]
        + f["has_artist"] * f["artist_score"]
        + f["genre_count"] * f["genre_mean"] / 10.0
        for f in features_list
    ]


# =====================================================================
# 7. Ranking & Submission Output
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
# 8. Part 2 Pipeline (with cold-start fallback)
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
# 10. Main
# =====================================================================


def main():
    # ================================================================
    # Load all data
    # ================================================================
    user_ratings = parse_training(TRAIN_FILE)
    track_meta = parse_tracks(TRACK_FILE)
    album_meta = parse_albums(ALBUM_FILE)
    test_users = parse_test(TEST_FILE)

    # ================================================================
    # Enrich tracks with album metadata
    # ================================================================
    print("\n" + "=" * 60)
    print("  ENRICHING TRACKS WITH ALBUM METADATA")
    print("=" * 60)
    enriched_meta = enrich_tracks_with_albums(track_meta, album_meta)

    # ================================================================
    # Build cold-start structures (on enriched metadata)
    # ================================================================
    print("\n" + "=" * 60)
    print("  PART 2: Building Fallback Structures (Enriched)")
    print("=" * 60)

    album_to_tracks = build_album_to_tracks(enriched_meta)
    global_stats = build_global_stats(user_ratings)

    # ================================================================
    # Part 2a: Global Fallback Only (no Dig Deeper)
    # ================================================================
    print("\n" + "-" * 60)
    print("  Part 2.a: Global Fallback Only")
    print("-" * 60)

    res_global_wa, _ = run_strategy_v2(
        user_ratings, enriched_meta, test_users,
        strategy_weighted_avg,
        album_to_tracks, global_stats,
        os.path.join(OUTPUT_DIR, "submission_p2a_global_weightedavg.csv"),
        use_dig_deeper=False, use_global=True,
    )

    res_global_mg, _ = run_strategy_v2(
        user_ratings, enriched_meta, test_users,
        strategy_max_genre,
        album_to_tracks, global_stats,
        os.path.join(OUTPUT_DIR, "submission_p2a_global_maxgenre.csv"),
        use_dig_deeper=False, use_global=True,
    )

    # ================================================================
    # Part 2b: Full Hierarchical (Direct -> Dig Deeper -> Global)
    # ================================================================
    print("\n" + "-" * 60)
    print("  Part 2.b: Full Hierarchical (Dig Deeper + Global)")
    print("-" * 60)

    res_hier_wa, src_hier_wa = run_strategy_v2(
        user_ratings, enriched_meta, test_users,
        strategy_weighted_avg,
        album_to_tracks, global_stats,
        os.path.join(OUTPUT_DIR, "submission_p2b_hier_weightedavg.csv"),
        use_dig_deeper=True, use_global=True,
    )

    res_hier_mg, src_hier_mg = run_strategy_v2(
        user_ratings, enriched_meta, test_users,
        strategy_max_genre,
        album_to_tracks, global_stats,
        os.path.join(OUTPUT_DIR, "submission_p2b_hier_maxgenre.csv"),
        use_dig_deeper=True, use_global=True,
    )

    res_hier_ev, _ = run_strategy_v2(
        user_ratings, enriched_meta, test_users,
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

    total = len(res_global_wa)

    # Global vs Hierarchical (WeightedAvg)
    changed_g_h_wa = sum(1 for (_, a), (_, b)
                         in zip(res_global_wa, res_hier_wa) if a != b)
    print(f"\n  WeightedAvg: Global vs Hierarchical:  "
          f"{changed_g_h_wa:,} changed ({100*changed_g_h_wa/total:.1f}%)")

    # Global vs Hierarchical (MaxGenre)
    changed_g_h_mg = sum(1 for (_, a), (_, b)
                         in zip(res_global_mg, res_hier_mg) if a != b)
    print(f"  MaxGenre:    Global vs Hierarchical:  "
          f"{changed_g_h_mg:,} changed ({100*changed_g_h_mg/total:.1f}%)")

    # Dig Deeper trigger rates
    print(f"\n  Dig-Deeper Trigger Rate (Hierarchical):")
    dd_wa = src_hier_wa.get("dig_deeper", 0)
    dd_mg = src_hier_mg.get("dig_deeper", 0)
    print(f"    WeightedAvg: {dd_wa:,} / {total:,}  "
          f"({100*dd_wa/total:.1f}%)")
    print(f"    MaxGenre:    {dd_mg:,} / {total:,}  "
          f"({100*dd_mg/total:.1f}%)")

    # Cross-strategy agreement (Part 2b)
    print(f"\n{'='*60}")
    print("  Part 2b Cross-Strategy Agreement")
    print(f"{'='*60}")
    for na, nb, ra, rb in [
        ("Hier+MaxGenre",    "Hier+WeightedAvg", res_hier_mg, res_hier_wa),
        ("Hier+MaxGenre",    "Hier+Evidence",    res_hier_mg, res_hier_ev),
        ("Hier+WeightedAvg", "Hier+Evidence",    res_hier_wa, res_hier_ev),
    ]:
        agree = sum(1 for (_, a), (_, b) in zip(ra, rb) if a == b)
        print(f"  {na:>18} vs {nb:<18}:  "
              f"{agree:,}/{total:,}  ({100*agree/total:.1f}%)")

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print("  PART 2 SUBMISSION FILES")
    print(f"{'='*60}")
    subs = [
        ("Part 2a -- Global+WeightedAvg",
         "submission_p2a_global_weightedavg.csv"),
        ("Part 2a -- Global+MaxGenre",
         "submission_p2a_global_maxgenre.csv"),
        ("Part 2b -- Hier+WeightedAvg",
         "submission_p2b_hier_weightedavg.csv"),
        ("Part 2b -- Hier+MaxGenre",
         "submission_p2b_hier_maxgenre.csv"),
        ("Part 2b -- Hier+Evidence",
         "submission_p2b_hier_evidence.csv"),
    ]
    for label, fname in subs:
        print(f"  {label:<35} {fname}")

    print("\nDone. All files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

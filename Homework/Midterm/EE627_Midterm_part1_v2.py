"""
EE627A Midterm v2 - Album-Enriched Heuristic Recommendation
Jude Eschete

    This version uses album metadata to ENRICH track metadata:
      - Tracks missing an ArtistID inherit it from their album.
      - Tracks missing genre IDs inherit genres from their album.
    This increases feature coverage (more genre/artist matches)
    without introducing global-imputation noise.

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
# 3. Feature Engineering
# =====================================================================


def compute_feature_vector(user_ratings, track_meta, uid, tid):
    """Build feature vector for one (user, candidate-track) pair."""
    ratings = user_ratings.get(uid, {})
    alb_id, art_id, genre_ids = track_meta.get(tid, (None, None, []))

    # Album score
    if alb_id is not None and alb_id in ratings:
        album_score, has_album = ratings[alb_id], 1
    else:
        album_score, has_album = 0, 0

    # Artist score
    if art_id is not None and art_id in ratings:
        artist_score, has_artist = ratings[art_id], 1
    else:
        artist_score, has_artist = 0, 0

    # Genre statistics
    genre_scores = [ratings[g] for g in genre_ids if g in ratings]
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
    }


# =====================================================================
# 4. Heuristic Strategies
# =====================================================================


def strategy_evidence_weighted(features_list):
    """Evidence-Weighted Composite (best from Part 1).

    Composite = has_album*album + has_artist*artist
                + genre_count*genre_mean / 10
    """
    return [
        f["has_album"] * f["album_score"]
        + f["has_artist"] * f["artist_score"]
        + f["genre_count"] * f["genre_mean"] / 10.0
        for f in features_list
    ]


def strategy_max_genre(features_list):
    """Maximum Genre Score.

    Composite = 0.30*album + 0.20*artist + 0.50*genre_max
    """
    return [
        0.30 * f["album_score"]
        + 0.20 * f["artist_score"]
        + 0.50 * f["genre_max"]
        for f in features_list
    ]


def strategy_weighted_avg(features_list):
    """Weighted Average.

    Composite = 0.40*album + 0.35*artist + 0.25*genre_mean
    """
    return [
        0.40 * f["album_score"]
        + 0.35 * f["artist_score"]
        + 0.25 * f["genre_mean"]
        for f in features_list
    ]


# =====================================================================
# 5. Ranking & Submission
# =====================================================================


def rank_top3(scores, track_ids):
    """Return {tid: 1|0} picking the top-3 scored tracks."""
    paired = sorted(zip(scores, track_ids), key=lambda x: x[0], reverse=True)
    top3 = set(tid for _, tid in paired[:3])
    return {tid: (1 if tid in top3 else 0) for tid in track_ids}


def write_submission(results, path):
    """Write Kaggle-format CSV: TrackID,Predictor."""
    with open(path, "w") as f:
        f.write("TrackID,Predictor\n")
        for key, label in results:
            f.write(f"{key},{label}\n")


def run_strategy(user_ratings, track_meta, test_users,
                 strategy_fn, out_path):
    """Run one heuristic strategy end-to-end."""
    name = strategy_fn.__name__
    print(f"\n>>> Running {name} ...")

    results = []
    for uid, candidates in test_users:
        feats = [compute_feature_vector(user_ratings, track_meta, uid, tid)
                 for tid in candidates]
        scores = strategy_fn(feats)
        recs = rank_top3(scores, candidates)
        for tid in candidates:
            results.append((f"{uid}_{tid}", recs[tid]))

    write_submission(results, out_path)
    ones = sum(v for _, v in results)
    zeros = len(results) - ones
    print(f"    {len(results):,} predictions  "
          f"({ones:,} recommend, {zeros:,} don't)")
    print(f"    Saved -> {out_path}")
    return results


# =====================================================================
# 6. Coverage Analysis
# =====================================================================


def coverage_analysis(user_ratings, track_meta, test_users, label=""):
    """Report how many test candidates have album/artist/genre matches."""
    n_total = n_album = n_artist = n_genre = 0
    gc_list = []
    for uid, candidates in test_users:
        for tid in candidates:
            feat = compute_feature_vector(user_ratings, track_meta, uid, tid)
            n_total += 1
            n_album += feat["has_album"]
            n_artist += feat["has_artist"]
            if feat["genre_count"] > 0:
                n_genre += 1
            gc_list.append(feat["genre_count"])
    print(f"\n  Coverage Analysis {label}:")
    print(f"    Total candidates:      {n_total:,}")
    print(f"    Have album rating:     {n_album:,}  ({100*n_album/n_total:.1f}%)")
    print(f"    Have artist rating:    {n_artist:,}  ({100*n_artist/n_total:.1f}%)")
    print(f"    Have >=1 genre rating: {n_genre:,}  ({100*n_genre/n_total:.1f}%)")
    print(f"    Avg genre matches:     {sum(gc_list)/len(gc_list):.2f}")
    return n_album, n_artist, n_genre


# =====================================================================
# 7. Main
# =====================================================================


def main():
    # Load data
    user_ratings = parse_training(TRAIN_FILE)
    track_meta = parse_tracks(TRACK_FILE)
    album_meta = parse_albums(ALBUM_FILE)
    test_users = parse_test(TEST_FILE)

    # Coverage BEFORE enrichment
    print("\n" + "=" * 60)
    print("  BASELINE (Original Track Metadata)")
    print("=" * 60)
    base_alb, base_art, base_gen = coverage_analysis(
        user_ratings, track_meta, test_users, "(baseline)")

    # Enrich tracks using album metadata
    print("\n" + "=" * 60)
    print("  ENRICHING TRACKS WITH ALBUM METADATA")
    print("=" * 60)
    enriched_meta = enrich_tracks_with_albums(track_meta, album_meta)

    # Coverage AFTER enrichment
    enr_alb, enr_art, enr_gen = coverage_analysis(
        user_ratings, enriched_meta, test_users, "(enriched)")

    # Show improvement
    print(f"\n  Coverage Improvement:")
    print(f"    Artist matches: {base_art:,} -> {enr_art:,}  "
          f"(+{enr_art - base_art:,})")
    print(f"    Genre matches:  {base_gen:,} -> {enr_gen:,}  "
          f"(+{enr_gen - base_gen:,})")

    # Run all three strategies on enriched data
    print("\n" + "=" * 60)
    print("  GENERATING SUBMISSIONS (Enriched Data)")
    print("=" * 60)

    res_ev = run_strategy(
        user_ratings, enriched_meta, test_users,
        strategy_evidence_weighted,
        os.path.join(OUTPUT_DIR, "submission_v2_evidence.csv"),
    )

    res_wa = run_strategy(
        user_ratings, enriched_meta, test_users,
        strategy_weighted_avg,
        os.path.join(OUTPUT_DIR, "submission_v2_weightedavg.csv"),
    )

    res_mg = run_strategy(
        user_ratings, enriched_meta, test_users,
        strategy_max_genre,
        os.path.join(OUTPUT_DIR, "submission_v2_maxgenre.csv"),
    )

    print(f"\n  Submission files for Kaggle:")
    print(f"    submission_v2_evidence.csv      (Evidence-Weighted, enriched)")
    print(f"    submission_v2_weightedavg.csv   (Weighted Average, enriched)")
    print(f"    submission_v2_maxgenre.csv      (Max Genre, enriched)")

    print("\nDone. Upload submission_v2_evidence.csv to Kaggle for best results.")


if __name__ == "__main__":
    main()

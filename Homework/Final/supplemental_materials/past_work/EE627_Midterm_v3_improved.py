"""
EE627A Midterm v3 - Conservative Improvements over v2
Jude Eschete

v2 baseline (0.871): has_album*album + has_artist*artist + genre_count*genre_mean/10

Lessons from failed v3 attempts:
  - User normalization HURTS on real test set (0.809-0.868)
  - Ensemble min-max normalization destroys raw signal (0.850)
  - The v2 formula is already strong; only small additive changes are safe

This version tries ONLY conservative modifications:
  1. Fine-grained weight grid search (narrow range around v2's 1.0/1.0/0.1)
  2. Small interaction bonus when album+artist signals agree
  3. genre_max as alternative to genre_mean
  4. Confidence multiplier (small boost when more evidence exists)
  5. Additive raw ensemble (sum of raw scores, no normalization)
"""

import os
import math
import random
from collections import defaultdict
from itertools import product

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
    stats = {"artist_filled": 0, "genres_added": 0, "tracks_enriched": 0}
    for tid, (alb, art, genres) in tracks.items():
        new_art = art
        new_genres = list(genres)
        if alb is not None and alb in albums:
            album_art, album_genres = albums[alb]
            if art is None and album_art is not None:
                new_art = album_art
                stats["artist_filled"] += 1
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
    print(f"  Album Enrichment: {stats['artist_filled']:,} artists filled, "
          f"{stats['tracks_enriched']:,} tracks got new genres")
    return enriched


# =====================================================================
# 3. Feature Engineering
# =====================================================================

def compute_features(user_ratings, track_meta, uid, tid):
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
    else:
        g_max = g_min = 0
        g_mean = g_var = 0.0

    # Evidence count
    evidence_count = has_album + has_artist + (1 if n > 0 else 0)

    # Interaction (only when both signals present)
    album_artist_interact = (album_score * artist_score / 100.0
                             if (has_album and has_artist) else 0.0)

    return {
        "album_score": album_score,
        "has_album": has_album,
        "artist_score": artist_score,
        "has_artist": has_artist,
        "genre_count": n,
        "genre_max": g_max,
        "genre_min": g_min,
        "genre_mean": g_mean,
        "genre_var": g_var,
        "evidence_count": evidence_count,
        "album_artist_interact": album_artist_interact,
    }


# =====================================================================
# 4. Scoring Strategies (all conservative v2 variants)
# =====================================================================

def strategy_v2_evidence(feats_list):
    """Original v2: has_album*album + has_artist*artist + genre_count*genre_mean/10
    Kaggle: 0.871"""
    return [
        f["has_album"] * f["album_score"]
        + f["has_artist"] * f["artist_score"]
        + f["genre_count"] * f["genre_mean"] / 10.0
        for f in feats_list
    ]


def strategy_tunable(feats_list, wa=1.0, war=1.0, wg=0.1):
    """v2 with tunable weights."""
    return [
        wa * f["has_album"] * f["album_score"]
        + war * f["has_artist"] * f["artist_score"]
        + wg * f["genre_count"] * f["genre_mean"]
        for f in feats_list
    ]


def strategy_v2_plus_interaction(feats_list, wi=0.1):
    """v2 + small bonus when album AND artist signals agree."""
    return [
        f["has_album"] * f["album_score"]
        + f["has_artist"] * f["artist_score"]
        + f["genre_count"] * f["genre_mean"] / 10.0
        + wi * f["album_artist_interact"]
        for f in feats_list
    ]


def strategy_v2_genre_max(feats_list):
    """v2 but using genre_max instead of genre_mean."""
    return [
        f["has_album"] * f["album_score"]
        + f["has_artist"] * f["artist_score"]
        + (1 if f["genre_count"] > 0 else 0) * f["genre_max"] / 5.0
        for f in feats_list
    ]


def strategy_v2_genre_both(feats_list):
    """v2 with both genre_mean and genre_max."""
    return [
        f["has_album"] * f["album_score"]
        + f["has_artist"] * f["artist_score"]
        + f["genre_count"] * f["genre_mean"] / 10.0
        + (1 if f["genre_count"] > 0 else 0) * f["genre_max"] / 20.0
        for f in feats_list
    ]


def strategy_confidence(feats_list):
    """v2 * (1 + small evidence bonus). Rewards candidates with more data."""
    return [
        (f["has_album"] * f["album_score"]
         + f["has_artist"] * f["artist_score"]
         + f["genre_count"] * f["genre_mean"] / 10.0)
        * (1.0 + 0.05 * f["evidence_count"])
        for f in feats_list
    ]


def strategy_additive_ensemble(feats_list):
    """Sum raw scores from v2 + interaction + confidence (no normalization).
    All three share the same scale, so summing preserves signal."""
    s_v2 = strategy_v2_evidence(feats_list)
    s_int = strategy_v2_plus_interaction(feats_list, wi=0.15)
    s_conf = strategy_confidence(feats_list)
    return [a + b + c for a, b, c in zip(s_v2, s_int, s_conf)]


# =====================================================================
# 5. Ranking & Submission
# =====================================================================

def rank_top3(scores, track_ids):
    paired = sorted(zip(scores, track_ids), key=lambda x: x[0], reverse=True)
    top3 = set(tid for _, tid in paired[:3])
    return {tid: (1 if tid in top3 else 0) for tid in track_ids}


def write_submission(results, path):
    with open(path, "w") as f:
        f.write("TrackID,Predictor\n")
        for key, label in results:
            f.write(f"{key},{label}\n")


def run_strategy(user_ratings, track_meta, test_users,
                 strategy_fn, out_path, strategy_name=None):
    name = strategy_name or strategy_fn.__name__
    print(f"\n>>> Running {name} ...")
    results = []
    for uid, candidates in test_users:
        feats = [compute_features(user_ratings, track_meta, uid, tid)
                 for tid in candidates]
        scores = strategy_fn(feats)
        recs = rank_top3(scores, candidates)
        for tid in candidates:
            results.append((f"{uid}_{tid}", recs[tid]))
    write_submission(results, out_path)
    ones = sum(v for _, v in results)
    print(f"    {len(results):,} predictions ({ones:,} rec, {len(results)-ones:,} don't)")
    print(f"    Saved -> {out_path}")
    return results


# =====================================================================
# 6. Validation & Grid Search
# =====================================================================

def build_validation_set(user_ratings, track_meta, fraction=0.1, seed=42):
    """Hold out tracks from 10% of users. Mimics test scenario."""
    rng = random.Random(seed)
    track_ids = set(track_meta.keys())
    all_uids = list(user_ratings.keys())
    rng.shuffle(all_uids)
    n_val = int(len(all_uids) * fraction)

    val_set = []
    val_user_held_items = {}
    for uid in all_uids[:n_val]:
        user_tracks = [iid for iid in user_ratings[uid] if iid in track_ids]
        if len(user_tracks) < 6:
            continue
        sample = rng.sample(user_tracks, 6)
        sample_ratings = {tid: user_ratings[uid][tid] for tid in sample}
        val_set.append((uid, sample, sample_ratings))
        val_user_held_items[uid] = set(sample)

    train_reduced = {}
    for uid, ratings in user_ratings.items():
        if uid in val_user_held_items:
            held = val_user_held_items[uid]
            train_reduced[uid] = {k: v for k, v in ratings.items()
                                   if k not in held}
        else:
            train_reduced[uid] = ratings

    print(f"  Validation: {len(val_set):,} users, {len(val_set)*6:,} candidates")
    return train_reduced, val_set


def compute_auc(val_set, user_ratings, track_meta, strategy_fn):
    """AUC on validation: do we rank the true top-3 higher than bottom-3?"""
    correct = 0
    total = 0
    for uid, items, true_ratings in val_set:
        feats = [compute_features(user_ratings, track_meta, uid, tid)
                 for tid in items]
        scores = strategy_fn(feats)
        rated = sorted(zip([true_ratings[iid] for iid in items], items),
                       reverse=True)
        positives = set(tid for _, tid in rated[:3])
        negatives = set(tid for _, tid in rated[3:])
        score_map = dict(zip(items, scores))
        for p in positives:
            for n in negatives:
                if score_map[p] > score_map[n]:
                    correct += 1
                elif score_map[p] == score_map[n]:
                    correct += 0.5
                total += 1
    return correct / total if total > 0 else 0.0


def grid_search(val_set, user_ratings, track_meta):
    """Fine-grained grid search around the v2 sweet spot (1.0/1.0/0.1)."""
    print("\n  Fine-grained grid search around v2 weights ...")
    best_auc = 0
    best_params = None

    # Narrow ranges centered on v2 defaults
    wa_range = [0.8, 0.9, 1.0, 1.1, 1.2]
    war_range = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    wg_range = [0.06, 0.08, 0.10, 0.12, 0.14]

    total = len(wa_range) * len(war_range) * len(wg_range)
    tested = 0

    for wa, war, wg in product(wa_range, war_range, wg_range):
        def strat(feats, _wa=wa, _war=war, _wg=wg):
            return strategy_tunable(feats, _wa, _war, _wg)
        auc = compute_auc(val_set, user_ratings, track_meta, strat)
        tested += 1
        if auc > best_auc:
            best_auc = auc
            best_params = (wa, war, wg)
            print(f"    [{tested}/{total}] NEW BEST AUC={auc:.4f} "
                  f"album={wa} artist={war} genre={wg}")

    print(f"\n  Best: album={best_params[0]}, artist={best_params[1]}, "
          f"genre={best_params[2]}  AUC={best_auc:.4f}")
    return best_params, best_auc


def grid_search_interaction(val_set, user_ratings, track_meta):
    """Search interaction weight added to v2."""
    print("\n  Grid searching interaction weight ...")
    best_auc = 0
    best_wi = 0

    for wi in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        def strat(feats, _wi=wi):
            return strategy_v2_plus_interaction(feats, _wi)
        auc = compute_auc(val_set, user_ratings, track_meta, strat)
        if auc > best_auc:
            best_auc = auc
            best_wi = wi
            print(f"    wi={wi:.2f}  AUC={auc:.4f} *")
        else:
            print(f"    wi={wi:.2f}  AUC={auc:.4f}")

    print(f"\n  Best interaction weight: {best_wi}  AUC={best_auc:.4f}")
    return best_wi, best_auc


# =====================================================================
# 7. Main
# =====================================================================

def main():
    user_ratings = parse_training(TRAIN_FILE)
    track_meta = parse_tracks(TRACK_FILE)
    album_meta = parse_albums(ALBUM_FILE)
    test_users = parse_test(TEST_FILE)

    print("\n" + "=" * 60)
    print("  ENRICHING TRACKS")
    print("=" * 60)
    enriched_meta = enrich_tracks_with_albums(track_meta, album_meta)

    # ---- Validation ----
    print("\n" + "=" * 60)
    print("  VALIDATION & GRID SEARCH")
    print("=" * 60)
    train_reduced, val_set = build_validation_set(user_ratings, enriched_meta)

    # Baseline
    v2_auc = compute_auc(val_set, train_reduced, enriched_meta,
                         strategy_v2_evidence)
    print(f"\n  v2 baseline val AUC: {v2_auc:.4f}")

    # Evaluate all conservative variants
    print("\n  Strategy comparison:")
    for name, fn in [
        ("v2_evidence", strategy_v2_evidence),
        ("v2+interaction(0.1)", lambda f: strategy_v2_plus_interaction(f, 0.1)),
        ("v2+interaction(0.15)", lambda f: strategy_v2_plus_interaction(f, 0.15)),
        ("v2_genre_max", strategy_v2_genre_max),
        ("v2_genre_both", strategy_v2_genre_both),
        ("confidence", strategy_confidence),
        ("additive_ensemble", strategy_additive_ensemble),
    ]:
        auc = compute_auc(val_set, train_reduced, enriched_meta, fn)
        delta = auc - v2_auc
        marker = " <-- BEST" if delta > 0 else ""
        print(f"    {name:30s} AUC={auc:.4f} ({delta:+.4f}){marker}")

    # Grid searches
    best_params, best_tune_auc = grid_search(val_set, train_reduced, enriched_meta)
    best_wi, best_int_auc = grid_search_interaction(val_set, train_reduced, enriched_meta)

    # ---- Generate submissions ----
    print("\n" + "=" * 60)
    print("  GENERATING SUBMISSIONS")
    print("=" * 60)

    # 1: Tuned weights
    wa, war, wg = best_params
    def tuned(feats):
        return strategy_tunable(feats, wa, war, wg)
    run_strategy(user_ratings, enriched_meta, test_users, tuned,
                 os.path.join(OUTPUT_DIR, "submission_v3_tuned.csv"),
                 f"tuned(a={wa},ar={war},g={wg})")

    # 2: v2 + interaction
    def interact(feats):
        return strategy_v2_plus_interaction(feats, best_wi)
    run_strategy(user_ratings, enriched_meta, test_users, interact,
                 os.path.join(OUTPUT_DIR, "submission_v3_interact.csv"),
                 f"v2+interaction(wi={best_wi})")

    # 3: Genre both (mean + max)
    run_strategy(user_ratings, enriched_meta, test_users,
                 strategy_v2_genre_both,
                 os.path.join(OUTPUT_DIR, "submission_v3_genre_both.csv"),
                 "v2_genre_both")

    # 4: Confidence
    run_strategy(user_ratings, enriched_meta, test_users,
                 strategy_confidence,
                 os.path.join(OUTPUT_DIR, "submission_v3_confidence.csv"),
                 "confidence")

    # 5: Additive ensemble (raw sum, no normalization)
    run_strategy(user_ratings, enriched_meta, test_users,
                 strategy_additive_ensemble,
                 os.path.join(OUTPUT_DIR, "submission_v3_additive.csv"),
                 "additive_ensemble")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  v2 baseline val AUC:   {v2_auc:.4f}  (Kaggle: 0.871)")
    print(f"  tuned val AUC:         {best_tune_auc:.4f}")
    print(f"  interaction val AUC:   {best_int_auc:.4f}")
    print()
    print("  Files to upload:")
    print(f"    submission_v3_tuned.csv       (alb={wa}, art={war}, g={wg})")
    print(f"    submission_v3_interact.csv    (wi={best_wi})")
    print(f"    submission_v3_genre_both.csv")
    print(f"    submission_v3_confidence.csv")
    print(f"    submission_v3_additive.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()

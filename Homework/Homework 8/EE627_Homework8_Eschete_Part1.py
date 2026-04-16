"""
EE627A Homework 8 - PySpark Matrix Factorization for the Final Project
Jude Eschete

Starting point: midterm v3 conservative heuristic (Kaggle 0.871).
v2 formula that v3 builds on:
    has_album*album + has_artist*artist + genre_count*genre_mean/10

HW7 lessons carried forward:
  - Matrix factorization decomposes R ~ P * Q^T with K latent factors.
    HW7 implemented SGD MF by hand in MATLAB (alpha, beta, K, steps).
    HW8 replaces that with PySpark ALS (rank, maxIter, regParam) -- same
    three knobs, different optimizer.
  - Report MSE/RMSE on a held-out split BEFORE trusting a submission.
  - Fix the random seed so reruns are comparable.
  - Rank K trades bias vs variance; too small underfits, too large overfits.
  - Non-negative factors are a sensible prior when ratings are bounded.

Build-on path from v3:
  - Reuse v3's parsers, enrichment, validation framework, and grid search.
  - Add an ALS-trained latent score as a NEW feature per (user, track).
  - Add a new strategy (v2 + wm * mf_score) and tune wm via grid search
    on the same held-out validation set v3 already uses.
  - The hybrid carries the heuristic's 0.871 floor while giving the MF
    model a chance to boost candidates the metadata features miss.

Lecture materials used: Week 11 PySpark ALS sample code (rank/maxIter/
regParam/nonnegative/coldStartStrategy configuration).
"""

import os
import sys
import random
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Point Spark at the local Temurin JDK before importing pyspark.
# Spark 3.5 + JDK 17+ needs the --add-opens flags to avoid reflection errors.
# ---------------------------------------------------------------------------
JAVA_HOME_CANDIDATES = [
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot",
    r"C:\Program Files\Eclipse Adoptium\jdk-25.0.2.10-hotspot",
]
for _candidate in JAVA_HOME_CANDIDATES:
    if os.path.isdir(_candidate):
        os.environ["JAVA_HOME"] = _candidate
        os.environ["PATH"] = _candidate + r"\bin;" + os.environ["PATH"]
        break
os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
_JAVA_OPTS = " ".join([
    "-XX:+IgnoreUnrecognizedVMOptions",
    # JDK 23+ removed Subject.getSubject; Hadoop still calls it.
    # This flag makes it return null instead of throwing.
    "-Djava.security.manager=allow",
    # Big thread stack so deep RDD lineage serialization (ALS at
    # high maxIter) doesn't StackOverflow. In local[*] mode the
    # driver JVM is launched before spark.driver.extraJavaOptions
    # is read, so this flag must go through PYSPARK_SUBMIT_ARGS.
    "-Xss64m",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
])
os.environ["PYSPARK_SUBMIT_ARGS"] = f'--driver-java-options "{_JAVA_OPTS}" pyspark-shell'

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# =====================================================================
# Configuration
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data", "music-recommender-2026s")

TRAIN_FILE = os.path.join(DATA_DIR, "trainItem2.txt")
TEST_FILE = os.path.join(DATA_DIR, "testItem2.txt")
TRACK_FILE = os.path.join(DATA_DIR, "trackData2.txt")
ALBUM_FILE = os.path.join(DATA_DIR, "albumData2.txt")

# Pre-flattened CSV versions of train/test provided by the professor
# for direct use with spark.read.csv (per Week 11 PySpark sample code).
# Format: userID,itemID,rating  (rating is 0 for test rows).
SPARK_TRAIN_FILE = os.path.join(SCRIPT_DIR, "Data", "trainItem.data")
SPARK_TEST_FILE = os.path.join(SCRIPT_DIR, "Data", "testItem.data")

OUTPUT_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# ALS hyperparameters. Week 11 sample code defaults are rank=5, maxIter=5,
# regParam=0.01. We bump them here to give ALS more capacity: rank=20
# captures more latent structure, maxIter=20 lets convergence finish,
# regParam=0.05 reduces variance on cold-ish users.
# HW7 (MATLAB SGD MF) used the same three knobs: K (rank), steps (maxIter),
# beta (regParam).
# ---------------------------------------------------------------------------
ALS_RANK = 20
ALS_MAX_ITER = 20
ALS_REG_PARAM = 0.05
ALS_SEED = 42

# Path to Week 6 supplemental test-track hierarchy (per-test-row lineage:
# userID|trackID|albumID|artistID|genre1|genre2|...). Used as an override
# for test candidates in compute_features so the heuristic always has
# the most complete lineage available.
HIERARCHY_FILE = os.path.join(SCRIPT_DIR, "Data", "testTrack_hierarchy.txt")


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


def parse_hierarchy(path):
    """Parse Week 6 testTrack_hierarchy.txt.

    Format per line: userID|trackID|albumID|artistID|genre1|genre2|...
    "None" tokens mean missing. Returns a dict keyed by (userID, trackID)
    so compute_features can override track_meta lookups for test candidates.
    """
    print("[hier] Loading test-track hierarchy ...")
    hier = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            try:
                uid = int(parts[0])
                tid = int(parts[1])
            except ValueError:
                continue
            alb = None if parts[2] == "None" else int(parts[2])
            art = None if parts[3] == "None" else int(parts[3])
            genres = []
            for g in parts[4:]:
                g = g.strip()
                if g and g != "None":
                    try:
                        genres.append(int(g))
                    except ValueError:
                        pass
            hier[(uid, tid)] = (alb, art, genres)
    print(f"       {len(hier):,} test-track hierarchy entries")
    return hier


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

def compute_features(user_ratings, track_meta, uid, tid, mf_map=None,
                     mf_default=0.0, hierarchy=None):
    """Build feature vector for one (user, candidate-track) pair.

    If mf_map is provided, the ALS prediction for (uid, tid) is added
    to the feature dict as `mf_score` so strategies can use it as a
    natural signal alongside the heuristic features.

    If `hierarchy` is provided and contains (uid, tid), its lineage
    (album, artist, genres) overrides the track_meta lookup. This lets
    us use the per-test-row lineage from Week 6's testTrack_hierarchy.txt
    even when track_meta has missing or incomplete entries.
    """
    ratings = user_ratings.get(uid, {})
    if hierarchy is not None and (uid, tid) in hierarchy:
        alb_id, art_id, genre_ids = hierarchy[(uid, tid)]
    else:
        alb_id, art_id, genre_ids = track_meta.get(tid, (None, None, []))
    mf_score = mf_default if mf_map is None else mf_map.get((uid, tid), mf_default)

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
        "mf_score": mf_score,
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


def strategy_v2_mf_tiebreaker(feats_list, epsilon=0.001):
    """v2 heuristic with ALS as a pure tiebreaker.

    score = v2_score + epsilon * mf_score

    Preserves v2 ordering exactly; MF only breaks exact ties. Known
    floor: 0.871 (v2). Kaggle result: 0.875 (+0.004 over v2).
    """
    return [
        f["has_album"] * f["album_score"]
        + f["has_artist"] * f["artist_score"]
        + f["genre_count"] * f["genre_mean"] / 10.0
        + epsilon * f["mf_score"]
        for f in feats_list
    ]


def strategy_v2_mf_gated(feats_list):
    """Evidence-gated MF hybrid.

    The pure tiebreaker only activates on exact v2 ties. That's rare --
    mostly users with zero album/artist/genre overlap on all 6 candidates.
    This strategy lets MF have meaningful voice when v2 has LITTLE
    evidence, and recedes to a tiebreaker when v2 has STRONG evidence:

        evidence_count == 0 : weight = 5.0   (heuristic silent)
        evidence_count == 1 : weight = 0.2   (one signal, weak opinion)
        evidence_count == 2 : weight = 0.01  (two signals, small nudge)
        evidence_count >= 3 : weight = 0.001 (strong, tiebreak only)

    The 5.0 weight on silent candidates is enough to actually reorder
    them by ALS rather than ties. The smaller weights at higher evidence
    are bounded to stay below the minimum v2 delta, preserving the
    strong-signal ordering the way the tiebreaker does.
    """
    GATE = {0: 5.0, 1: 0.2, 2: 0.01}
    DEFAULT = 0.001
    scores = []
    for f in feats_list:
        base = (f["has_album"] * f["album_score"]
                + f["has_artist"] * f["artist_score"]
                + f["genre_count"] * f["genre_mean"] / 10.0)
        w = GATE.get(f["evidence_count"], DEFAULT)
        scores.append(base + w * f["mf_score"])
    return scores


# =====================================================================
# 4b. PySpark ALS trainer  (Week 11 sample code, adapted verbatim)
# =====================================================================
#
# Source: Lectures/Week11_04-13/supplemental_materials/
#         "PySpark Recommendation Code for the Final Project.py"
#
# The sample code reads the professor-provided pre-flattened CSV files
# (userID,itemID,rating) with spark.read.csv directly -- no custom parser.
# We follow that pattern and produce a dict (uid, iid) -> prediction that
# v3's compute_features() can then use as a new feature.
# =====================================================================

def run_pyspark_als(rank=ALS_RANK, max_iter=ALS_MAX_ITER,
                   reg_param=ALS_REG_PARAM, seed=ALS_SEED):
    """Train PySpark ALS and return (train_pred_map, test_pred_map, rmse, mse).

    Uses the pre-flattened CSV files from Week 11's supplemental data
    (trainItem.data / testItem.data). Mirrors the Week 11 sample code
    configuration: rank=5, maxIter=5, regParam=0.01, nonnegative=True,
    coldStartStrategy="drop".

    Returns:
        train_pred_map: dict (uid, iid) -> ALS prediction, for every row
                        in the training set. Used to enrich v3's feature
                        table so the heuristic strategies can reference
                        an MF signal on observed tracks too.
        test_pred_map:  dict (uid, iid) -> ALS prediction for every row
                        in testItem.data (the Kaggle submission set).
        rmse, mse:      holdout metrics (HW7 sanity check discipline).
    """
    print("\n" + "=" * 60)
    print("  PYSPARK ALS TRAINING  (Week 11 sample code pattern)")
    print("=" * 60)
    print(f"  rank={rank}  maxIter={max_iter}  regParam={reg_param}")
    print(f"  Reading {SPARK_TRAIN_FILE}")
    print(f"  Reading {SPARK_TEST_FILE}")

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("EE627_HW8_MF")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    # Note: we can't use setCheckpointDir on Windows without winutils.exe.
    # Instead we rely on the -Xss64m stack bump (128x default) configured
    # above to survive the deep RDD lineage serialization at maxIter=20.
    print(f"  Spark {spark.version} ready")

    # --- Step 1: Load training data (sample code pattern) ---
    training = spark.read.csv(SPARK_TRAIN_FILE, header=False)
    training = (training
                .withColumnRenamed("_c0", "userID")
                .withColumnRenamed("_c1", "itemID")
                .withColumnRenamed("_c2", "rating"))
    training = (training
                .withColumn("userID", training["userID"].cast(IntegerType()))
                .withColumn("itemID", training["itemID"].cast(IntegerType()))
                .withColumn("rating", training["rating"].cast(FloatType())))
    print(f"  Training rows: {training.count():,}")

    # --- HW7 lesson: hold out a slice, report MSE before trusting it ---
    train_df, holdout_df = training.randomSplit([0.9, 0.1], seed=seed)

    # --- Step 2: Configure ALS (sample code config) ---
    als = ALS(
        maxIter=max_iter,
        rank=rank,
        regParam=reg_param,
        userCol="userID",
        itemCol="itemID",
        ratingCol="rating",
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop",
        seed=seed,
    )

    # --- Step 3: Train on reduced, evaluate holdout, refit on full ---
    holdout_model = als.fit(train_df)
    holdout_preds = holdout_model.transform(holdout_df)
    evaluator = RegressionEvaluator(metricName="rmse",
                                    labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(holdout_preds)
    mse = rmse ** 2
    print(f"  Holdout RMSE = {rmse:.4f}   MSE = {mse:.4f}")

    print("  Refitting on full training set ...")
    model = als.fit(training)

    # --- Step 4: Load testing data (sample code pattern) ---
    testing = spark.read.csv(SPARK_TEST_FILE, header=False)
    testing = (testing
               .withColumnRenamed("_c0", "userID")
               .withColumnRenamed("_c1", "itemID")
               .withColumnRenamed("_c2", "rating"))
    testing = (testing
               .withColumn("userID", testing["userID"].cast(IntegerType()))
               .withColumn("itemID", testing["itemID"].cast(IntegerType()))
               .withColumn("rating", testing["rating"].cast(FloatType())))
    print(f"  Testing rows: {testing.count():,}")

    # --- Step 5: Predict on both training and testing ---
    # We need training predictions too so v3's strategies (which compute
    # features over candidate tracks the user may or may not have rated)
    # can look up the MF signal regardless.
    test_predictions = model.transform(testing)
    train_predictions = model.transform(training)

    print("  Collecting ALS predictions ...")
    global_mean = training.agg({"rating": "avg"}).first()[0]
    print(f"  Training mean rating = {global_mean:.3f}")

    # Keep the full ordered list of test (uid, iid) pairs so the pure-MF
    # submission is guaranteed to have exactly 120k rows even when ALS drops
    # cold-start predictions.
    test_pairs = [(row["userID"], row["itemID"]) for row in testing.collect()]

    test_pred_map = {}
    for row in test_predictions.collect():
        pred = row["prediction"]
        if pred is None:
            continue
        test_pred_map[(row["userID"], row["itemID"])] = float(pred)

    train_pred_map = {}
    for row in train_predictions.collect():
        pred = row["prediction"]
        if pred is None:
            continue
        train_pred_map[(row["userID"], row["itemID"])] = float(pred)

    print(f"  Test pairs: {len(test_pairs):,}   "
          f"ALS predictions: {len(test_pred_map):,}   "
          f"cold-start filled with global mean: "
          f"{len(test_pairs) - len(test_pred_map):,}")
    print(f"  Train predictions: {len(train_pred_map):,}")

    # --- Step 6: Save the pure-MF submission per sample code convention ---
    # Iterate the full test_pairs list so the submission has 120k rows even
    # if ALS dropped some predictions (cold-start users/items).
    pure_mf_path = os.path.join(OUTPUT_DIR, "submission_hw8_pyspark_mf.csv")
    print(f"  Writing pure-MF submission to {pure_mf_path}")
    from collections import defaultdict as _dd
    by_user = _dd(list)
    for uid, iid in test_pairs:
        pred = test_pred_map.get((uid, iid), global_mean)
        by_user[uid].append((iid, pred))
    pure_results = []
    for uid, items in by_user.items():
        ranked_items = sorted(items, key=lambda x: x[1], reverse=True)
        top3 = set(tid for tid, _ in ranked_items[:3])
        for iid, _ in items:
            pure_results.append((f"{uid}_{iid}", 1 if iid in top3 else 0))
    write_submission(pure_results, pure_mf_path)
    print(f"    {len(pure_results):,} rows written "
          f"(expected 120,000)")

    spark.stop()
    return train_pred_map, test_pred_map, rmse, mse, global_mean


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
                 strategy_fn, out_path, strategy_name=None, mf_map=None,
                 hierarchy=None):
    name = strategy_name or strategy_fn.__name__
    print(f"\n>>> Running {name} ...")
    results = []
    for uid, candidates in test_users:
        feats = [compute_features(user_ratings, track_meta, uid, tid,
                                  mf_map=mf_map, hierarchy=hierarchy)
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


def compute_auc(val_set, user_ratings, track_meta, strategy_fn, mf_map=None,
                hierarchy=None):
    """AUC on validation: do we rank the true top-3 higher than bottom-3?"""
    correct = 0
    total = 0
    for uid, items, true_ratings in val_set:
        feats = [compute_features(user_ratings, track_meta, uid, tid,
                                  mf_map=mf_map, hierarchy=hierarchy)
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


# =====================================================================
# 7. Main
# =====================================================================

class _Tee:
    """Duplicate stdout writes to a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
)


def main():
    log_fh = open(RESULTS_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    print("=" * 60)
    print("  EE627A Homework 8 - PySpark Matrix Factorization")
    print("  Jude Eschete")
    print("=" * 60)

    user_ratings = parse_training(TRAIN_FILE)
    track_meta = parse_tracks(TRACK_FILE)
    album_meta = parse_albums(ALBUM_FILE)
    test_users = parse_test(TEST_FILE)
    hierarchy = parse_hierarchy(HIERARCHY_FILE)

    print("\n" + "=" * 60)
    print("  ENRICHING TRACKS")
    print("=" * 60)
    enriched_meta = enrich_tracks_with_albums(track_meta, album_meta)

    # ---- Validation ----
    print("\n" + "=" * 60)
    print("  VALIDATION & GRID SEARCH")
    print("=" * 60)
    train_reduced, val_set = build_validation_set(user_ratings, enriched_meta)

    # Baseline (+ same baseline with hierarchy lineage for test-time features)
    v2_auc = compute_auc(val_set, train_reduced, enriched_meta,
                         strategy_v2_evidence)
    print(f"\n  v2 baseline val AUC:              {v2_auc:.4f}")
    v2_hier_auc = compute_auc(val_set, train_reduced, enriched_meta,
                              strategy_v2_evidence, hierarchy=hierarchy)
    print(f"  v2 baseline val AUC (+hierarchy): {v2_hier_auc:.4f}")

    # ---- PySpark ALS training (HW8 Part 1 core) ----
    # Run ALS on the Week-11 pre-flattened CSVs using the sample code
    # pattern. Also writes submission_hw8_pyspark_mf.csv (pure MF).
    train_pred_map, test_pred_map, holdout_rmse, holdout_mse, _mf_mean = \
        run_pyspark_als()

    # Sanity-check both hybrids against the v2 floor locally.
    tiebreaker_val_auc = compute_auc(
        val_set, train_reduced, enriched_meta,
        strategy_v2_mf_tiebreaker,
        mf_map=train_pred_map, hierarchy=hierarchy,
    )
    gated_val_auc = compute_auc(
        val_set, train_reduced, enriched_meta,
        strategy_v2_mf_gated,
        mf_map=train_pred_map, hierarchy=hierarchy,
    )
    print(f"\n  v2+MF tiebreaker val AUC (+hier): {tiebreaker_val_auc:.4f}")
    print(f"  v2+MF evidence-gated val AUC (+hier): {gated_val_auc:.4f}")

    # ---- Generate hybrid submissions ----
    print("\n" + "=" * 60)
    print("  GENERATING HYBRID SUBMISSIONS")
    print("=" * 60)
    run_strategy(
        user_ratings, enriched_meta, test_users,
        strategy_v2_mf_tiebreaker,
        os.path.join(OUTPUT_DIR, "submission_hw8_hybrid.csv"),
        strategy_name="v2 + MF tiebreaker (+hierarchy)",
        mf_map=test_pred_map, hierarchy=hierarchy,
    )
    run_strategy(
        user_ratings, enriched_meta, test_users,
        strategy_v2_mf_gated,
        os.path.join(OUTPUT_DIR, "submission_hw8_hybrid_gated.csv"),
        strategy_name="v2 + MF evidence-gated (+hierarchy)",
        mf_map=test_pred_map, hierarchy=hierarchy,
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  v2 baseline val AUC:                  {v2_auc:.4f}")
    print(f"  v2 baseline +hier val AUC:            {v2_hier_auc:.4f}")
    print(f"  v2+MF tiebreaker (+hier) val AUC:     {tiebreaker_val_auc:.4f}")
    print(f"  v2+MF evidence-gated (+hier) val AUC: {gated_val_auc:.4f}")
    print(f"  ALS holdout RMSE:                     {holdout_rmse:.4f}   "
          f"MSE: {holdout_mse:.4f}  "
          f"(rank={ALS_RANK}, maxIter={ALS_MAX_ITER}, regParam={ALS_REG_PARAM})")
    print()
    print("  Files to upload for HW8 Part 1:")
    print(f"    submission_hw8_pyspark_mf.csv       <-- pure PySpark MF")
    print(f"    submission_hw8_hybrid.csv           <-- v2 + MF tiebreaker")
    print(f"    submission_hw8_hybrid_gated.csv     <-- v2 + MF evidence-gated")
    print(f"\n  Full run log written to: {RESULTS_PATH}")
    print("\nDone.")
    log_fh.close()


def _log_crash(exc):
    """Write a clean, timestamped crash report next to Results/."""
    err_path = os.path.join(
        RESULTS_DIR,
        f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    with open(err_path, "w", encoding="utf-8") as fh:
        fh.write("=" * 60 + "\n")
        fh.write(f"  CRASH: {type(exc).__name__}\n")
        fh.write(f"  When:  {datetime.now().isoformat()}\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"{type(exc).__name__}: {exc}\n\n")
        fh.write("Traceback:\n")
        fh.write("-" * 60 + "\n")
        traceback.print_exc(file=fh)
        fh.write("\n" + "-" * 60 + "\n")
    # Also append a one-line summary to a rolling error index.
    index_path = os.path.join(RESULTS_DIR, "errors_index.log")
    with open(index_path, "a", encoding="utf-8") as fh:
        fh.write(
            f"{datetime.now().isoformat()}  "
            f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}  "
            f"-> {os.path.basename(err_path)}\n"
        )
    return err_path


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err_path = _log_crash(exc)
        print(f"\n[!] Crashed: {type(exc).__name__}: {exc}", file=sys.__stderr__)
        print(f"[!] Full traceback written to: {err_path}", file=sys.__stderr__)
        print(f"[!] Rolling index: {os.path.join(RESULTS_DIR, 'errors_index.log')}",
              file=sys.__stderr__)
        sys.exit(1)

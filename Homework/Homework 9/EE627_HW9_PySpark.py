"""
EE627A Homework 9 - PySpark ML Classifiers for Music Recommendation
Jude Eschete

Approach
--------
Feature matrix: 28 hand-engineered features per (user, track) pair,
ported from the midterm v5 model (EE627_Midterm_v5_full.py).
The MF embedding feature is omitted (requires GPU training); all other
29 base features are retained minus that one slot = 28 features.

All features are purely numerical, so no StringIndexer / one-hot
encoding is needed -- we go straight to VectorAssembler then classifiers.

Ground truth: test2_new.txt (1000 users x 6 tracks = 6000 labeled rows).
Prediction target: full testItem2.txt test set (Kaggle submission).

Four PySpark classifiers (Week 12 slides):
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. Gradient-Boosted Tree Classifier  (+ cross-validated tuned variant)

Each generates a Kaggle-format submission CSV.
All output is tee'd to results_hw9.txt for post-run analysis.
"""

import os
import sys
import math
import csv
import time
import statistics
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MIDTERM_DIR = os.path.join(SCRIPT_DIR, "..", "Midterm")
DATA_DIR    = os.path.join(MIDTERM_DIR, "Data", "music-recommender-2026s")

TRAIN_FILE  = os.path.join(DATA_DIR, "trainItem2.txt")
TEST_FILE   = os.path.join(DATA_DIR, "testItem2.txt")
TRACK_FILE  = os.path.join(DATA_DIR, "trackData2.txt")
ALBUM_FILE  = os.path.join(DATA_DIR, "albumData2.txt")
LABEL_FILE  = os.path.join(SCRIPT_DIR, "test2_new.txt")

FEATURE_CSV  = os.path.join(SCRIPT_DIR, "features_labeled.csv")
TEST_CSV     = os.path.join(SCRIPT_DIR, "features_test.csv")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results_hw9.txt")

FEATURE_NAMES = [
    "album_score", "has_album",
    "artist_score", "has_artist",
    "genre_count", "genre_max", "genre_min", "genre_mean", "genre_var",
    # "evidence_count",       # 0.000 importance on DT, GBT, GBT-CV -- dropped
    "album_artist_int",
    # "genre_artist_int",     # 0.000 on DT, GBT-CV -- dropped
    # "album_genre_int",      # 0.000 on DT, GBT, GBT-CV -- dropped
    "user_mean", "user_std", "user_n_log",
    "album_above_mean", "artist_above_mean", "genre_above_mean",
    "alb_global_avg", "art_global_avg", "genre_global_avg",
    "alb_nrat_log", "art_nrat_log", "genre_nrat_log",
    "cf_score", "cf_count_log",
    "v2_score",
]


# ---------------------------------------------------------------------------
# Logging: tee all print() output to file
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Data Parsing
# ---------------------------------------------------------------------------

def parse_training(path):
    t0 = time.time()
    print("Loading training data ...")
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

    total_ratings = sum(len(r) for r in users.values())
    all_counts    = [len(r) for r in users.values()]
    all_ratings   = [r for u in users.values() for r in u.values()]

    print(f"  {len(users):,} users  |  {total_ratings:,} ratings  [{elapsed(t0)}]")
    print(f"  Ratings per user : min={min(all_counts)}  "
          f"median={sorted(all_counts)[len(all_counts)//2]}  "
          f"max={max(all_counts)}")
    print(f"  Rating values    : min={min(all_ratings)}  "
          f"mean={sum(all_ratings)/len(all_ratings):.2f}  "
          f"max={max(all_ratings)}")
    return users


def parse_tracks(path):
    t0 = time.time()
    print("Loading track metadata ...")
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

    no_album  = sum(1 for a, _, _ in tracks.values() if a is None)
    no_artist = sum(1 for _, a, _ in tracks.values() if a is None)
    genre_lens = [len(g) for _, _, g in tracks.values()]
    print(f"  {len(tracks):,} tracks  [{elapsed(t0)}]")
    print(f"  Missing album : {no_album:,}  |  missing artist: {no_artist:,}")
    print(f"  Genres per track: min={min(genre_lens)}  "
          f"mean={sum(genre_lens)/len(genre_lens):.1f}  "
          f"max={max(genre_lens)}")
    return tracks


def parse_albums(path):
    t0 = time.time()
    print("Loading album metadata ...")
    albums = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 1:
                continue
            alb_id = int(parts[0])
            art    = None if len(parts) < 2 or parts[1] == "None" \
                     else int(parts[1])
            genres = [int(g) for g in parts[2:]
                      if g.strip() and g != "None"]
            albums[alb_id] = (art, genres)
    print(f"  {len(albums):,} albums  [{elapsed(t0)}]")
    return albums


def parse_test(path):
    t0 = time.time()
    print("Loading test candidates ...")
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

    n_cands = [len(t) for _, t in test]
    print(f"  {len(test):,} test users  |  "
          f"{sum(n_cands):,} total candidates  [{elapsed(t0)}]")
    print(f"  Candidates per user: "
          f"min={min(n_cands)}  max={max(n_cands)}  "
          f"(expected 6 each)")
    return test


def parse_labels(path):
    t0 = time.time()
    print("Loading ground-truth labels ...")
    labels = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                labels[(int(parts[0]), int(parts[1]))] = int(parts[2])

    n_pos = sum(v for v in labels.values())
    n_neg = len(labels) - n_pos
    uids  = set(u for u, _ in labels)
    print(f"  {len(labels):,} rows  |  {len(uids):,} users  [{elapsed(t0)}]")
    print(f"  Label balance: {n_pos:,} positives ({100*n_pos/len(labels):.1f}%)  "
          f"|  {n_neg:,} negatives ({100*n_neg/len(labels):.1f}%)")
    return labels


# ---------------------------------------------------------------------------
# 2. Enrichment
# ---------------------------------------------------------------------------

def enrich_tracks(tracks, albums):
    t0 = time.time()
    print("Enriching tracks with album metadata ...")
    enriched       = {}
    artist_filled  = 0
    genres_added   = 0
    for tid, (alb, art, genres) in tracks.items():
        new_art    = art
        new_genres = list(genres)
        if alb is not None and alb in albums:
            album_art, album_genres = albums[alb]
            if art is None and album_art is not None:
                new_art = album_art
                artist_filled += 1
            seen = set(genres)
            for g in album_genres:
                if g not in seen:
                    new_genres.append(g)
                    seen.add(g)
                    genres_added += 1
        enriched[tid] = (alb, new_art, new_genres)

    no_artist_after = sum(1 for _, a, _ in enriched.values() if a is None)
    print(f"  Artists filled via album : {artist_filled:,}")
    print(f"  Genres added via album   : {genres_added:,}")
    print(f"  Tracks still missing artist after enrichment: "
          f"{no_artist_after:,}  [{elapsed(t0)}]")
    return enriched


# ---------------------------------------------------------------------------
# 3. Auxiliary structures
# ---------------------------------------------------------------------------

def build_global_item_stats(user_ratings):
    t0       = time.time()
    item_sum = defaultdict(float)
    item_cnt = defaultdict(int)
    for ratings in user_ratings.values():
        for iid, r in ratings.items():
            item_sum[iid] += r
            item_cnt[iid] += 1
    stats = {iid: (item_sum[iid] / item_cnt[iid], item_cnt[iid])
             for iid in item_sum}

    counts = [c for _, c in stats.values()]
    avgs   = [a for a, _ in stats.values()]
    print(f"  Global item stats: {len(stats):,} items  [{elapsed(t0)}]")
    print(f"  Raters per item  : min={min(counts)}  "
          f"median={sorted(counts)[len(counts)//2]}  "
          f"max={max(counts)}")
    print(f"  Global avg rating: min={min(avgs):.1f}  "
          f"mean={sum(avgs)/len(avgs):.2f}  "
          f"max={max(avgs):.1f}")
    return stats


def build_track_mappings(track_meta):
    album_to_tracks  = defaultdict(list)
    artist_to_tracks = defaultdict(list)
    for tid, (alb, art, genres) in track_meta.items():
        if alb is not None:
            album_to_tracks[alb].append(tid)
        if art is not None:
            artist_to_tracks[art].append(tid)
    alb_sizes = [len(v) for v in album_to_tracks.values()]
    art_sizes = [len(v) for v in artist_to_tracks.values()]
    print(f"  Album->tracks  : {len(album_to_tracks):,} albums  "
          f"(avg {sum(alb_sizes)/len(alb_sizes):.1f} tracks/album)")
    print(f"  Artist->tracks : {len(artist_to_tracks):,} artists  "
          f"(avg {sum(art_sizes)/len(art_sizes):.1f} tracks/artist)")
    return dict(album_to_tracks), dict(artist_to_tracks)


# ---------------------------------------------------------------------------
# 4. Feature computation (28 features -- v5 base minus MF slot)
# ---------------------------------------------------------------------------

def compute_user_profile(ratings):
    vals = list(ratings.values())
    n    = len(vals)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(vals) / n
    std  = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
    return mean, std, n


def compute_features(uid, tid, ratings, track_meta, global_stats,
                     album_to_tracks, artist_to_tracks):
    """28 numerical features per (user, track) pair.

    Matches v5 base features 0-26 and 28 (v2_score).
    Feature 27 (MF dot product) is omitted -- requires GPU MF training.
    """
    alb_id, art_id, genre_ids = track_meta.get(tid, (None, None, []))
    u_mean, u_std, u_n        = compute_user_profile(ratings)

    # Album
    if alb_id is not None and alb_id in ratings:
        album_score, has_album = float(ratings[alb_id]), 1.0
    else:
        album_score, has_album = 0.0, 0.0

    # Artist
    if art_id is not None and art_id in ratings:
        artist_score, has_artist = float(ratings[art_id]), 1.0
    else:
        artist_score, has_artist = 0.0, 0.0

    # Genre stats
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

    # Interaction terms
    aa_int = album_score * artist_score / 100.0 \
             if (has_album and has_artist) else 0.0
    ga_int = g_mean * artist_score / 100.0 \
             if (has_artist and gc > 0) else 0.0
    ag_int = album_score * g_mean / 100.0 \
             if (has_album and gc > 0) else 0.0

    # User-relative scores
    album_above  = (album_score  - u_mean) if has_album  else 0.0
    artist_above = (artist_score - u_mean) if has_artist else 0.0
    genre_above  = (g_mean       - u_mean) if gc > 0     else 0.0

    # Global popularity stats (log-scaled to suppress heavy-tail outliers)
    alb_gavg, alb_nrat = global_stats.get(alb_id, (0.0, 0)) \
                         if alb_id else (0.0, 0)
    art_gavg, art_nrat = global_stats.get(art_id, (0.0, 0)) \
                         if art_id else (0.0, 0)
    g_gavg_sum = g_nrat_sum = g_items = 0.0
    for gid in genre_ids:
        if gid in global_stats:
            gavg, nrat  = global_stats[gid]
            g_gavg_sum += gavg
            g_nrat_sum += nrat
            g_items    += 1
    avg_genre_gavg = g_gavg_sum / g_items if g_items > 0 else 0.0
    avg_genre_nrat = g_nrat_sum / g_items if g_items > 0 else 0.0

    # CF sibling features
    sibling_ratings = []
    if alb_id is not None:
        for sib in album_to_tracks.get(alb_id, []):
            if sib != tid and sib in ratings:
                sibling_ratings.append(ratings[sib])
    if art_id is not None:
        album_sibs = set(album_to_tracks.get(alb_id, [])) if alb_id else set()
        for sib in artist_to_tracks.get(art_id, []):
            if sib != tid and sib not in album_sibs and sib in ratings:
                sibling_ratings.append(ratings[sib])
    cf_score = sum(sibling_ratings) / len(sibling_ratings) \
               if sibling_ratings else 0.0
    cf_count = len(sibling_ratings)

    # v2 heuristic score (most informative single signal, Kaggle-validated)
    v2_score = (has_album  * album_score
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
        v2_score,                                        # 27
    ]


# ---------------------------------------------------------------------------
# 5. Build feature CSVs + inline diagnostics
# ---------------------------------------------------------------------------

def build_labeled_csv(user_ratings, track_meta, global_stats,
                      album_to_tracks, artist_to_tracks, labels, out_path):
    t0 = time.time()
    print(f"Building labeled feature CSV ...")
    header = ["user_id", "track_id"] + FEATURE_NAMES + ["label"]

    all_feats  = [[] for _ in FEATURE_NAMES]  # per-feature value lists
    rows_written = 0
    missing_user = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (uid, tid), label in labels.items():
            ratings = user_ratings.get(uid, {})
            if not ratings:
                missing_user += 1
            feats = compute_features(uid, tid, ratings, track_meta,
                                     global_stats, album_to_tracks,
                                     artist_to_tracks)
            writer.writerow([uid, tid] + feats + [label])
            for i, v in enumerate(feats):
                all_feats[i].append(v)
            rows_written += 1

    print(f"  {rows_written:,} rows written  "
          f"|  {missing_user:,} users with no training history  [{elapsed(t0)}]")

    # Feature distribution diagnostics
    print("\n  Feature distributions (labeled set):")
    print(f"  {'Feature':<25} {'mean':>8} {'std':>8} "
          f"{'min':>8} {'max':>8} {'zero%':>7}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for i, name in enumerate(FEATURE_NAMES):
        vals  = all_feats[i]
        mean  = sum(vals) / len(vals)
        std   = statistics.pstdev(vals)
        vmin  = min(vals)
        vmax  = max(vals)
        zpct  = 100.0 * sum(1 for v in vals if v == 0.0) / len(vals)
        flag  = "  <- mostly zero, consider dropping" if zpct > 90 else ""
        print(f"  {name:<25} {mean:>8.2f} {std:>8.2f} "
              f"{vmin:>8.2f} {vmax:>8.2f} {zpct:>6.1f}%{flag}")


def build_test_csv(user_ratings, track_meta, global_stats,
                   album_to_tracks, artist_to_tracks, test_users, out_path):
    t0 = time.time()
    print(f"Building test feature CSV ...")
    header       = ["user_id", "track_id"] + FEATURE_NAMES
    rows_written = 0
    missing_user = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for uid, candidates in test_users:
            ratings = user_ratings.get(uid, {})
            if not ratings:
                missing_user += 1
            for tid in candidates:
                feats = compute_features(uid, tid, ratings, track_meta,
                                         global_stats, album_to_tracks,
                                         artist_to_tracks)
                writer.writerow([uid, tid] + feats)
                rows_written += 1
    print(f"  {rows_written:,} test rows written  "
          f"|  {missing_user:,} users with no training history  [{elapsed(t0)}]")


# ---------------------------------------------------------------------------
# 6. PySpark ML pipeline
# ---------------------------------------------------------------------------

def log_classifier_metrics(name, preds_val, evaluator, train_df):
    """Print AUC, precision, recall, confusion matrix on validation set."""
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    auc = evaluator.evaluate(preds_val,
                             {evaluator.metricName: "areaUnderROC"})
    mc  = MulticlassClassificationEvaluator(labelCol="label",
                                            predictionCol="prediction")
    acc  = mc.evaluate(preds_val, {mc.metricName: "accuracy"})
    prec = mc.evaluate(preds_val, {mc.metricName: "weightedPrecision"})
    rec  = mc.evaluate(preds_val, {mc.metricName: "weightedRecall"})
    f1   = mc.evaluate(preds_val, {mc.metricName: "f1"})

    # Confusion matrix counts
    pdf = preds_val.select("label", "prediction").toPandas()
    tp = int(((pdf["label"] == 1) & (pdf["prediction"] == 1)).sum())
    tn = int(((pdf["label"] == 0) & (pdf["prediction"] == 0)).sum())
    fp = int(((pdf["label"] == 0) & (pdf["prediction"] == 1)).sum())
    fn = int(((pdf["label"] == 1) & (pdf["prediction"] == 0)).sum())

    print(f"\n  [{name}] Validation metrics:")
    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    Confusion matrix (val set):")
    print(f"      Predicted ->     0       1")
    print(f"      Actual 0    {tn:>6,}  {fp:>6,}   (spec={tn/(tn+fp):.3f})")
    print(f"      Actual 1    {fn:>6,}  {tp:>6,}   (sens={tp/(tp+fn):.3f})")
    return auc


def log_feature_importances(feat_importances, label, feature_names=None):
    """Print full ranked feature importance table."""
    if feature_names is None:
        feature_names = FEATURE_NAMES
    arr    = feat_importances.toArray()
    ranked = sorted(enumerate(arr), key=lambda x: x[1], reverse=True)
    print(f"\n  [{label}] Feature importances (all {len(arr)} features):")
    print(f"  {'Rank':<5} {'Feature':<25} {'Importance':>12}  Bar")
    print(f"  {'-'*5} {'-'*25} {'-'*12}  {'-'*20}")
    max_imp = ranked[0][1] if ranked else 1.0
    for rank, (idx, imp) in enumerate(ranked, 1):
        bar  = "#" * int(20 * imp / max_imp)
        flag = "  <- low signal" if imp < 0.005 else ""
        print(f"  {rank:<5} {feature_names[idx]:<25} {imp:>12.6f}  {bar}{flag}")


def log_cv_fold_scores(cv_model):
    """Print per-fold AUC from cross-validation."""
    avg_metrics = cv_model.avgMetrics
    print(f"\n  [GBT-CV] Per-parameter-combination average AUC "
          f"({len(avg_metrics)} combinations):")
    for i, score in enumerate(avg_metrics):
        print(f"    Combo {i+1:2d}: {score:.4f}")
    best_idx = avg_metrics.index(max(avg_metrics))
    print(f"  Best combination index: {best_idx+1}  "
          f"(AUC = {avg_metrics[best_idx]:.4f})")


def collect_probs(pred_df):
    """Collect (uid, tid, prob1) tuples from a Spark prediction DataFrame.
    Returns dict {(uid, tid): prob1}."""
    rows = pred_df.select("user_id", "track_id", "probability").collect()
    return {(int(r["user_id"]), int(r["track_id"])): float(r["probability"][1])
            for r in rows}


def compute_oof_probs(df_full, builder, n_folds=5, seed=2018, name="model"):
    """5-fold out-of-fold predictions for the labeled set.

    Splits df_full into n_folds. For each fold, trains a fresh classifier on
    the OTHER folds and predicts on the held-out fold. The result is a dict
    {(uid, tid): prob1} where every labeled row's prediction comes from a
    model that did NOT see it during training -- the correct way to feed a
    stacked meta-learner.

    `builder` is a zero-arg callable that returns a fresh, untrained
    classifier (so we don't reuse fitted state across folds).
    """
    print(f"\n  [{name}] Computing {n_folds}-fold OOF predictions ...")
    t0 = time.time()

    # Spark's randomSplit with equal weights gives us reproducible folds
    weights = [1.0 / n_folds] * n_folds
    folds   = df_full.randomSplit(weights, seed=seed)

    oof = {}
    for k in range(n_folds):
        holdout = folds[k]
        train_parts = [folds[j] for j in range(n_folds) if j != k]
        train_df    = train_parts[0]
        for tp in train_parts[1:]:
            train_df = train_df.union(tp)

        clf       = builder()
        model     = clf.fit(train_df)
        preds     = model.transform(holdout)
        fold_oof  = collect_probs(preds)
        oof.update(fold_oof)
        print(f"    fold {k+1}/{n_folds}: trained on {train_df.count():,}, "
              f"predicted {len(fold_oof):,} held-out rows")

    print(f"  [{name}] OOF complete: {len(oof):,} rows  [{elapsed(t0)}]")
    return oof


def write_blend_submission(blend_dict, name, script_dir, label):
    """Take a {(uid, tid): score} dict and write a top-3-per-user CSV.
    `label` is a short string used in the printed summary."""
    by_user = defaultdict(list)
    for (uid, tid), s in blend_dict.items():
        by_user[uid].append((tid, s))

    sub_path = os.path.join(script_dir, f"submission_hw9_{name}.csv")
    n_ones, n_total = 0, 0
    with open(sub_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TrackID", "Predictor"])
        for uid, items in by_user.items():
            top3 = set(t for t, _ in sorted(items, key=lambda x: x[1],
                                            reverse=True)[:3])
            for tid, _ in items:
                pred = 1 if tid in top3 else 0
                writer.writerow([f"{uid}_{tid}", pred])
                n_ones  += pred
                n_total += 1
    print(f"  {name:<22} ({label})  "
          f"{n_ones:,} rec / {n_total-n_ones:,} not rec  "
          f"-> {os.path.basename(sub_path)}")
    return sub_path


def run_pyspark(feature_csv, test_csv, script_dir):
    t_spark = time.time()

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import (LogisticRegression,
                                            DecisionTreeClassifier,
                                            RandomForestClassifier,
                                            GBTClassifier)
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

    # ---- SparkSession ----
    # - Driver memory bumped to 12g: large RF/GBT task results can exceed 6g,
    #   producing StreamCorruptedException ("invalid type code: 00") during
    #   task-result deserialization on Windows.
    # - Kryo serializer: faster + more robust than Java serialization for
    #   ML model objects.
    # - Thread stack 32M: prevents StackOverflowError when serializing deep
    #   tree models (default 1M overflows at maxDepth >= ~15).
    spark = (SparkSession.builder
             .appName("EE627-HW9")
             .config("spark.driver.memory",            "12g")
             .config("spark.driver.maxResultSize",     "4g")
             .config("spark.serializer",
                     "org.apache.spark.serializer.KryoSerializer")
             .config("spark.kryoserializer.buffer.max","512m")
             .config("spark.driver.extraJavaOptions",  "-Xss32m")
             .config("spark.executor.extraJavaOptions","-Xss32m")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    section("LOADING & PREPARING DATA IN SPARK")

    labeled = spark.read.csv(feature_csv, header=True, inferSchema=True)
    labeled = labeled.withColumn("label", col("label").cast("double"))
    n_total = labeled.count()
    n_pos   = labeled.filter(col("label") == 1).count()
    n_neg   = n_total - n_pos
    print(f"  Labeled rows : {n_total:,}  "
          f"({n_pos:,} pos = {100*n_pos/n_total:.1f}%  "
          f"|  {n_neg:,} neg = {100*n_neg/n_total:.1f}%)")

    assembler   = VectorAssembler(inputCols=FEATURE_NAMES, outputCol="features")
    prep_model  = Pipeline(stages=[assembler]).fit(labeled)
    df_full     = prep_model.transform(labeled) \
                            .select("label", "features", "user_id", "track_id")

    # 70/30 split for hyperparameter selection / validation reporting only
    train, val = df_full.randomSplit([0.7, 0.3], seed=2018)
    n_train    = train.count()
    n_val      = val.count()
    n_val_pos  = val.filter(col("label") == 1).count()
    print(f"  Full labeled : {n_total:,}  (used to refit final models)")
    print(f"  Train (70%)  : {n_train:,} rows  (used for CV & tuning)")
    print(f"  Validation   : {n_val:,} rows  "
          f"({n_val_pos:,} pos = {100*n_val_pos/n_val:.1f}%)")

    # Load full Kaggle test set
    test_df_raw = spark.read.csv(test_csv, header=True, inferSchema=True)
    test_df     = prep_model.transform(test_df_raw) \
                            .select("user_id", "track_id", "features")
    print(f"  Kaggle test  : {test_df.count():,} rows")

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    results   = {}   # name -> (auc, prob_dict)  prob_dict = {(uid,tid): p}

    # Best hyperparameters captured during each classifier's CV.  Used at
    # the end of the script to rebuild fresh classifiers for the OOF
    # probability matrix (so the meta-learner in the Final project gets
    # un-leaked predictions for the labeled rows).
    best_params = {}

    # ------------------------------------------------------------------
    # 1. Logistic Regression  (elastic net + light tuning)
    # ------------------------------------------------------------------
    section("[1/7] LOGISTIC REGRESSION (elastic-net CV)")
    t0 = time.time()
    lr_tune = LogisticRegression(featuresCol="features", labelCol="label",
                                 maxIter=50)
    lr_grid = (ParamGridBuilder()
               .addGrid(lr_tune.regParam,        [0.0, 0.01, 0.1])
               .addGrid(lr_tune.elasticNetParam, [0.0, 0.5, 1.0])
               .build())
    print(f"  Grid size: {len(lr_grid)} combos x 3 folds = {len(lr_grid)*3} fits")
    lr_cv    = CrossValidator(estimator=lr_tune,
                              estimatorParamMaps=lr_grid,
                              evaluator=evaluator,
                              numFolds=3,
                              seed=42)
    lr_cv_model = lr_cv.fit(train)
    lr_best     = lr_cv_model.bestModel
    lr_val      = lr_cv_model.transform(val)
    lr_auc      = log_classifier_metrics("LR", lr_val, evaluator, train)

    print(f"\n  [LR-CV] Best hyperparameters:")
    print(f"    regParam        = {lr_best.getRegParam()}")
    print(f"    elasticNetParam = {lr_best.getElasticNetParam()}")

    coeffs   = lr_best.coefficients.toArray()
    ranked_c = sorted(enumerate(coeffs), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  [LR] Top-10 coefficients (|beta|):")
    print(f"  {'Rank':<5} {'Feature':<25} {'Coefficient':>12}")
    print(f"  {'-'*5} {'-'*25} {'-'*12}")
    for rank, (idx, coef) in enumerate(ranked_c[:10], 1):
        sign = "+" if coef >= 0 else ""
        print(f"  {rank:<5} {FEATURE_NAMES[idx]:<25} {sign}{coef:>11.4f}")

    # Refit on full 6,000 rows with best params
    print(f"\n  [LR] Refitting on full {n_total:,} rows with best params...")
    lr_full = LogisticRegression(featuresCol="features", labelCol="label",
                                 maxIter=50,
                                 regParam=lr_best.getRegParam(),
                                 elasticNetParam=lr_best.getElasticNetParam())
    lr_full_model = lr_full.fit(df_full)
    print(f"  LR-CV training time: {elapsed(t0)}")
    results["lr"] = (lr_auc, collect_probs(lr_full_model.transform(test_df)))
    best_params["lr"] = {
        "regParam":        lr_best.getRegParam(),
        "elasticNetParam": lr_best.getElasticNetParam(),
    }

    # ------------------------------------------------------------------
    # 2. Decision Tree (kept for HW9 spec, single tree baseline)
    # ------------------------------------------------------------------
    section("[2/7] DECISION TREE CLASSIFIER")
    t0 = time.time()
    dt       = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                      maxDepth=5)
    dt_model = dt.fit(train)
    dt_val   = dt_model.transform(val)
    dt_auc   = log_classifier_metrics("DT", dt_val, evaluator, train)
    log_feature_importances(dt_model.featureImportances, "DT")
    print(f"\n  DT depth used : {dt_model.depth}  "
          f"|  nodes: {dt_model.numNodes}")
    # Refit DT on full set
    dt_full = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                     maxDepth=5).fit(df_full)
    print(f"  DT training time: {elapsed(t0)}")
    results["dt"] = (dt_auc, collect_probs(dt_full.transform(test_df)))
    best_params["dt"] = {"maxDepth": 5}

    # ------------------------------------------------------------------
    # 3. Random Forest (BIG CV grid + refit on full set)
    # ------------------------------------------------------------------
    section("[3/7] RANDOM FOREST CLASSIFIER (Big CV-tuned)")
    t0 = time.time()
    rf_tune = RandomForestClassifier(featuresCol="features", labelCol="label",
                                     seed=42)
    # NOTE: maxDepth >= 20 causes StackOverflowError during serialization
    # on Windows JVMs with default -Xss1m. Cap at 15.
    rf_grid = (ParamGridBuilder()
               .addGrid(rf_tune.numTrees,               [200, 500])
               .addGrid(rf_tune.maxDepth,               [10, 12, 15])
               .addGrid(rf_tune.featureSubsetStrategy,  ["sqrt", "log2"])
               .build())
    print(f"  Grid size: {len(rf_grid)} combos x 5 folds = {len(rf_grid)*5} fits")
    rf_cv = CrossValidator(estimator=rf_tune,
                           estimatorParamMaps=rf_grid,
                           evaluator=evaluator,
                           numFolds=5,
                           seed=42)
    rf_cv_model = rf_cv.fit(train)
    rf_best     = rf_cv_model.bestModel
    rf_val      = rf_cv_model.transform(val)
    rf_auc      = log_classifier_metrics("RF", rf_val, evaluator, train)

    print(f"\n  [RF-CV] Best hyperparameters:")
    print(f"    numTrees             = {rf_best.getNumTrees}")
    print(f"    maxDepth             = {rf_best.getMaxDepth()}")
    print(f"    featureSubsetStrategy= {rf_best.getFeatureSubsetStrategy()}")
    log_cv_fold_scores(rf_cv_model)
    log_feature_importances(rf_best.featureImportances, "RF-CV best")

    # Refit on FULL 6,000 rows -- this is the big winner from last run
    print(f"\n  [RF] Refitting on full {n_total:,} rows with best params...")
    rf_full = RandomForestClassifier(
        featuresCol="features", labelCol="label", seed=42,
        numTrees=rf_best.getNumTrees,
        maxDepth=rf_best.getMaxDepth(),
        featureSubsetStrategy=rf_best.getFeatureSubsetStrategy()
    ).fit(df_full)
    print(f"  RF-CV+refit training time: {elapsed(t0)}")
    results["rf"] = (rf_auc, collect_probs(rf_full.transform(test_df)))
    best_params["rf"] = {
        "numTrees":              rf_best.getNumTrees,
        "maxDepth":              rf_best.getMaxDepth(),
        "featureSubsetStrategy": rf_best.getFeatureSubsetStrategy(),
    }

    # ------------------------------------------------------------------
    # 4. Gradient-Boosted Tree (base)
    # ------------------------------------------------------------------
    section("[4/7] GRADIENT-BOOSTED TREE (base)")
    t0 = time.time()
    gbt       = GBTClassifier(featuresCol="features", labelCol="label",
                               maxIter=10, stepSize=0.1, seed=42)
    gbt_model = gbt.fit(train)
    gbt_val   = gbt_model.transform(val)
    gbt_auc   = log_classifier_metrics("GBT", gbt_val, evaluator, train)
    log_feature_importances(gbt_model.featureImportances, "GBT")
    gbt_full = GBTClassifier(featuresCol="features", labelCol="label",
                             maxIter=10, stepSize=0.1, seed=42).fit(df_full)
    print(f"  GBT training time: {elapsed(t0)}")
    results["gbt"] = (gbt_auc, collect_probs(gbt_full.transform(test_df)))
    best_params["gbt"] = {"maxIter": 10, "stepSize": 0.1}

    # ------------------------------------------------------------------
    # 5. GBT + Big Cross-Validated Hyperparameter Tuning + refit
    # ------------------------------------------------------------------
    section("[5/7] GBT + BIG CROSS-VALIDATION TUNING")
    t0       = time.time()
    gbt_tune = GBTClassifier(featuresCol="features", labelCol="label",
                              seed=42)
    param_grid = (ParamGridBuilder()
                  .addGrid(gbt_tune.maxDepth,        [3, 5])
                  .addGrid(gbt_tune.maxIter,         [50, 100, 200])
                  .addGrid(gbt_tune.stepSize,        [0.05, 0.1])
                  .addGrid(gbt_tune.subsamplingRate, [0.7, 1.0])
                  .build())
    print(f"  Grid size: {len(param_grid)} combos x 5 folds "
          f"= {len(param_grid)*5} fits")

    cv = CrossValidator(estimator=gbt_tune,
                        estimatorParamMaps=param_grid,
                        evaluator=evaluator,
                        numFolds=5,
                        seed=42)
    cv_model = cv.fit(train)
    cv_val   = cv_model.transform(val)
    cv_auc   = log_classifier_metrics("GBT-CV", cv_val, evaluator, train)

    best = cv_model.bestModel
    print(f"\n  [GBT-CV] Best hyperparameters:")
    print(f"    maxDepth        = {best.getMaxDepth()}")
    print(f"    maxIter         = {best.getMaxIter()}")
    print(f"    stepSize        = {best.getStepSize()}")
    print(f"    subsamplingRate = {best.getSubsamplingRate()}")
    log_cv_fold_scores(cv_model)
    log_feature_importances(best.featureImportances, "GBT-CV best")

    print(f"\n  [GBT-CV] Refitting on full {n_total:,} rows with best params...")
    gbt_cv_full = GBTClassifier(
        featuresCol="features", labelCol="label", seed=42,
        maxDepth=best.getMaxDepth(),
        maxIter=best.getMaxIter(),
        stepSize=best.getStepSize(),
        subsamplingRate=best.getSubsamplingRate()
    ).fit(df_full)
    print(f"  GBT-CV+refit training time: {elapsed(t0)}")
    results["gbt_cv"] = (cv_auc, collect_probs(gbt_cv_full.transform(test_df)))
    best_params["gbt_cv"] = {
        "maxDepth":        best.getMaxDepth(),
        "maxIter":         best.getMaxIter(),
        "stepSize":        best.getStepSize(),
        "subsamplingRate": best.getSubsamplingRate(),
    }

    # ------------------------------------------------------------------
    # 6. RF without v2_score (forces model to use raw features)
    # ------------------------------------------------------------------
    section("[6/7] RF WITHOUT v2_score (raw-feature variant)")
    t0 = time.time()
    no_v2_features = [f for f in FEATURE_NAMES if f != "v2_score"]
    print(f"  Features used: {len(no_v2_features)} (dropped v2_score)")
    asm_nov2 = VectorAssembler(inputCols=no_v2_features, outputCol="features")
    df_nov2  = asm_nov2.transform(labeled).select("label", "features",
                                                  "user_id", "track_id")
    test_nov2 = asm_nov2.transform(test_df_raw).select("user_id", "track_id",
                                                       "features")
    train_nv, val_nv = df_nov2.randomSplit([0.7, 0.3], seed=2018)

    rf_nv = RandomForestClassifier(
        featuresCol="features", labelCol="label", seed=42,
        numTrees=rf_best.getNumTrees,
        maxDepth=rf_best.getMaxDepth(),
        featureSubsetStrategy=rf_best.getFeatureSubsetStrategy()
    )
    rf_nv_model = rf_nv.fit(train_nv)
    rf_nv_val   = rf_nv_model.transform(val_nv)
    rf_nv_auc   = log_classifier_metrics("RF-no-v2", rf_nv_val, evaluator,
                                          train_nv)
    log_feature_importances(rf_nv_model.featureImportances, "RF-no-v2",
                             feature_names=no_v2_features)

    rf_nv_full = RandomForestClassifier(
        featuresCol="features", labelCol="label", seed=42,
        numTrees=rf_best.getNumTrees,
        maxDepth=rf_best.getMaxDepth(),
        featureSubsetStrategy=rf_best.getFeatureSubsetStrategy()
    ).fit(df_nov2)
    print(f"  RF-no-v2 training time: {elapsed(t0)}")
    results["rf_no_v2"] = (rf_nv_auc, collect_probs(rf_nv_full.transform(test_nov2)))
    best_params["rf_no_v2"] = {
        "numTrees":              rf_best.getNumTrees,
        "maxDepth":              rf_best.getMaxDepth(),
        "featureSubsetStrategy": rf_best.getFeatureSubsetStrategy(),
    }

    # ------------------------------------------------------------------
    # 7. Extra Trees -- RF with all-feature selection per split (more random)
    # ------------------------------------------------------------------
    section("[7/7] EXTRA TREES (RF, more randomness)")
    t0 = time.time()
    rf_extra = RandomForestClassifier(
        featuresCol="features", labelCol="label", seed=123,
        numTrees=500,
        maxDepth=15,
        featureSubsetStrategy="onethird",
        subsamplingRate=0.8
    )
    rf_extra_model = rf_extra.fit(train)
    rf_extra_val   = rf_extra_model.transform(val)
    rf_extra_auc   = log_classifier_metrics("ExtraT", rf_extra_val, evaluator,
                                             train)
    log_feature_importances(rf_extra_model.featureImportances, "ExtraT")

    rf_extra_full = RandomForestClassifier(
        featuresCol="features", labelCol="label", seed=123,
        numTrees=500, maxDepth=15,
        featureSubsetStrategy="onethird", subsamplingRate=0.8
    ).fit(df_full)
    print(f"  ExtraT training time: {elapsed(t0)}")
    results["extra"] = (rf_extra_auc,
                        collect_probs(rf_extra_full.transform(test_df)))
    best_params["extra"] = {
        "numTrees":              500,
        "maxDepth":              15,
        "featureSubsetStrategy": "onethird",
        "subsamplingRate":       0.8,
        "seed":                  123,
    }

    # ------------------------------------------------------------------
    # Side-by-side summary
    # ------------------------------------------------------------------
    section("VALIDATION AUC SUMMARY")
    display_names = {
        "lr":       "Logistic Regression CV",
        "dt":       "Decision Tree         ",
        "rf":       "Random Forest (Big CV)",
        "gbt":      "GBT (base)            ",
        "gbt_cv":   "GBT (Big CV)          ",
        "rf_no_v2": "RF without v2_score   ",
        "extra":    "Extra Trees           ",
    }
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    best_name, (best_auc, _) = sorted_results[0]
    print(f"  {'Classifier':<26} {'Val AUC':>9}  {'vs best':>8}")
    print(f"  {'-'*26} {'-'*9}  {'-'*8}")
    for name, (auc, _) in sorted_results:
        delta = auc - best_auc
        marker = "  <- BEST" if name == best_name else ""
        print(f"  {display_names[name]:<26} {auc:>9.4f}  {delta:>+8.4f}{marker}")

    # ------------------------------------------------------------------
    # Write per-classifier Kaggle submissions
    # ------------------------------------------------------------------
    section("WRITING PER-CLASSIFIER KAGGLE SUBMISSIONS")
    for name, (auc, probs) in results.items():
        write_blend_submission(probs, name, script_dir,
                                label=f"AUC={auc:.4f}")

    # ------------------------------------------------------------------
    # ENSEMBLE BLENDS  -- where the magic happens
    # ------------------------------------------------------------------
    section("ENSEMBLE BLENDS")

    def blend(weights):
        """weights: dict {classifier_name: weight}.  Returns merged prob dict."""
        keys  = list(results[next(iter(weights))][1].keys())
        out   = {}
        wsum  = sum(weights.values())
        for k in keys:
            score = 0.0
            for name, w in weights.items():
                score += w * results[name][1].get(k, 0.0)
            out[k] = score / wsum
        return out

    # 1. RF + GBT-CV equal blend  (top-2 individual models)
    print(f"\n  Building rf_gbt_blend (RF 0.5 + GBT-CV 0.5)...")
    blend_rf_gbt = blend({"rf": 1.0, "gbt_cv": 1.0})
    write_blend_submission(blend_rf_gbt, "blend_rf_gbt", script_dir,
                            label="RF + GBT-CV equal")

    # 2. RF heavy + GBT-CV light  (RF won on Kaggle -- weight it more)
    print(f"  Building rf_heavy_blend (RF 2 + GBT-CV 1)...")
    blend_rf_heavy = blend({"rf": 2.0, "gbt_cv": 1.0})
    write_blend_submission(blend_rf_heavy, "blend_rf_heavy", script_dir,
                            label="RF 2 + GBT-CV 1")

    # 3. Three-model average: RF + GBT-CV + LR
    print(f"  Building three_model_blend (RF + GBT-CV + LR equal)...")
    blend_three = blend({"rf": 1.0, "gbt_cv": 1.0, "lr": 1.0})
    write_blend_submission(blend_three, "blend_three", script_dir,
                            label="RF + GBT-CV + LR")

    # 4. Tree ensemble: RF + GBT-CV + Extra + RF-no-v2 (4 tree variants)
    print(f"  Building tree_ensemble (RF + GBT-CV + Extra + RF-no-v2)...")
    blend_trees = blend({"rf": 1.0, "gbt_cv": 1.0,
                         "extra": 1.0, "rf_no_v2": 1.0})
    write_blend_submission(blend_trees, "blend_trees", script_dir,
                            label="4-tree ensemble")

    # 5. Mega blend: ALL classifiers AUC-weighted
    print(f"  Building mega_blend (all 7 classifiers, AUC-weighted)...")
    auc_weights = {n: a for n, (a, _) in results.items()}
    blend_mega  = blend(auc_weights)
    write_blend_submission(blend_mega, "blend_mega", script_dir,
                            label="ALL, AUC-weighted")

    # 6. Rank-based blend: convert probs to within-user ranks, then average.
    #    This is more robust to probability calibration differences.
    print(f"  Building rank_blend (rank-averaged across models)...")
    rank_blend = rank_average_blend(["rf", "gbt_cv", "extra", "rf_no_v2"],
                                     results)
    write_blend_submission(rank_blend, "blend_rank", script_dir,
                            label="rank-avg of 4 trees")

    # ------------------------------------------------------------------
    # Save all per-(uid,tid) probabilities to disk for Final project use.
    #
    # CRITICAL: The 6,000 labeled (uid, tid) rows get OUT-OF-FOLD
    # predictions -- each row's prediction comes from a model that was
    # trained on a fold not containing that row.  This prevents leakage
    # when the Final project's meta-learner trains on test2_new.txt
    # labels using these probabilities as features.
    #
    # The 120,000 Kaggle test rows keep the existing full-fit predictions
    # since their labels are unknown to all training paths anyway.
    # ------------------------------------------------------------------
    section("SAVING PROBABILITY MATRIX (OOF) FOR FINAL PROJECT")
    prob_matrix_path = os.path.join(script_dir, "hw9_probabilities.csv")
    model_order = ["lr", "dt", "rf", "gbt", "gbt_cv", "rf_no_v2", "extra"]

    # Builder closures: each returns a fresh, untrained classifier with the
    # best hyperparameters from this run's CV.  Used by compute_oof_probs.
    def _build_lr():
        return LogisticRegression(featuresCol="features", labelCol="label",
                                   maxIter=50,
                                   regParam=best_params["lr"]["regParam"],
                                   elasticNetParam=best_params["lr"]
                                                   ["elasticNetParam"])

    def _build_dt():
        return DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                       maxDepth=best_params["dt"]["maxDepth"])

    def _build_rf():
        p = best_params["rf"]
        return RandomForestClassifier(
            featuresCol="features", labelCol="label", seed=42,
            numTrees=p["numTrees"], maxDepth=p["maxDepth"],
            featureSubsetStrategy=p["featureSubsetStrategy"])

    def _build_gbt():
        p = best_params["gbt"]
        return GBTClassifier(featuresCol="features", labelCol="label",
                              seed=42,
                              maxIter=p["maxIter"], stepSize=p["stepSize"])

    def _build_gbt_cv():
        p = best_params["gbt_cv"]
        return GBTClassifier(
            featuresCol="features", labelCol="label", seed=42,
            maxDepth=p["maxDepth"], maxIter=p["maxIter"],
            stepSize=p["stepSize"], subsamplingRate=p["subsamplingRate"])

    def _build_rf_no_v2():
        p = best_params["rf_no_v2"]
        return RandomForestClassifier(
            featuresCol="features", labelCol="label", seed=42,
            numTrees=p["numTrees"], maxDepth=p["maxDepth"],
            featureSubsetStrategy=p["featureSubsetStrategy"])

    def _build_extra():
        p = best_params["extra"]
        return RandomForestClassifier(
            featuresCol="features", labelCol="label", seed=p["seed"],
            numTrees=p["numTrees"], maxDepth=p["maxDepth"],
            featureSubsetStrategy=p["featureSubsetStrategy"],
            subsamplingRate=p["subsamplingRate"])

    builders = {
        "lr":       (_build_lr,       df_full),
        "dt":       (_build_dt,       df_full),
        "rf":       (_build_rf,       df_full),
        "gbt":      (_build_gbt,      df_full),
        "gbt_cv":   (_build_gbt_cv,   df_full),
        "rf_no_v2": (_build_rf_no_v2, df_nov2),   # different feature set!
        "extra":    (_build_extra,    df_full),
    }

    # Compute OOF predictions for each classifier on its labeled set.
    oof_probs = {}
    for name in model_order:
        builder, src_df = builders[name]
        oof_probs[name] = compute_oof_probs(src_df, builder,
                                             n_folds=5, seed=2018, name=name)

    # Build merged matrix:
    #   - For labeled (uid, tid) keys: use OOF predictions (no leak)
    #   - For Kaggle test (uid, tid) keys: use full-fit predictions
    print(f"\n  Merging OOF (labeled) + full-fit (Kaggle test) predictions ...")
    labeled_keys = set(oof_probs[model_order[0]].keys())
    test_keys    = set(results["rf"][1].keys()) - labeled_keys
    all_keys     = sorted(labeled_keys | test_keys)

    with open(prob_matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "track_id"] + model_order)
        for k in all_keys:
            row = [k[0], k[1]]
            for m in model_order:
                if k in labeled_keys:
                    row.append(oof_probs[m].get(k, 0.0))
                else:
                    row.append(results[m][1].get(k, 0.0))
            writer.writerow(row)

    print(f"  Wrote {len(all_keys):,} rows x {len(model_order)} models -> "
          f"{os.path.basename(prob_matrix_path)}")
    print(f"    {len(labeled_keys):,} labeled rows  (OOF predictions)")
    print(f"    {len(test_keys):,} Kaggle test rows  (full-fit predictions)")
    print(f"  This file is the un-leaked input matrix for the Final "
          f"project meta-learner.")

    section("IMPROVEMENT OPPORTUNITIES")
    print("  After this run, the natural next steps are:")
    print("  1. Submit blend_rank, blend_trees, blend_rf_heavy first --")
    print("     these are the highest-EV blends (diverse signal sources).")
    print("  2. Compare blend Kaggle scores to the best single model --")
    print("     if mega_blend > rf alone, ensembling is paying off.")
    print("  3. If RF-no-v2 scored well, it means v2_score was redundant --")
    print("     try regenerating features without it for the final.")
    print("  4. The RF-CV best (numTrees, maxDepth) tells you whether the")
    print("     grid was at a corner: if it picked the max value of a param,")
    print("     extend the grid in that direction next run.")
    print("  5. hw9_probabilities.csv is now ready as input to the Final.")

    print(f"\n  Total PySpark time: {elapsed(t_spark)}")
    spark.stop()


def rank_average_blend(model_names, results_dict):
    """Within each user, convert each model's probs to ranks (0..5), then
    average ranks across models.  Returns {(uid,tid): avg_rank}."""
    by_user_per_model = {n: defaultdict(list) for n in model_names}
    for n in model_names:
        for (uid, tid), p in results_dict[n][1].items():
            by_user_per_model[n][uid].append((tid, p))

    avg_rank = {}
    sample_user = next(iter(by_user_per_model[model_names[0]]))
    all_users   = set(by_user_per_model[model_names[0]].keys())
    for uid in all_users:
        rank_sum = defaultdict(float)
        for n in model_names:
            ranked = sorted(by_user_per_model[n][uid], key=lambda x: x[1])
            for r, (tid, _) in enumerate(ranked):
                rank_sum[tid] += r
        for tid, rs in rank_sum.items():
            avg_rank[(uid, tid)] = rs / len(model_names)
    return avg_rank


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    tee = Tee(RESULTS_FILE)
    sys.stdout = tee
    t_total = time.time()

    try:
        section("EE627 HOMEWORK 9 - PySpark Music Recommender")
        print(f"  Log file: {RESULTS_FILE}")
        print(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")

        section("LOADING RAW DATA")
        user_ratings = parse_training(TRAIN_FILE)
        track_meta   = parse_tracks(TRACK_FILE)
        album_meta   = parse_albums(ALBUM_FILE)
        test_users   = parse_test(TEST_FILE)
        labels       = parse_labels(LABEL_FILE)

        section("ENRICHMENT & AUXILIARY STRUCTURES")
        enriched = enrich_tracks(track_meta, album_meta)
        global_stats = build_global_item_stats(user_ratings)
        album_to_tracks, artist_to_tracks = build_track_mappings(enriched)

        section("BUILDING FEATURE CSVs")
        if not os.path.exists(FEATURE_CSV):
            build_labeled_csv(user_ratings, enriched, global_stats,
                              album_to_tracks, artist_to_tracks,
                              labels, FEATURE_CSV)
        else:
            print(f"  Labeled CSV already exists, skipping rebuild.")
            print(f"  (Delete {FEATURE_CSV} to force rebuild)")

        if not os.path.exists(TEST_CSV):
            build_test_csv(user_ratings, enriched, global_stats,
                           album_to_tracks, artist_to_tracks,
                           test_users, TEST_CSV)
        else:
            print(f"  Test CSV already exists, skipping rebuild.")
            print(f"  (Delete {TEST_CSV} to force rebuild)")

        run_pyspark(FEATURE_CSV, TEST_CSV, SCRIPT_DIR)

        section("COMPLETE")
        print(f"  Total wall time : {elapsed(t_total)}")
        print(f"  Full log saved  : {RESULTS_FILE}")

    finally:
        tee.close()


if __name__ == "__main__":
    main()

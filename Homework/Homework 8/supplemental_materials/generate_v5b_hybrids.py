"""
Generate hybrid submissions at multiple thresholds using v5b's predictions.
v5b hybrid at gap>5 scored 0.908 -- sweep to find the optimal threshold.
"""

import os
import csv
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data", "music-recommender-2026s")

V5B_PURE = os.path.join(SCRIPT_DIR, "submission_v5b_nn.csv")
BASELINE = os.path.join(SCRIPT_DIR, "Part 1 Results", "submission_p1_evidence.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "trainItem2.txt")
TEST_FILE = os.path.join(DATA_DIR, "testItem2.txt")
TRACK_FILE = os.path.join(DATA_DIR, "trackData2.txt")
ALBUM_FILE = os.path.join(DATA_DIR, "albumData2.txt")


def parse_training(path):
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
    return users


def parse_tracks(path):
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
    return tracks


def parse_albums(path):
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
    return albums


def enrich_tracks(tracks, albums):
    enriched = {}
    for tid, (alb, art, genres) in tracks.items():
        new_art = art
        new_genres = list(genres)
        if alb is not None and alb in albums:
            album_art, album_genres = albums[alb]
            if art is None and album_art is not None:
                new_art = album_art
            track_genre_set = set(genres)
            for g in album_genres:
                if g not in track_genre_set:
                    new_genres.append(g)
                    track_genre_set.add(g)
        enriched[tid] = (alb, new_art, new_genres)
    return enriched


def parse_test(path):
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
    return test


def read_submission(path):
    rows = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows[row[0]] = int(row[1])
    return rows


def main():
    print("Loading data ...")
    user_ratings = parse_training(TRAIN_FILE)
    track_meta = parse_tracks(TRACK_FILE)
    album_meta = parse_albums(ALBUM_FILE)
    enriched = enrich_tracks(track_meta, album_meta)
    test_users = parse_test(TEST_FILE)

    print("Loading v5b predictions ...")
    v5b_preds = read_submission(V5B_PURE)

    # Compute v2 scores and confidence gaps per user
    print("Computing v2 scores ...")
    thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]

    # Pre-compute v2 per user
    user_data = []  # (uid, candidates, v2_top3, gap)
    for uid, candidates in test_users:
        ratings = user_ratings.get(uid, {})
        v2_scores = []
        for tid in candidates:
            alb_id, art_id, genre_ids = enriched.get(tid, (None, None, []))
            s = 0.0
            if alb_id is not None and alb_id in ratings:
                s += ratings[alb_id]
            if art_id is not None and art_id in ratings:
                s += ratings[art_id]
            gs = [ratings[g] for g in genre_ids if g in ratings]
            if gs:
                s += len(gs) * (sum(gs) / len(gs)) / 10.0
            v2_scores.append(s)

        ranked = sorted(zip(v2_scores, candidates), reverse=True)
        v2_top3 = set(tid for _, tid in ranked[:3])
        ss = sorted(v2_scores, reverse=True)
        gap = ss[2] - ss[3]
        user_data.append((uid, candidates, v2_top3, gap))

    # Generate hybrids
    print(f"\n{'Thresh':>6}  {'v2 users':>10}  {'v5b users':>10}  "
          f"{'v5b pct':>8}  {'v2 agree':>10}  File")
    print("-" * 75)

    for thresh in thresholds:
        rows = []
        n_v2 = 0
        n_v5b = 0

        for uid, candidates, v2_top3, gap in user_data:
            if gap > thresh:
                n_v2 += 1
                for tid in candidates:
                    key = f"{uid}_{tid}"
                    rows.append((key, 1 if tid in v2_top3 else 0))
            else:
                n_v5b += 1
                for tid in candidates:
                    key = f"{uid}_{tid}"
                    rows.append((key, v5b_preds[key]))

        fname = f"submission_v5b_hybrid_gap{thresh}.csv"
        fpath = os.path.join(SCRIPT_DIR, fname)
        with open(fpath, "w") as f:
            f.write("TrackID,Predictor\n")
            for key, label in rows:
                f.write(f"{key},{label}\n")

        # Agreement with v2
        v2_agree = sum(1 for uid, candidates, v2_top3, _ in user_data
                       for tid in candidates
                       if (v5b_preds.get(f"{uid}_{tid}", 0)
                           == (1 if tid in v2_top3 else 0)))
        # Actually compute for this hybrid
        total = len(rows)
        v5b_pct = 100 * n_v5b / len(user_data)

        print(f"{thresh:>6}  {n_v2:>10,}  {n_v5b:>10,}  "
              f"{v5b_pct:>7.1f}%  {'':>10}  {fname}")

    print(f"\nDone. {len(thresholds)} hybrid submissions generated.")
    print(f"\nv5b hybrid gap5 scored 0.908 on Kaggle.")
    print(f"Try gap3, gap4, gap6, gap7 to bracket the optimum.")


if __name__ == "__main__":
    main()

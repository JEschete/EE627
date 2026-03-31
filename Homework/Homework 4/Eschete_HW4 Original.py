"""
EE627 Homework 4 - Yahoo Music Recommendation
Jude Eschete

Approach: Multi-signal scoring combining content-based filtering
(genre, artist, album preferences) with collaborative filtering
(similar-user overlap) and track popularity.

For each test user's 6 candidate tracks:
  - Score using genre/artist/album match, popularity, and CF signals
  - Recommend top 3 (label=1), not recommend bottom 3 (label=0)
"""

import math
import os
from collections import defaultdict

# --- File paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "eschete_submission.csv")


def parse_training(path):
    """Parse training data: user rating histories.
    Format: UserID|NumTracks, then TrackID\\tRating lines."""
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
    total_ratings = sum(len(r) for r in users.values())
    print(f"  {len(users)} users, {total_ratings} ratings")
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
    """Parse test data: UserID|NumTracks, then TrackID lines."""
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


def main():
    # ---- Load all data ----
    user_ratings = parse_training(os.path.join(DATA_DIR, "trainItem2.txt"))
    track_meta = parse_tracks(os.path.join(DATA_DIR, "trackData2.txt"))
    test_users = parse_test(os.path.join(DATA_DIR, "testItem2.txt"))

    # ---- Track popularity stats ----
    print("Computing track statistics...")
    track_count = defaultdict(int)
    track_rsum = defaultdict(float)
    for ratings in user_ratings.values():
        for tid, r in ratings.items():
            track_count[tid] += 1
            track_rsum[tid] += r
    max_log_pop = math.log1p(max(track_count.values())) if track_count else 1.0

    # ---- Inverted index: track -> set of users who rated it (for CF) ----
    print("Building collaborative filtering index...")
    track_raters = defaultdict(set)
    for uid, ratings in user_ratings.items():
        for tid in ratings:
            track_raters[tid].add(uid)

    # ---- User profiles: genre preferences, known artists/albums ----
    print("Building user profiles...")
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

    # ---- Score candidates and generate predictions ----
    print("Generating predictions...")
    results = []

    for i, (uid, candidates) in enumerate(test_users):
        if i % 5000 == 0 and i > 0:
            print(f"  {i}/{len(test_users)} users processed...")

        has_profile = uid in user_genres
        gp = user_genres.get(uid, {})
        arts = user_artists.get(uid, set())
        albs = user_albums.get(uid, set())

        # Build similar-user set for collaborative filtering
        # Use user's top 30 highest-rated tracks to find like-minded users
        similar_users = set()
        if uid in user_ratings:
            sorted_tracks = sorted(
                user_ratings[uid].items(), key=lambda x: x[1], reverse=True
            )[:30]
            for t, r in sorted_tracks:
                if t in track_raters:
                    similar_users.update(track_raters[t])
                if len(similar_users) > 50000:
                    break
            similar_users.discard(uid)

        # Compute raw scores for each signal
        raw_pop = []
        raw_genre = []
        raw_cf = []
        artist_match = []
        album_match = []

        for tid in candidates:
            # (1) Popularity: log-scaled count, normalized
            if tid in track_count:
                raw_pop.append(math.log1p(track_count[tid]) / max_log_pop)
            else:
                raw_pop.append(0.0)

            # (2) Genre preference match: sum of user's genre weights for track's genres
            gs = 0.0
            if has_profile and tid in track_meta:
                for g in track_meta[tid][2]:
                    gs += gp.get(g, 0.0)
            raw_genre.append(gs)

            # (3) Artist match: has user listened to this artist before?
            am = 0.0
            if has_profile and tid in track_meta:
                art = track_meta[tid][1]
                if art is not None and art in arts:
                    am = 1.0
            artist_match.append(am)

            # (4) Album match: has user listened to tracks from this album?
            abm = 0.0
            if has_profile and tid in track_meta:
                alb = track_meta[tid][0]
                if alb is not None and alb in albs:
                    abm = 1.0
            album_match.append(abm)

            # (5) CF: fraction of similar users who also rated this candidate
            cf = 0.0
            if similar_users and tid in track_raters:
                overlap = len(track_raters[tid] & similar_users)
                cf = overlap / len(similar_users)
            raw_cf.append(cf)

        # Normalize genre and CF scores within this user's 6 candidates
        def normalize(vals):
            mx, mn = max(vals), min(vals)
            if mx == mn:
                return [0.5] * len(vals)
            return [(v - mn) / (mx - mn) for v in vals]

        genre_n = normalize(raw_genre)
        cf_n = normalize(raw_cf)

        # Weighted combination of all signals
        final = []
        for j in range(len(candidates)):
            s = (
                0.10 * raw_pop[j]
                + 0.25 * genre_n[j]
                + 0.25 * artist_match[j]
                + 0.15 * album_match[j]
                + 0.25 * cf_n[j]
            )
            final.append((candidates[j], s))

        # Rank: top 3 -> recommend (1), bottom 3 -> don't (0)
        final.sort(key=lambda x: x[1], reverse=True)
        top3 = set(f[0] for f in final[:3])

        for tid in candidates:
            results.append((f"{uid}_{tid}", 1 if tid in top3 else 0))

    # ---- Write submission CSV ----
    print(f"Writing {len(results)} predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="") as f:
        f.write("TrackID,Predictor\n")
        for key, label in results:
            f.write(f"{key},{label}\n")

    print(f"\nDone! {len(results)} predictions saved to eschete_submission.csv")
    print("Upload this file to Kaggle.")


if __name__ == "__main__":
    main()

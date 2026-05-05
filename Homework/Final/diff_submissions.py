"""Pairwise + multi-way diff/agreement analysis for Kaggle submissions.

For each user, the submissions agree on a top-3 selection (3 of 6 = 1).
Two submissions can disagree on:
  - 0 candidates  (identical for that user)
  - 2 candidates  (one swap: one rec became not-rec, one not-rec became rec)
  - 4 candidates  (two swaps)
  - 6 candidates  (max disagreement -- inverted top-3)

Reports:
  1. Pairwise table: total disagreed rows + users with any disagreement
  2. Multi-way: rows where ALL submissions agree vs at least one disagrees
  3. Per-user disagreement histogram
  4. The actual disagreed (uid, tid) rows for the closest pair, so you
     can eyeball where the bets differ.
"""

import csv
import os
import sys
from collections import defaultdict
from itertools import combinations


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_submission(path):
    """Returns {(uid, tid): 0_or_1}.  Key format in CSV is 'uid_tid'."""
    out = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            key  = row[0]
            pred = int(row[1])
            uid_str, tid_str = key.split("_")
            out[(int(uid_str), int(tid_str))] = pred
    return out


def per_user_picks(sub):
    """{uid: frozenset of tids predicted 1}."""
    by_user = defaultdict(set)
    for (uid, tid), pred in sub.items():
        if pred == 1:
            by_user[uid].add(tid)
    return {uid: frozenset(tids) for uid, tids in by_user.items()}


def pairwise_diff(sub_a, sub_b):
    """Return:
      n_rows_disagree : int
      n_users_disagree : int
      per_user_disagreement : {uid: int n disagreed candidates (0,2,4,6)}
    """
    keys = sub_a.keys() & sub_b.keys()
    if len(keys) != len(sub_a) or len(keys) != len(sub_b):
        print(f"    WARN: key mismatch -- a:{len(sub_a)} b:{len(sub_b)} "
              f"shared:{len(keys)}")
    n_rows = 0
    per_user = defaultdict(int)
    for k in keys:
        if sub_a[k] != sub_b[k]:
            n_rows  += 1
            per_user[k[0]] += 1
    n_users = len(per_user)
    return n_rows, n_users, dict(per_user)


def multiway_agreement(subs_dict):
    """How many rows do ALL N submissions agree on?

    Returns:
      n_all_agree   : rows where every sub gives the same prediction
      n_any_disagree : rows where at least one sub differs
      n_users_split : users where at least one row disagrees somewhere
    """
    keys = None
    for s in subs_dict.values():
        keys = s.keys() if keys is None else keys & s.keys()
    n_all_agree = 0
    n_any_disagree = 0
    users_split = set()
    for k in keys:
        vals = {s[k] for s in subs_dict.values()}
        if len(vals) == 1:
            n_all_agree += 1
        else:
            n_any_disagree += 1
            users_split.add(k[0])
    return n_all_agree, n_any_disagree, len(users_split)


def hist(per_user_disagreement):
    """Distribution of disagreement counts (always even: 0/2/4/6)."""
    h = defaultdict(int)
    for v in per_user_disagreement.values():
        h[v] += 1
    return dict(sorted(h.items()))


def main():
    targets = [
        ("v1",        "submission_final_stacked.csv",        "PastLogs/V3"),
        ("v2",        "submission_final_stacked_v2.csv",     "PastLogs/V3"),
        ("v2_nnls",   "submission_final_stacked_v2_nnls.csv","PastLogs/V3"),
        ("v3_all",    "submission_final_stacked_v3.csv",     "PastLogs/V3"),
        ("v3_C",      "submission_final_stacked_v3_C_ndcg.csv","PastLogs/V3"),
        ("v3_A",      "submission_final_stacked_v3_A_lgbm.csv","PastLogs/V3"),
        ("v3_B",      "submission_final_stacked_v3_B_iso.csv","PastLogs/V3"),
        ("v3_V2ref",  "submission_final_stacked_v3_V2ref.csv","PastLogs/V3"),
        ("v4",        "submission_final_stacked_v4.csv",     "."),
        ("v4_iicf",   "submission_final_stacked_v4_iicf_only.csv","."),
        ("hybrid10",  "submission_final_v5_hybrid_gap10.csv","PastLogs/V3"),
        ("v5_nn",     "submission_final_v5_nn.csv",          "PastLogs/V3"),
    ]

    print("=" * 70)
    print("  SUBMISSION DIFF / AGREEMENT ANALYSIS")
    print("=" * 70)

    subs = {}
    for name, fname, subdir in targets:
        path = os.path.join(SCRIPT_DIR, subdir, fname)
        if not os.path.exists(path):
            # try root final dir as fallback
            alt = os.path.join(SCRIPT_DIR, fname)
            if os.path.exists(alt):
                path = alt
            else:
                print(f"  [skip] {name}: not found")
                continue
        subs[name] = load_submission(path)
        n_rec = sum(subs[name].values())
        print(f"  Loaded {name:<10} ({n_rec:,} rec / {len(subs[name]):,} rows)  "
              f"<- {os.path.relpath(path, SCRIPT_DIR)}")

    if len(subs) < 2:
        print("\nNeed >= 2 submissions to compare.")
        return

    print()
    print("=" * 70)
    print("  PAIRWISE AGREEMENT")
    print("=" * 70)
    print(f"  Two submissions disagreeing on a user always swap an even "
          f"count\n  of candidates (2/4/6); 'rows' counts each candidate "
          f"flip,\n  'users' counts users with any disagreement.\n")
    print(f"  {'A':<10} {'B':<10} {'rows diff':>10} "
          f"{'users diff':>12} {'% rows':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
    closest = None  # (n_rows, name_a, name_b, per_user)
    for a, b in combinations(subs.keys(), 2):
        n_rows, n_users, per_u = pairwise_diff(subs[a], subs[b])
        pct = 100.0 * n_rows / len(subs[a])
        print(f"  {a:<10} {b:<10} {n_rows:>10,} {n_users:>12,} "
              f"{pct:>7.2f}%")
        if closest is None or (n_rows < closest[0] and n_rows > 0):
            closest = (n_rows, a, b, per_u)

    print()
    print("=" * 70)
    print("  MULTI-WAY AGREEMENT (all submissions vs any disagreement)")
    print("=" * 70)
    n_all, n_any, n_split = multiway_agreement(subs)
    total = n_all + n_any
    print(f"  Submissions in pool : {len(subs)}")
    print(f"  Total rows          : {total:,}")
    print(f"  All agree           : {n_all:,}  ({100*n_all/total:.2f}%)")
    print(f"  At least 1 disagrees: {n_any:,}  ({100*n_any/total:.2f}%)")
    print(f"  Users with any split: {n_split:,}  "
          f"({100*n_split/(total//6):.2f}% of {total//6:,} users)")

    # Restricted to the three near-tied 0.919 submissions
    tied_names = [n for n in ("v2", "v3_all", "v3_C") if n in subs]
    if len(tied_names) >= 2:
        print()
        print("=" * 70)
        print(f"  AGREEMENT WITHIN THE 0.919 CLUSTER ({', '.join(tied_names)})")
        print("=" * 70)
        sub_set = {n: subs[n] for n in tied_names}
        n_all, n_any, n_split = multiway_agreement(sub_set)
        total = n_all + n_any
        print(f"  Total rows          : {total:,}")
        print(f"  All 0.919 agree     : {n_all:,}  ({100*n_all/total:.2f}%)")
        print(f"  At least 1 differs  : {n_any:,}  ({100*n_any/total:.2f}%)")
        print(f"  Users with any split: {n_split:,}  "
              f"({100*n_split/(total//6):.2f}% of {total//6:,} users)")
        print(f"\n  -> the {n_any:,} disagreed rows are where these "
              f"submissions place\n     different bets while scoring "
              f"identically on Kaggle.")

    # v4 (0.922) vs 0.919 cluster -- where did iicf actually move things?
    if "v4" in subs and tied_names:
        print()
        print("=" * 70)
        print(f"  v4 (0.922) vs 0.919 CLUSTER -- WHERE iicf MOVED THINGS")
        print("=" * 70)
        # Build a "0.919 majority" reference: 1 if >=2 of cluster predict 1
        cluster_votes = defaultdict(int)
        for name in tied_names:
            for k, v in subs[name].items():
                cluster_votes[k] += v
        majority_919 = {k: 1 if v >= 2 else 0
                          for k, v in cluster_votes.items()}
        # Compare v4 against that majority
        n_rows_diff = 0
        n_v4_added  = 0   # v4 predicts 1, majority predicts 0
        n_v4_dropped = 0  # v4 predicts 0, majority predicts 1
        users_changed = set()
        for k, v4_pred in subs["v4"].items():
            maj = majority_919.get(k)
            if maj is None or maj == v4_pred:
                continue
            n_rows_diff += 1
            users_changed.add(k[0])
            if v4_pred == 1:
                n_v4_added += 1
            else:
                n_v4_dropped += 1
        print(f"  Rows where v4 differs from 0.919 majority : "
              f"{n_rows_diff:,}")
        print(f"    v4 newly recommends  (0->1)            : "
              f"{n_v4_added:,}")
        print(f"    v4 newly drops       (1->0)            : "
              f"{n_v4_dropped:,}")
        print(f"  Users where v4 changed at least one pick : "
              f"{len(users_changed):,}")
        print(f"\n  Kaggle lift v4 - cluster = +0.003")
        print(f"  Approx rows that flipped CORRECT in v4   : ~{int(0.003 * 120000):,}")
        print(f"  Net good flips out of {n_rows_diff:,} changes "
              f"= roughly {int(0.003 * 120000) / max(n_rows_diff,1) * 100:.0f}% "
              f"of v4's changes were wins.")

    if closest is not None:
        n_rows, a, b, per_u = closest
        print()
        print("=" * 70)
        print(f"  PER-USER DISAGREEMENT HISTOGRAM (closest pair: "
              f"{a} vs {b})")
        print("=" * 70)
        print(f"  {'# disagreements':<20} {'# users':>10}")
        print(f"  {'-'*20} {'-'*10}")
        # If a user has 0 disagreements, they're not in per_u; reconstruct.
        all_users = set()
        for k in subs[a].keys():
            all_users.add(k[0])
        zero_users = len(all_users) - len(per_u)
        print(f"  {'0  (identical)':<20} {zero_users:>10,}")
        for n_diff, n_u in hist(per_u).items():
            label = f"{n_diff}  ({n_diff//2} swap{'s' if n_diff//2>1 else ''})"
            print(f"  {label:<20} {n_u:>10,}")

    print()
    print("=" * 70)
    print("  INTERPRETATION HINTS")
    print("=" * 70)
    print("  - If your 0.919 submissions disagree on N rows out of 120,000")
    print("    but all score 0.919, those N rows are 50/50 coin-flips for")
    print("    your stack.  An ensemble that resolves them better wins.")
    print("  - 0.921 vs 0.919 = 0.002 lift = ~240 rows out of 120K.")
    print("    Compare against the 'rows diff' column above -- if your")
    print("    closest pair already disagrees on ~240+ rows, the gap is")
    print("    pure tie-breaker variance, not a fundamentally better model.")
    print("  - Majority-vote across your 0.919 submissions targets exactly")
    print("    these ambiguous rows.  Worth a sub.")


if __name__ == "__main__":
    main()

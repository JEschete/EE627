"""
EE627 HW4 - Final Blended Submission Generator

Builds one reproducible final Kaggle CSV in a single run by blending:
1) Heuristic scorer from Eschete_HW4.py (continuous scores)
2) Scikit-learn model from HW4_ML_Model.py (continuous scores)

The weights here represent the values that gave the best kaggle score on the public leaderboard.

Example:
  python Final_Submission.py
  python Final_Submission.py --quick
  python Final_Submission.py --output submissions/final_blend.csv
"""

import argparse
import os
import time

import Eschete_HW4_Heur as heur
import Eschete_HW4_ML_Model as ml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "eschete_submission.csv")


def log(msg):
    print(msg, flush=True)


def rank_normalize(score_map):
    items = sorted(score_map.items(), key=lambda kv: kv[1])
    n = len(items)
    denom = max(1, n - 1)
    out = {}
    for rank, (key, _) in enumerate(items):
        out[key] = rank / denom
    return out


def write_output(test_users, blended_map, output_path, binary):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = 0
    ones = 0

    with open(output_path, "w", newline="") as f:
        f.write("TrackID,Predictor\n")

        if binary:
            for uid, candidates in test_users:
                keys = [f"{uid}_{tid}" for tid in candidates]
                scored = [(k, blended_map[k]) for k in keys]
                top3 = set(
                    k for k, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:3]
                )
                for k in keys:
                    label = 1 if k in top3 else 0
                    ones += label
                    total += 1
                    f.write(f"{k},{label}\n")
        else:
            for uid, candidates in test_users:
                for tid in candidates:
                    key = f"{uid}_{tid}"
                    total += 1
                    f.write(f"{key},{blended_map[key]:.8f}\n")

    if binary:
        log(
            f"  wrote binary predictions: rows={total:,}, ones={ones:,}, zeros={total - ones:,}"
        )
    else:
        vals = list(blended_map.values())
        log(
            f"  wrote continuous predictions: rows={total:,}, "
            f"min={min(vals):.8f}, max={max(vals):.8f}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Generate final blended HW4 submission")

    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--binary",
        action="store_true",
        help="Write top-3 binary labels instead of continuous blend",
    )

    # Heuristic side
    p.add_argument(
        "--heur-content", type=float, default=0.80, help="Heuristic content weight"
    )
    p.add_argument("--heur-cf", type=float, default=0.20, help="Heuristic CF weight")

    # ML side
    p.add_argument("--ml-n-users", type=int, default=25000, help="ML sampled users")
    p.add_argument(
        "--ml-neg-per-pos", type=int, default=2, help="ML negatives per positive"
    )
    p.add_argument(
        "--ml-max-iter", type=int, default=450, help="ML max boosting iterations"
    )

    # Blend side
    p.add_argument(
        "--blend-heur",
        type=float,
        default=0.50,
        help="Blend weight for heuristic rank scores",
    )
    p.add_argument(
        "--blend-ml", type=float, default=0.50, help="Blend weight for ML rank scores"
    )

    p.add_argument("--quick", action="store_true", help="Quick smoke mode")

    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.ml_n_users = min(args.ml_n_users, 2500)
        args.ml_max_iter = min(args.ml_max_iter, 120)

    weight_sum = args.blend_heur + args.blend_ml
    if weight_sum <= 0:
        raise ValueError("Blend weights must sum to a positive value.")

    blend_heur = args.blend_heur / weight_sum
    blend_ml = args.blend_ml / weight_sum

    log("=" * 72)
    log("Final HW4 Blended Submission")
    log(f"  output={args.output}")
    log(f"  binary={args.binary}")
    log(f"  heuristic weights: content={args.heur_content:.3f}, cf={args.heur_cf:.3f}")
    log(
        f"  ml params: n_users={args.ml_n_users:,}, "
        f"neg_per_pos={args.ml_neg_per_pos}, max_iter={args.ml_max_iter}"
    )
    log(f"  blend weights (normalized): heur={blend_heur:.3f}, ml={blend_ml:.3f}")
    log("=" * 72)

    t0_all = time.time()

    # Parse once
    log("\n[1/6] Loading data...")
    t0 = time.time()
    user_ratings = heur.parse_training(os.path.join(DATA_DIR, "trainItem2.txt"))
    track_meta = heur.parse_tracks(os.path.join(DATA_DIR, "trackData2.txt"))
    test_users = heur.parse_test(os.path.join(DATA_DIR, "testItem2.txt"))
    log(f"  data loaded in {time.time() - t0:.1f}s")

    # Heuristic continuous scores
    log("\n[2/6] Building heuristic model...")
    t0 = time.time()
    extra_tids = set()
    for _, cands in test_users:
        extra_tids.update(cands)
    heur_model = heur.build_model(
        user_ratings, track_meta, extra_tids=extra_tids, use_svd=False
    )
    heur_scores = heur.score_users(
        heur_model,
        test_users,
        (0.0, args.heur_content, args.heur_cf),
        output_continuous=True,
    )
    heur_map = {k: float(v) for k, v in heur_scores}
    log(f"  heuristic scoring done in {time.time() - t0:.1f}s")

    # ML continuous scores
    log("\n[3/6] Building ML features and samples...")
    t0 = time.time()
    ml_stats = ml.build_stats(user_ratings, track_meta)
    X, y = ml.generate_training_samples(
        user_ratings,
        ml_stats,
        n_users=args.ml_n_users,
        neg_per_pos=args.ml_neg_per_pos,
        seed=args.seed,
    )
    log(f"  ML sample generation done in {time.time() - t0:.1f}s")

    log("\n[4/6] Training ML model...")
    t0 = time.time()
    ml_model, feat_mean, feat_std = ml.train_model(
        X, y, max_iter=args.ml_max_iter, seed=args.seed
    )
    log(f"  ML training done in {time.time() - t0:.1f}s")

    log("\n[5/6] Scoring ML model on test users...")
    t0 = time.time()
    ml_scores = ml.score_test_users(
        ml_model, test_users, ml_stats, feat_mean, feat_std, continuous=True
    )
    ml_map = {k: float(v) for k, v in ml_scores}
    log(f"  ML scoring done in {time.time() - t0:.1f}s")

    # Blend by global rank to reduce scale mismatch
    log("\n[6/6] Rank blending and writing final CSV...")
    t0 = time.time()
    common = set(heur_map.keys()) & set(ml_map.keys())
    if len(common) != len(heur_map) or len(common) != len(ml_map):
        raise RuntimeError(
            f"Key mismatch during blending: heur={len(heur_map)}, ml={len(ml_map)}, common={len(common)}"
        )

    heur_rank = rank_normalize(heur_map)
    ml_rank = rank_normalize(ml_map)

    blended = {}
    for key in common:
        blended[key] = blend_heur * heur_rank[key] + blend_ml * ml_rank[key]

    write_output(test_users, blended, args.output, args.binary)
    log(f"  blend+write done in {time.time() - t0:.1f}s")

    log(f"\nDone. Total runtime: {time.time() - t0_all:.1f}s")


if __name__ == "__main__":
    main()

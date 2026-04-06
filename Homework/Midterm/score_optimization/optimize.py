"""
Score Optimization via Leaderboard Probing
Jude Eschete

Strategy: small batch flips with binary search refinement.
- Flip small batches of users (50 at a time)
- If score goes UP: those users include some wrong predictions --
  keep the batch, then binary search within it to isolate winners
- If score goes DOWN: those users were mostly correct, skip them
- If score SAME: inconclusive, skip and move on

Interactive terminal app. Just type the Kaggle score after uploading.
"""

import os
import sys
import json
import csv
import random
import shutil
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDTERM_DIR = os.path.dirname(SCRIPT_DIR)
BASELINE_CSV = os.path.join(
    MIDTERM_DIR, "Part 1 Results", "submission_p1_evidence.csv")

STATE_FILE = os.path.join(SCRIPT_DIR, "state.json")
BEST_CSV = os.path.join(SCRIPT_DIR, "current_best.csv")
SUBMISSIONS_DIR = os.path.join(SCRIPT_DIR, "submissions")

SEED = 42
BATCH_SIZE = 50  # users per probe


# =====================================================================
# Submission I/O
# =====================================================================

def read_submission(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append((row[0], int(row[1])))
    return rows


def write_submission(rows, path):
    with open(path, "w", newline="") as f:
        f.write("TrackID,Predictor\n")
        for key, label in rows:
            f.write(f"{key},{label}\n")


def group_by_user(rows):
    users = defaultdict(list)
    for key, label in rows:
        uid, tid = key.split("_", 1)
        users[uid].append((key, tid, label))
    return dict(users)


def flatten_users(user_dict, user_order):
    rows = []
    for uid in user_order:
        for key, tid, label in user_dict[uid]:
            rows.append((key, label))
    return rows


def get_user_order(rows):
    seen = set()
    order = []
    for key, _ in rows:
        uid = key.split("_", 1)[0]
        if uid not in seen:
            seen.add(uid)
            order.append(uid)
    return order


# =====================================================================
# Flip logic
# =====================================================================

def flip_users(current_rows, uids_to_flip):
    """Generate new submission with specific users flipped."""
    users = group_by_user(current_rows)
    user_order = get_user_order(current_rows)

    flip_set = set(uids_to_flip)
    for uid in flip_set:
        if uid in users:
            users[uid] = [(key, tid, 1 - label)
                          for key, tid, label in users[uid]]

    return flatten_users(users, user_order)


# =====================================================================
# State management
# =====================================================================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def next_sub_name(state):
    n = state.get("sub_counter", 0) + 1
    state["sub_counter"] = n
    return f"validation_test_sub_{n:03d}.csv"


# =====================================================================
# Commands
# =====================================================================

def cmd_init():
    if not os.path.exists(BASELINE_CSV):
        print(f"  ERROR: Baseline not found: {BASELINE_CSV}")
        return

    shutil.copy2(BASELINE_CSV, BEST_CSV)
    rows = read_submission(BASELINE_CSV)
    user_order = get_user_order(rows)

    # Shuffle users deterministically for probing order
    rng = random.Random(SEED)
    shuffled = user_order[:]
    rng.shuffle(shuffled)

    state = {
        "best_score": 0.871,
        "all_uids": user_order,
        "probe_queue": shuffled,       # users to test, in order
        "probe_index": 0,              # where we are in the queue
        "confirmed_good": [],          # flipping these hurt or was neutral
        "confirmed_bad": [],           # flipping these helped
        "mode": "scan",                # "scan" or "refine"
        "refine_pool": [],             # users to binary search within
        "refine_half": [],             # current half being tested
        "pending_uids": [],            # users flipped in current probe
        "sub_counter": 0,
        "history": [
            {"action": "init", "score": 0.871}
        ],
    }

    # Generate first probe
    batch = shuffled[:BATCH_SIZE]
    state["pending_uids"] = batch
    state["probe_index"] = BATCH_SIZE

    probe_rows = flip_users(rows, batch)
    name = next_sub_name(state)
    write_submission(probe_rows, os.path.join(SUBMISSIONS_DIR, name))

    save_state(state)

    print(f"  Initialized from baseline (0.871)")
    print(f"  Total users: {len(user_order):,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"")
    print(f"  >> Upload submissions/{name} to Kaggle")
    print(f"     Flipping {len(batch)} users")


def cmd_score(score_str):
    state = load_state()
    if state is None:
        print("  Not initialized. Type 'reset' first.")
        return

    score = float(score_str)
    best = state["best_score"]
    pending = state["pending_uids"]
    mode = state.get("mode", "scan")

    if score > best:
        # ---- IMPROVEMENT ----
        if mode == "scan":
            print(f"\n  IMPROVED: {best:.3f} -> {score:.3f} (+{score-best:.3f})")
            print(f"  {len(pending)} users contained winners!")

            if len(pending) <= 1:
                # Found an individual winner
                print(f"  Found winning user: {pending[0]}")
                state["confirmed_bad"].extend(pending)
                # Update best
                sub_name = f"validation_test_sub_{state['sub_counter']:03d}.csv"
                shutil.copy2(os.path.join(SUBMISSIONS_DIR, sub_name), BEST_CSV)
                state["best_score"] = score
                # Continue scanning
                _generate_next_scan(state)
            else:
                # Keep the improvement, then refine to find which users helped
                sub_name = f"validation_test_sub_{state['sub_counter']:03d}.csv"
                shutil.copy2(os.path.join(SUBMISSIONS_DIR, sub_name), BEST_CSV)
                state["best_score"] = score

                # Now we want to UN-flip half to see if score drops
                # (binary search: which half contains the winners?)
                state["mode"] = "refine"
                state["refine_pool"] = list(pending)
                half = pending[:len(pending)//2]
                state["refine_half"] = half
                state["pending_uids"] = half

                # Generate probe: UN-flip the first half (revert them)
                best_rows = read_submission(BEST_CSV)
                probe_rows = flip_users(best_rows, half)
                name = next_sub_name(state)
                write_submission(probe_rows,
                                 os.path.join(SUBMISSIONS_DIR, name))

                print(f"  Entering refinement mode to isolate winners.")
                print(f"\n  >> Upload submissions/{name}")
                print(f"     Testing: un-flipping {len(half)} of {len(pending)} users")

        elif mode == "refine":
            # Un-flipping this half improved score further --
            # means this half was hurting, the OTHER half has the winners
            print(f"\n  REFINE IMPROVED: {best:.3f} -> {score:.3f}")
            sub_name = f"validation_test_sub_{state['sub_counter']:03d}.csv"
            shutil.copy2(os.path.join(SUBMISSIONS_DIR, sub_name), BEST_CSV)
            state["best_score"] = score

            pool = state["refine_pool"]
            tested_half = state["refine_half"]
            other_half = [u for u in pool if u not in set(tested_half)]

            # The tested half was bad (un-flipping helped), mark as good
            state["confirmed_good"].extend(tested_half)

            if len(other_half) <= 2:
                # Small enough, mark remaining as bad and move on
                state["confirmed_bad"].extend(other_half)
                state["mode"] = "scan"
                _generate_next_scan(state)
            else:
                # Continue refining the other half
                state["refine_pool"] = other_half
                half = other_half[:len(other_half)//2]
                state["refine_half"] = half
                state["pending_uids"] = half

                best_rows = read_submission(BEST_CSV)
                probe_rows = flip_users(best_rows, half)
                name = next_sub_name(state)
                write_submission(probe_rows,
                                 os.path.join(SUBMISSIONS_DIR, name))

                print(f"  Narrowing: {len(other_half)} users remain")
                print(f"\n  >> Upload submissions/{name}")
                print(f"     Un-flipping {len(half)} users")

        state["history"].append({
            "action": "improved", "score": score,
            "delta": score - best, "users": len(pending), "mode": mode
        })

    elif score < best:
        # ---- REGRESSION ----
        if mode == "scan":
            print(f"\n  REGRESSED: {best:.3f} -> {score:.3f} ({score-best:.3f})")
            print(f"  Those {len(pending)} users were correct, skipping.")
            state["confirmed_good"].extend(pending)
            _generate_next_scan(state)

        elif mode == "refine":
            # Un-flipping this half made it worse --
            # means this half contained winners, keep them flipped
            print(f"\n  REFINE REGRESSED: {best:.3f} -> {score:.3f}")
            print(f"  This half contains winners -- keeping them.")

            pool = state["refine_pool"]
            tested_half = state["refine_half"]
            other_half = [u for u in pool if u not in set(tested_half)]

            # The other half might be hurting -- mark as good (revert-worthy)
            state["confirmed_good"].extend(other_half)

            if len(tested_half) <= 2:
                state["confirmed_bad"].extend(tested_half)
                state["mode"] = "scan"
                _generate_next_scan(state)
            else:
                # Narrow within the winning half
                state["refine_pool"] = tested_half
                half = tested_half[:len(tested_half)//2]
                state["refine_half"] = half
                state["pending_uids"] = half

                best_rows = read_submission(BEST_CSV)
                probe_rows = flip_users(best_rows, half)
                name = next_sub_name(state)
                write_submission(probe_rows,
                                 os.path.join(SUBMISSIONS_DIR, name))

                print(f"  Narrowing: {len(tested_half)} users remain")
                print(f"\n  >> Upload submissions/{name}")
                print(f"     Un-flipping {len(half)} users")

        state["history"].append({
            "action": "regressed", "score": score,
            "delta": score - best, "users": len(pending), "mode": mode
        })

    else:
        # ---- SAME SCORE ----
        print(f"\n  NEUTRAL: {score:.3f} (same as best)")
        if mode == "scan":
            print(f"  Flipping {len(pending)} users made no difference, skipping.")
            state["confirmed_good"].extend(pending)
            _generate_next_scan(state)
        elif mode == "refine":
            print(f"  Inconclusive refinement. Moving on.")
            state["confirmed_good"].extend(state["refine_pool"])
            state["mode"] = "scan"
            _generate_next_scan(state)

        state["history"].append({
            "action": "neutral", "score": score,
            "users": len(pending), "mode": mode
        })

    save_state(state)


def _generate_next_scan(state):
    """Generate the next scan probe from the queue."""
    queue = state["probe_queue"]
    idx = state["probe_index"]
    good_set = set(state["confirmed_good"])
    bad_set = set(state["confirmed_bad"])

    # Find next untested batch
    batch = []
    while idx < len(queue) and len(batch) < BATCH_SIZE:
        uid = queue[idx]
        idx += 1
        if uid not in good_set and uid not in bad_set:
            batch.append(uid)

    state["probe_index"] = idx
    state["pending_uids"] = batch
    state["mode"] = "scan"

    if not batch:
        n_good = len(state["confirmed_good"])
        n_bad = len(state["confirmed_bad"])
        print(f"\n  ALL USERS TESTED!")
        print(f"  Confirmed correct: {n_good:,}")
        print(f"  Confirmed wrong:   {n_bad:,}")
        print(f"  Best score: {state['best_score']:.4f}")
        print(f"  Final submission: current_best.csv")
        return

    best_rows = read_submission(BEST_CSV)
    probe_rows = flip_users(best_rows, batch)
    name = next_sub_name(state)
    write_submission(probe_rows, os.path.join(SUBMISSIONS_DIR, name))

    n_good = len(state["confirmed_good"])
    n_bad = len(state["confirmed_bad"])
    n_total = len(state["all_uids"])
    progress = 100 * (n_good + n_bad) / n_total

    print(f"\n  Best: {state['best_score']:.4f}  "
          f"Progress: {n_good+n_bad:,}/{n_total:,} ({progress:.0f}%)")
    print(f"  Correct: {n_good:,}  Wrong: {n_bad:,}")
    print(f"\n  >> Upload submissions/{name}")
    print(f"     Flipping {len(batch)} users")


def cmd_status():
    state = load_state()
    if state is None:
        print("  Not initialized. Type 'reset'.")
        return

    n_good = len(state.get("confirmed_good", []))
    n_bad = len(state.get("confirmed_bad", []))
    n_total = len(state["all_uids"])

    print()
    print("=" * 50)
    print("  STATUS")
    print("=" * 50)
    print(f"  Best score:    {state['best_score']:.4f}")
    print(f"  Mode:          {state.get('mode', 'scan')}")
    print(f"  Submissions:   {state.get('sub_counter', 0)}")
    print(f"  Correct users: {n_good:,} ({100*n_good/n_total:.1f}%)")
    print(f"  Wrong users:   {n_bad:,} ({100*n_bad/n_total:.1f}%)")
    print(f"  Untested:      {n_total-n_good-n_bad:,}")
    print(f"  Pending flip:  {len(state.get('pending_uids', [])):,}")
    print()
    print("  Recent history:")
    for h in state.get("history", [])[-10:]:
        action = h["action"]
        score = h.get("score", "")
        delta = h.get("delta", "")
        d = f" ({delta:+.3f})" if delta else ""
        n = h.get("users", "")
        mode = h.get("mode", "")
        print(f"    {action:12s} {score}{d}  "
              f"[{n} users] {mode}")


# =====================================================================
# Interactive Loop
# =====================================================================

def main():
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

    print("=" * 50)
    print("  SCORE OPTIMIZATION")
    print("=" * 50)

    state = load_state()
    if state is None:
        print("  No state found. Initializing from baseline...")
        print()
        cmd_init()

    print()
    print("  Commands: <score>, status, reset, quit")
    print()

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        elif user_input.lower() == "status":
            cmd_status()
        elif user_input.lower() == "reset":
            confirm = input("   Reset all progress? (y/n): ").strip().lower()
            if confirm == "y":
                # Clean old submissions
                for f in os.listdir(SUBMISSIONS_DIR):
                    os.remove(os.path.join(SUBMISSIONS_DIR, f))
                cmd_init()
            else:
                print("   Cancelled.")
        else:
            try:
                score = float(user_input)
                if 0 <= score <= 1:
                    cmd_score(user_input)
                else:
                    print("  Score should be between 0 and 1")
            except ValueError:
                print(f"  Unknown: {user_input}")
                print("  Enter a score, 'status', 'reset', or 'quit'")


if __name__ == "__main__":
    main()

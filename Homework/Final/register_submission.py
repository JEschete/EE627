"""
register_submission.py - Register a new timestamped LS-ensemble run.

Run this after uploading a fresh submission_final_ensemble_Eschete_<ts>.csv
to Kaggle.  It will:
  1. Detect any such CSVs in this directory not yet in past_predict_ensemble/.
  2. Prompt for the Kaggle score for each.
  3. Move the CSV into past_predict_ensemble/.
  4. Append a new line under the "--- LS ensemble ---" section of
     Submission Results.txt (existing data left untouched).

Usage:
  python register_submission.py
  python register_submission.py <path-to-csv>      # register a specific file
"""

import os
import re
import shutil
import sys

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_DIR   = os.path.join(SCRIPT_DIR, "past_predict_ensemble")
RESULTS_FILE   = os.path.join(SCRIPT_DIR, "Submission Results.txt")
SECTION_HEADER = "--- LS ensemble ---"
PATTERN        = re.compile(
    r"^submission_final_ensemble_Eschete_\d{8}_\d{6}\.csv$")


def find_new_files():
    """CSVs in SCRIPT_DIR matching the timestamped ensemble pattern that
    aren't already present in past_predict_ensemble/."""
    if not os.path.isdir(ENSEMBLE_DIR):
        os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    existing = set(os.listdir(ENSEMBLE_DIR))
    return sorted(
        f for f in os.listdir(SCRIPT_DIR)
        if PATTERN.match(f) and f not in existing
    )


def prompt_score(filename):
    """Ask for a Kaggle score; blank input skips the file."""
    while True:
        raw = input(f"  Kaggle score for {filename} "
                    f"(blank to skip): ").strip()
        if not raw:
            return None
        try:
            v = float(raw)
        except ValueError:
            print(f"    '{raw}' is not a number. Try again.")
            continue
        if not (0.0 <= v <= 1.0):
            print(f"    {v} is outside [0, 1]. Try again.")
            continue
        return v


def append_to_results(filename, score):
    """Append `<score>  <filename>` under the LS ensemble section.

    - Existing lines are NOT reordered or rewritten.
    - If the filename is already logged, the function is a no-op.
    - If the section header doesn't exist, it's created near the top.
    """
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_line = f"{score:.3f}  {filename}\n"

    start_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip() == SECTION_HEADER),
        None,
    )

    if start_idx is None:
        # Insert a fresh section above the first --- header (or at EOF).
        insert_at = next(
            (i for i, ln in enumerate(lines)
             if ln.strip().startswith("---")),
            len(lines),
        )
        block = [SECTION_HEADER + "\n", new_line, "\n"]
        lines[insert_at:insert_at] = block
    else:
        # End of section = next --- header (or EOF).
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].strip().startswith("---"):
                end_idx = i
                break

        # Idempotency: if filename is already in this section, do nothing.
        for ln in lines[start_idx + 1:end_idx]:
            if filename in ln:
                print(f"  Already logged in Submission Results.txt: "
                      f"{filename}  (no change)")
                return

        # Insert just after the last non-blank line of the section so the
        # blank separator before the next --- header is preserved.
        insert_at = end_idx
        while (insert_at > start_idx + 1
               and lines[insert_at - 1].strip() == ""):
            insert_at -= 1
        lines[insert_at:insert_at] = [new_line]

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


def register(filename, score):
    src = os.path.join(SCRIPT_DIR, filename)
    dst = os.path.join(ENSEMBLE_DIR, filename)
    if not os.path.exists(src):
        print(f"  ERROR: {src} not found")
        return False
    if os.path.exists(dst):
        print(f"  ERROR: {dst} already exists -- not overwriting")
        return False
    shutil.move(src, dst)
    append_to_results(filename, score)
    print(f"  Registered  {filename}  ->  {score:.3f}")
    return True


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        filename = os.path.basename(path)
        if not PATTERN.match(filename):
            print(f"Filename does not match the timestamped ensemble "
                  f"pattern: {filename}")
            sys.exit(1)
        # Allow absolute or relative paths; normalize to SCRIPT_DIR location.
        if os.path.dirname(os.path.abspath(path)) != SCRIPT_DIR:
            print(f"Expected file in {SCRIPT_DIR}; got "
                  f"{os.path.abspath(path)}")
            sys.exit(1)
        new_files = [filename]
    else:
        new_files = find_new_files()

    if not new_files:
        print("No new timestamped ensemble submissions to register.")
        return

    print(f"Found {len(new_files)} candidate file(s):")
    for f in new_files:
        print(f"  - {f}")
    print()

    registered = 0
    for filename in new_files:
        score = prompt_score(filename)
        if score is None:
            print(f"  Skipped     {filename}")
            continue
        if register(filename, score):
            registered += 1

    print(f"\nDone. Registered {registered}/{len(new_files)} file(s).")


if __name__ == "__main__":
    main()

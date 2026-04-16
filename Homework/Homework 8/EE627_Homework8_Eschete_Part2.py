"""
EE627A Homework 8 - Part 2: PySpark ALS Parameter Sweeps on re_u.data
Jude Eschete

Three experiments per the HW8 Part 2 instructions:
  1. Fix maxIter=20. Vary rank in {5, 7, 10, 20}. Measure MSE.
  2. Fix rank=20. Vary maxIter in {2, 5, 10, 20}. Measure MSE.
  3. Fix rank=20, maxIter=20. Vary data size in
     {2000, 5000, 10000, 20000, 50000, 100000}. Measure MSE.

Data: re_u.data (MovieLens-100k style)
  Format: userID,itemID,rating    (rating in 1-5, 100000 rows total)
  Taken from Week 11 supplemental materials, copied into Data/.

Per the instructions, data slices for experiment 3 are built with
`sc.textFile(...).take(n)` (first n rows), then converted to DataFrame,
mirroring the code fragment in the assignment.

Outputs:
  - Results/part2_run_YYYYMMDD_HHMMSS.txt   : full teed stdout log
  - Results/part2_metrics_YYYYMMDD_HHMMSS.csv : one row per ALS run with
      experiment, rank, maxIter, data_size, holdout_rmse, holdout_mse
  - Results/error_YYYYMMDD_HHMMSS.log       : crash report on failure
"""

import os
import sys
import csv
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Environment setup (same pattern as Part 1).
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
    "-Djava.security.manager=allow",
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

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "Data", "re_u.data")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = os.path.join(RESULTS_DIR, f"part2_run_{TIMESTAMP}.txt")
METRICS_CSV = os.path.join(RESULTS_DIR, f"part2_metrics_{TIMESTAMP}.csv")

SEED = 42
HOLDOUT_FRACTION = 0.2  # 80/20 split per run

# Experiment 1: fix maxIter, sweep rank
EXP1_MAX_ITER = 20
EXP1_RANKS = [5, 7, 10, 20]

# Experiment 2: fix rank, sweep maxIter
EXP2_RANK = 20
EXP2_MAX_ITERS = [2, 5, 10, 20]

# Experiment 3: fix rank + maxIter, sweep data size
EXP3_RANK = 20
EXP3_MAX_ITER = 20
EXP3_SIZES = [2000, 5000, 10000, 20000, 50000, 100000]


class _Tee:
    """Duplicate stdout/stderr writes to a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
RATING_SCHEMA = StructType([
    StructField("userID", IntegerType(), False),
    StructField("itemID", IntegerType(), False),
    StructField("rating", FloatType(), False),
])


def load_full(spark):
    """Load the entire re_u.data file as a DataFrame."""
    df = spark.read.csv(DATA_FILE, header=False, schema=StructType([
        StructField("userID", IntegerType(), True),
        StructField("itemID", IntegerType(), True),
        StructField("rating", FloatType(), True),
    ]))
    return df


def load_first_n(spark, n):
    """Take the first n rows of re_u.data per the HW8 instructions example:

        data = sc.textFile("re_u.data")
        pData = data.take(n)

    Then convert the resulting list back into a DataFrame for ALS.
    """
    lines = spark.sparkContext.textFile(DATA_FILE).take(n)
    rows = []
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        rows.append(Row(userID=int(parts[0]),
                        itemID=int(parts[1]),
                        rating=float(parts[2])))
    return spark.createDataFrame(rows, schema=RATING_SCHEMA)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------
def run_als(df, rank, max_iter, label, metrics_writer):
    """Train ALS with the given config, return holdout RMSE/MSE.

    Splits df 80/20 for honest held-out evaluation. Uses the Week 11
    sample code ALS configuration (nonnegative, coldStart=drop).
    """
    train_df, holdout_df = df.randomSplit(
        [1.0 - HOLDOUT_FRACTION, HOLDOUT_FRACTION], seed=SEED
    )

    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=0.05,
        userCol="userID",
        itemCol="itemID",
        ratingCol="rating",
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop",
        seed=SEED,
    )

    model = als.fit(train_df)
    preds = model.transform(holdout_df)
    evaluator = RegressionEvaluator(metricName="rmse",
                                    labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(preds)
    mse = rmse ** 2

    total_rows = df.count()
    train_rows = train_df.count()
    holdout_rows = holdout_df.count()
    print(f"    rank={rank:<3d} maxIter={max_iter:<3d} "
          f"size={total_rows:<7d} train={train_rows:<6d} "
          f"holdout={holdout_rows:<6d} "
          f"RMSE={rmse:.4f}   MSE={mse:.4f}")

    metrics_writer.writerow({
        "experiment": label,
        "rank": rank,
        "maxIter": max_iter,
        "data_size": total_rows,
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "holdout_rmse": f"{rmse:.6f}",
        "holdout_mse": f"{mse:.6f}",
    })
    return rmse, mse


def experiment_rank_sweep(spark, full_df, metrics_writer):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: rank sweep")
    print(f"  Fixed maxIter={EXP1_MAX_ITER}")
    print(f"  Varying rank in {EXP1_RANKS}")
    print("=" * 60)
    results = []
    for rank in EXP1_RANKS:
        rmse, mse = run_als(full_df, rank=rank, max_iter=EXP1_MAX_ITER,
                            label="exp1_rank_sweep",
                            metrics_writer=metrics_writer)
        results.append((rank, rmse, mse))
    return results


def experiment_iter_sweep(spark, full_df, metrics_writer):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: maxIter sweep")
    print(f"  Fixed rank={EXP2_RANK}")
    print(f"  Varying maxIter in {EXP2_MAX_ITERS}")
    print("=" * 60)
    results = []
    for max_iter in EXP2_MAX_ITERS:
        rmse, mse = run_als(full_df, rank=EXP2_RANK, max_iter=max_iter,
                            label="exp2_maxIter_sweep",
                            metrics_writer=metrics_writer)
        results.append((max_iter, rmse, mse))
    return results


def experiment_size_sweep(spark, metrics_writer):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: data size sweep")
    print(f"  Fixed rank={EXP3_RANK}, maxIter={EXP3_MAX_ITER}")
    print(f"  Varying size in {EXP3_SIZES}")
    print("  (using sc.textFile.take(n) per the HW8 instructions)")
    print("=" * 60)
    results = []
    for size in EXP3_SIZES:
        df = load_first_n(spark, size)
        rmse, mse = run_als(df, rank=EXP3_RANK, max_iter=EXP3_MAX_ITER,
                            label="exp3_size_sweep",
                            metrics_writer=metrics_writer)
        results.append((size, rmse, mse))
    return results


def summarize(exp1, exp2, exp3):
    def mse_range(results):
        mses = [r[2] for r in results]
        return max(mses) - min(mses), min(mses), max(mses)

    e1_delta, e1_min, e1_max = mse_range(exp1)
    e2_delta, e2_min, e2_max = mse_range(exp2)
    e3_delta, e3_min, e3_max = mse_range(exp3)

    print("\n" + "=" * 60)
    print("  SUMMARY: which factor moves MSE the most?")
    print("=" * 60)
    print(f"  Exp 1 (rank sweep):     MSE range = {e1_delta:.4f}   "
          f"[{e1_min:.4f} .. {e1_max:.4f}]")
    print(f"  Exp 2 (maxIter sweep):  MSE range = {e2_delta:.4f}   "
          f"[{e2_min:.4f} .. {e2_max:.4f}]")
    print(f"  Exp 3 (data size):      MSE range = {e3_delta:.4f}   "
          f"[{e3_min:.4f} .. {e3_max:.4f}]")

    ranked = sorted(
        [("rank", e1_delta), ("maxIter", e2_delta), ("data size", e3_delta)],
        key=lambda x: x[1], reverse=True,
    )
    print("\n  Most impactful -> least impactful:")
    for i, (name, delta) in enumerate(ranked, start=1):
        print(f"    {i}. {name:10s}   delta_MSE = {delta:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log_fh = open(RESULTS_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    print("=" * 60)
    print("  EE627A Homework 8 - PART 2: Parameter Sweeps")
    print("  Jude Eschete")
    print(f"  Run started: {datetime.now().isoformat()}")
    print("=" * 60)
    print(f"  Data file: {DATA_FILE}")
    print(f"  Results log: {RESULTS_PATH}")
    print(f"  Metrics CSV: {METRICS_CSV}")

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("EE627_HW8_Part2")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"  Spark {spark.version} ready")

    full_df = load_full(spark)
    print(f"  Loaded {full_df.count():,} rows from re_u.data")
    full_df.show(5)

    metrics_fh = open(METRICS_CSV, "w", encoding="utf-8", newline="")
    metrics_writer = csv.DictWriter(
        metrics_fh,
        fieldnames=[
            "experiment", "rank", "maxIter", "data_size",
            "train_rows", "holdout_rows",
            "holdout_rmse", "holdout_mse",
        ],
    )
    metrics_writer.writeheader()

    exp1 = experiment_rank_sweep(spark, full_df, metrics_writer)
    exp2 = experiment_iter_sweep(spark, full_df, metrics_writer)
    exp3 = experiment_size_sweep(spark, metrics_writer)

    metrics_fh.close()

    summarize(exp1, exp2, exp3)

    print(f"\n  Run log:     {RESULTS_PATH}")
    print(f"  Metrics CSV: {METRICS_CSV}")
    print("\nDone.")
    log_fh.close()
    spark.stop()


def _log_crash(exc):
    err_path = os.path.join(
        RESULTS_DIR,
        f"part2_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    with open(err_path, "w", encoding="utf-8") as fh:
        fh.write("=" * 60 + "\n")
        fh.write(f"  CRASH: {type(exc).__name__}  (Part 2)\n")
        fh.write(f"  When:  {datetime.now().isoformat()}\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"{type(exc).__name__}: {exc}\n\n")
        fh.write("Traceback:\n")
        fh.write("-" * 60 + "\n")
        traceback.print_exc(file=fh)
        fh.write("\n" + "-" * 60 + "\n")
    index_path = os.path.join(RESULTS_DIR, "errors_index.log")
    with open(index_path, "a", encoding="utf-8") as fh:
        fh.write(
            f"{datetime.now().isoformat()}  PART2  "
            f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}  "
            f"-> {os.path.basename(err_path)}\n"
        )
    return err_path


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err_path = _log_crash(exc)
        print(f"\n[!] Part 2 crashed: {type(exc).__name__}: {exc}",
              file=sys.__stderr__)
        print(f"[!] Full traceback written to: {err_path}",
              file=sys.__stderr__)
        sys.exit(1)

"""
Higgs Boson Machine Learning Challenge - XGBoost Approach
=========================================================
Author: Jude
Course: Stevens Institute of Technology - Applied AI
Date: February 2026

Objective:
    Classify events as signal ('s') or background ('b') using the ATLAS
    experiment dataset. The evaluation metric is the Approximate Median
    Significance (AMS), which measures the statistical significance of
    the Higgs boson signal over background noise.

Approach:
    - XGBoost gradient-boosted decision tree classifier
    - Custom AMS metric integration for threshold optimization
    - Feature engineering for missing values (-999.0 sentinel)
    - Weighted classification to respect physics event weights
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_PATH = "training/training.csv"
TEST_PATH = "test/test.csv"
OUTPUT_SUBMISSION = "random_submission/Eschete_Submission.csv"
RANDOM_SEED = 42
N_FOLDS = 5

# GPU Detection — fallback to CPU if CUDA is unavailable
USE_GPU = False
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        USE_GPU = True
        print("[GPU] NVIDIA GPU detected — using CUDA acceleration")
        for line in result.stdout.split('\n'):
            if 'RTX' in line or 'GTX' in line or 'Tesla' in line or 'A100' in line:
                print(f"[GPU] {line.strip()}")
                break
    else:
        print("[GPU] nvidia-smi failed — falling back to CPU")
except FileNotFoundError:
    print("[GPU] nvidia-smi not found — falling back to CPU")

# ============================================================================
# 1. AMS METRIC IMPLEMENTATION
# ============================================================================
def AMS(s, b, b_reg=10.0):
    """
    Approximate Median Significance (AMS).
    AMS = sqrt(2 * ((s + b + b_reg) * ln(1 + s/(b + b_reg)) - s))

    Parameters:
        s: weighted true positive rate (signal correctly classified as signal)
        b: weighted false positive rate (background incorrectly classified as signal)
        b_reg: regularization constant (default=10)
    """
    return np.sqrt(2.0 * ((s + b + b_reg) * np.log(1.0 + s / (b + b_reg)) - s))


# ============================================================================
# 2. DATA LOADING & EXPLORATION
# ============================================================================
print("=" * 70)
print("HIGGS BOSON ML CHALLENGE - XGBoost Approach")
print("=" * 70)

print("\n[1] Loading data...")
t0 = time.time()
train_df = pd.read_csv(TRAIN_PATH)
print(f"    Training set loaded: {train_df.shape[0]:,} events, {train_df.shape[1]} columns")
print(f"    Time: {time.time()-t0:.2f}s")

# Identify feature columns (all DER_* and PRI_* columns)
feature_cols = [c for c in train_df.columns if c.startswith(('DER_', 'PRI_'))]
print(f"    Features: {len(feature_cols)}")

# Class Distribution
label_counts = train_df['Label'].value_counts()
print(f"\n[2] Class Distribution:")
print(f"    Background (b): {label_counts['b']:,} ({label_counts['b']/len(train_df)*100:.1f}%)")
print(f"    Signal (s):     {label_counts['s']:,} ({label_counts['s']/len(train_df)*100:.1f}%)")

# Missing Value Analysis
print(f"\n[3] Missing Value Analysis (sentinel value = -999.0):")
missing_counts = {}
for col in feature_cols:
    n_missing = (train_df[col] == -999.0).sum()
    if n_missing > 0:
        missing_counts[col] = n_missing

print(f"    Features with missing values: {len(missing_counts)} / {len(feature_cols)}")
for col, count in sorted(missing_counts.items(), key=lambda x: -x[1]):
    print(f"      {col:30s}: {count:>7,} ({count/len(train_df)*100:5.1f}%)")

# Feature Statistics
print(f"\n[4] Feature Statistics (non-missing values):")
X_train_raw = train_df[feature_cols].copy()
X_stats = X_train_raw.replace(-999.0, np.nan).describe().T
print(X_stats[['mean', 'std', 'min', 'max']].to_string())


# ============================================================================
# 3. FEATURE ENGINEERING & PREPROCESSING
# ============================================================================
print(f"\n[5] Preprocessing...")

# Replace sentinel with NaN (XGBoost handles NaN natively)
X_train = train_df[feature_cols].replace(-999.0, np.nan).values
y_train = (train_df['Label'] == 's').astype(int).values  # 1=signal, 0=background
weights = train_df['Weight'].values

# Compute scale_pos_weight for class imbalance
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos = n_neg / n_pos
print(f"    scale_pos_weight: {scale_pos:.4f}")
print(f"    NaN replacement: -999.0 -> NaN for native XGBoost handling")


# ============================================================================
# 4. XGBOOST MODEL CONFIGURATION
# ============================================================================
print(f"\n[6] XGBoost Configuration:")

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,        # L1 regularization
    'reg_lambda': 1.0,        # L2 regularization
    'scale_pos_weight': scale_pos,
    'seed': RANDOM_SEED,
    'tree_method': 'hist',
    'device': 'cuda' if USE_GPU else 'cpu',
    'n_jobs': -1,
}

for k, v in xgb_params.items():
    print(f"    {k:25s}: {v}")


# ============================================================================
# 5. CROSS-VALIDATION WITH AMS TRACKING
# ============================================================================
print(f"\n[7] {N_FOLDS}-Fold Stratified Cross-Validation:")
print("    (Tracking AMS across folds to estimate leaderboard performance)")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_ams_scores = []
fold_auc_scores = []
fold_thresholds = []
oof_predictions = np.zeros(len(y_train))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n    --- Fold {fold_idx+1}/{N_FOLDS} ---")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    w_tr, w_val = weights[train_idx], weights[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val, feature_names=feature_cols)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    best_iter = model.best_iteration
    val_auc = model.best_score
    print(f"    Best iteration: {best_iter}, Val AUC: {val_auc:.6f}")

    # Get predicted probabilities for AMS threshold optimization
    val_probs = model.predict(dval)
    oof_predictions[val_idx] = val_probs

    # AMS Threshold Optimization
    best_ams = 0
    best_thresh = 0.5

    for thresh in np.arange(0.05, 0.95, 0.01):
        pred_labels = (val_probs >= thresh).astype(int)

        # Compute weighted s and b
        s = w_val[(pred_labels == 1) & (y_val == 1)].sum()
        b = w_val[(pred_labels == 1) & (y_val == 0)].sum()

        if s > 0 and b > 0:
            ams_val = AMS(s, b)
            if ams_val > best_ams:
                best_ams = ams_val
                best_thresh = thresh

    print(f"    Best AMS: {best_ams:.4f} at threshold: {best_thresh:.2f}")
    fold_ams_scores.append(best_ams)
    fold_auc_scores.append(val_auc)
    fold_thresholds.append(best_thresh)

print(f"\n    Cross-Validation Summary:")
print(f"    {'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print(f"    {'-'*55}")
print(f"    {'AMS':<15} {np.mean(fold_ams_scores):>10.4f} {np.std(fold_ams_scores):>10.4f} "
      f"{np.min(fold_ams_scores):>10.4f} {np.max(fold_ams_scores):>10.4f}")
print(f"    {'AUC':<15} {np.mean(fold_auc_scores):>10.6f} {np.std(fold_auc_scores):>10.6f} "
      f"{np.min(fold_auc_scores):>10.6f} {np.max(fold_auc_scores):>10.6f}")
print(f"    {'Threshold':<15} {np.mean(fold_thresholds):>10.4f} {np.std(fold_thresholds):>10.4f} "
      f"{np.min(fold_thresholds):>10.4f} {np.max(fold_thresholds):>10.4f}")


# ============================================================================
# 6. TRAIN FINAL MODEL ON FULL DATASET
# ============================================================================
print(f"\n[8] Training final model on full training set...")

num_rounds_final = 500

dtrain_full = xgb.DMatrix(X_train, label=y_train, weight=weights, feature_names=feature_cols)

final_model = xgb.train(
    xgb_params,
    dtrain_full,
    num_boost_round=num_rounds_final,
    evals=[(dtrain_full, 'train')],
    verbose_eval=100
)

# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print(f"\n[9] Feature Importance (top 15 by gain):")
importance = final_model.get_score(importance_type='gain')
importance_sorted = sorted(importance.items(), key=lambda x: -x[1])

print(f"    {'Feature':<35} {'Gain':>12}")
print(f"    {'-'*47}")
for feat, gain in importance_sorted[:15]:
    bar = '█' * int(gain / importance_sorted[0][1] * 30)
    print(f"    {feat:<35} {gain:>12.2f}  {bar}")

# Save feature importance plot
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    top_feats = importance_sorted[:15]
    feat_names = [f[0] for f in top_feats][::-1]
    feat_gains = [f[1] for f in top_feats][::-1]
    ax.barh(feat_names, feat_gains, color='steelblue')
    ax.set_xlabel('Gain')
    ax.set_title('XGBoost Feature Importance (Top 15 by Gain)')
    plt.tight_layout()
    fig.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("    [Saved: feature_importance.png]")
except Exception as e:
    print(f"    [Could not save plot: {e}]")


# ============================================================================
# 8. THRESHOLD OPTIMIZATION ON FULL TRAINING SET (OOF Predictions)
# ============================================================================
print(f"\n[10] Global Threshold Optimization (using OOF predictions):")

best_ams_global = 0
best_thresh_global = 0.5
ams_vs_thresh = []

for thresh in np.arange(0.05, 0.95, 0.005):
    pred_labels = (oof_predictions >= thresh).astype(int)
    s = weights[(pred_labels == 1) & (y_train == 1)].sum()
    b = weights[(pred_labels == 1) & (y_train == 0)].sum()

    if s > 0 and b > 0:
        ams_val = AMS(s, b)
        ams_vs_thresh.append((thresh, ams_val))
        if ams_val > best_ams_global:
            best_ams_global = ams_val
            best_thresh_global = thresh

print(f"    Optimal threshold: {best_thresh_global:.4f}")
print(f"    Optimal AMS:       {best_ams_global:.4f}")

# Save AMS vs threshold plot
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    thresholds_plot = [t[0] for t in ams_vs_thresh]
    ams_plot = [t[1] for t in ams_vs_thresh]
    ax.plot(thresholds_plot, ams_plot, 'b-', linewidth=2)
    ax.axvline(best_thresh_global, color='red', linestyle='--', label=f'Optimal: {best_thresh_global:.3f}')
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('AMS Score')
    ax.set_title('AMS Score vs Classification Threshold (OOF Predictions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig('ams_vs_threshold.png', dpi=150)
    plt.close()
    print("    [Saved: ams_vs_threshold.png]")
except Exception as e:
    print(f"    [Could not save plot: {e}]")


# ============================================================================
# 9. GENERATE TEST PREDICTIONS & SUBMISSION FILE
# ============================================================================
print(f"\n[11] Generating test predictions...")

if os.path.exists(TEST_PATH):
    test_df = pd.read_csv(TEST_PATH)
    print(f"    Test set loaded: {test_df.shape[0]:,} events")

    X_test = test_df[feature_cols].replace(-999.0, np.nan).values
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)

    test_probs = final_model.predict(dtest)

    # Apply optimal threshold
    test_labels = np.where(test_probs >= best_thresh_global, 's', 'b')

    # Create RankOrder (higher rank = more signal-like)
    rank_order = test_probs.argsort().argsort() + 1  # 1-indexed ranking

    submission = pd.DataFrame({
        'EventId': test_df['EventId'],
        'RankOrder': rank_order.astype(int),
        'Class': test_labels
    })

    submission.to_csv(OUTPUT_SUBMISSION, index=False)

    n_signal = (test_labels == 's').sum()
    n_background = (test_labels == 'b').sum()
    print(f"    Predicted signals:     {n_signal:,} ({n_signal/len(test_labels)*100:.1f}%)")
    print(f"    Predicted background:  {n_background:,} ({n_background/len(test_labels)*100:.1f}%)")
    print(f"    Submission saved to:   {OUTPUT_SUBMISSION}")
    print(f"    Submission shape:      {submission.shape}")
    print(f"\n    First 5 rows:")
    print(submission.head().to_string(index=False))
else:
    print(f"    WARNING: {TEST_PATH} not found. Skipping submission generation.")
    print(f"    Place test.csv in the working directory and re-run.")


# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print(f"""
{'='*70}
FINAL SUMMARY
{'='*70}

MODEL PERFORMANCE:
   - Cross-validated AMS: {np.mean(fold_ams_scores):.4f} +/- {np.std(fold_ams_scores):.4f}
   - Cross-validated AUC: {np.mean(fold_auc_scores):.6f} +/- {np.std(fold_auc_scores):.6f}
   - Optimal threshold:   {best_thresh_global:.4f} (vs default 0.5)
{'='*70}
""")
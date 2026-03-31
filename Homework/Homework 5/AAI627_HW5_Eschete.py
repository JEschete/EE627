import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
matplotlib.rcParams['figure.dpi'] = 150

# ===================
# Part 1: Manual ROC Curve Construction
# ===================

probabilities  = [0.967, 0.448, 0.568, 0.879, 0.015, 0.780, 0.978, 0.004]
classifications = [1,     0,     1,     0,     1,     0,     1,     0    ]

n_pos = sum(classifications)          # 4
n_neg = len(classifications) - n_pos  # 4

# Sort by descending probability
sorted_pairs = sorted(zip(probabilities, classifications), reverse=True)

# Build ROC points
fpr_points = [0.0]
tpr_points = [0.0]
tp = 0
fp = 0
for prob, label in sorted_pairs:
    if label == 1:
        tp += 1
    else:
        fp += 1
    fpr_points.append(fp / n_neg)
    tpr_points.append(tp / n_pos)

# AUC via trapezoidal rule
auc_p1 = np.trapezoid(tpr_points, fpr_points)

print("Sorted (prob, class):", sorted_pairs)
print("FPR points:", fpr_points)
print("TPR points:", tpr_points)
print(f"Part 1 AUC = {auc_p1:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr_points, tpr_points, 'b-o', linewidth=2, markersize=6,
        label=f'ROC Curve (AUC = {auc_p1:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('ROC Curve - Part 1', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('roc_part1.png', bbox_inches='tight')
plt.close()
print("Saved roc_part1.png")

# ===================
# Part 2: Logistic Regression on dataSet_1.mat
# ===================

mat = scipy.io.loadmat(
    'd:/judee/Google Drive/School/Classes/EE627/Homework/Homework 5/dataSet_1.mat'
)
predictor = mat['predictor']   # (4000, 476)
response  = mat['response'].ravel().astype(int)  # (4000,)

print(f"\nDataset: {predictor.shape[0]} samples, {predictor.shape[1]} features")
print(f"Class balance: {response.sum()} positive, {(response==0).sum()} negative")

# -------------------------------------------------------
# Task 1: Full-dataset logistic regression
# -------------------------------------------------------
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(predictor)

clf_full = LogisticRegression(max_iter=2000, solver='lbfgs')
clf_full.fit(X_full_scaled, response)
prob_full = clf_full.predict_proba(X_full_scaled)[:, 1]

fpr_full, tpr_full, _ = roc_curve(response, prob_full)
auc_full = auc(fpr_full, tpr_full)
print(f"\nTask 1 - Full dataset AUC = {auc_full:.4f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr_full, tpr_full, 'b-', linewidth=2,
        label=f'ROC Curve (AUC = {auc_full:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('ROC Curve - Task 1 (Full Dataset)', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('roc_task1.png', bbox_inches='tight')
plt.close()
print("Saved roc_task1.png")

# -------------------------------------------------------
# Task 2: Train/validation split (rows 1-3000 / 3001-4000)
# -------------------------------------------------------
X_train = predictor[:3000, :]
y_train = response[:3000]
X_val   = predictor[3000:, :]
y_val   = response[3000:]

scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)
X_val_scaled   = scaler_train.transform(X_val)

clf_train = LogisticRegression(max_iter=2000, solver='lbfgs')
clf_train.fit(X_train_scaled, y_train)

prob_train = clf_train.predict_proba(X_train_scaled)[:, 1]
prob_val   = clf_train.predict_proba(X_val_scaled)[:, 1]

fpr_train, tpr_train, _ = roc_curve(y_train, prob_train)
fpr_val,   tpr_val,   _ = roc_curve(y_val,   prob_val)
auc_train = auc(fpr_train, tpr_train)
auc_val   = auc(fpr_val,   tpr_val)

print(f"\nTask 2 - Training AUC   = {auc_train:.4f}")
print(f"Task 2 - Validation AUC = {auc_val:.4f}")
print(f"Task 2 - Difference     = {auc_train - auc_val:.4f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr_train, tpr_train, 'b-', linewidth=2,
        label=f'Training   (AUC = {auc_train:.4f})')
ax.plot(fpr_val, tpr_val, 'r-', linewidth=2,
        label=f'Validation (AUC = {auc_val:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('ROC Curves - Task 2 (Train vs. Validation)', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('roc_task2.png', bbox_inches='tight')
plt.close()
print("Saved roc_task2.png")

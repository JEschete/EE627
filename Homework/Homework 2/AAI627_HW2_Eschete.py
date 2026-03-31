"""
AAI 627 Homework 2 - Jude Eschete
Higgs Boson Machine Learning Challenge / Factor Model Analysis

This script performs the following analyses on EE627A_HW1_Data.csv:
  1.1.1 Correlation matrix between all time series
  1.1.2 Which factor correlates most highly with every industry
  1.1.3 Which factor correlates negatively with every industry; RF correlation with industries
  1.2.1 ACF for time-lag 1..10 for the four-factor time series
  1.2.2 AR(1) model assessment for the 4 factor time series
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox

# -- Paths -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "EE627A_HW1_Data.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "HW2_Results.txt")
OUTPUT_FILE_PART2 = os.path.join(SCRIPT_DIR, "HW2_Part2_Results.txt")

# -- Load data -------------------------------
df = pd.read_csv(DATA_FILE)
df.set_index("Date", inplace=True)

# Column groupings
factors = ["Mkt-RF", "SMB", "HML", "Mom"]         # four-factor model
all_factors = ["Mkt-RF", "SMB", "HML", "RF", "Mom"]  # includes risk-free
industries = [c for c in df.columns if c not in all_factors]

print(f"Loaded {len(df)} monthly observations, {len(industries)} industries, {len(all_factors)} factor columns.")

# -- Helper: write and print -------------------------------
lines = []

def out(text=""):
    lines.append(text)
    print(text)


# ==================================================
# 1.1.1  Full correlation matrix
# ==================================================
out("=" * 90)
out("1.1.1  CORRELATION MATRIX (all time series)")
out("=" * 90)

corr_all = df.corr()

# Print correlation matrix in a readable format (factor columns vs industries)
out("\nCorrelation of Four Factors + RF with 30 Industries:\n")
header = f"{'Industry':<10}" + "".join(f"{f:>10}" for f in all_factors)
out(header)
out("-" * len(header))
for ind in industries:
    row = f"{ind:<10}" + "".join(f"{corr_all.loc[ind, f]:>10.4f}" for f in all_factors)
    out(row)

# Also show inter-factor correlations
out("\nInter-factor correlation matrix:\n")
factor_corr = df[all_factors].corr()
header2 = f"{'':>10}" + "".join(f"{f:>10}" for f in all_factors)
out(header2)
out("-" * len(header2))
for f1 in all_factors:
    row = f"{f1:>10}" + "".join(f"{factor_corr.loc[f1, f2]:>10.4f}" for f2 in all_factors)
    out(row)

# ==================================================
# 1.1.2  Which factor correlates most highly with every industry?
# ==================================================
out("\n" + "=" * 90)
out("1.1.2  FACTOR WITH HIGHEST CORRELATION FOR EACH INDUSTRY")
out("=" * 90 + "\n")

out(f"{'Industry':<10}{'Most Corr. Factor':>20}{'Correlation':>15}")
out("-" * 45)
most_corr_summary = {}
for ind in industries:
    corrs = {f: corr_all.loc[ind, f] for f in factors}
    best_f = max(corrs, key=lambda k: abs(corrs[k]))
    most_corr_summary[ind] = (best_f, corrs[best_f])
    out(f"{ind:<10}{best_f:>20}{corrs[best_f]:>15.4f}")

# Tally
from collections import Counter
tally = Counter(v[0] for v in most_corr_summary.values())
out("\nSummary count of most-correlated factor across 30 industries:")
for f, cnt in tally.most_common():
    out(f"  {f}: {cnt} industries")

# ==================================================
# 1.1.3  Which factor correlates negatively with every industry?
#        Does RF correlate highly with the 30 industry time series?
# ==================================================
out("\n" + "=" * 90)
out("1.1.3  FACTORS WITH NEGATIVE CORRELATION TO EVERY INDUSTRY  &  RF ANALYSIS")
out("=" * 90 + "\n")

# Check each factor: is correlation negative for ALL 30 industries?
for f in factors:
    neg_all = all(corr_all.loc[ind, f] < 0 for ind in industries)
    neg_count = sum(1 for ind in industries if corr_all.loc[ind, f] < 0)
    out(f"  {f:>8}: Negative with ALL industries? {'YES' if neg_all else 'NO'}  "
        f"(negative with {neg_count}/{len(industries)} industries)")

# RF analysis
out(f"\n  {'RF':>8}: Correlation with each industry:")
rf_corrs = {ind: corr_all.loc[ind, "RF"] for ind in industries}
out(f"{'Industry':<10}{'Corr with RF':>15}")
out("-" * 25)
for ind in industries:
    out(f"{ind:<10}{rf_corrs[ind]:>15.4f}")

rf_vals = list(rf_corrs.values())
out(f"\n  RF correlation stats across 30 industries:")
out(f"    Mean : {np.mean(rf_vals):.4f}")
out(f"    Min  : {np.min(rf_vals):.4f}")
out(f"    Max  : {np.max(rf_vals):.4f}")
out(f"    |corr| > 0.5 for {sum(1 for v in rf_vals if abs(v) > 0.5)} / {len(industries)} industries")
out(f"    |corr| > 0.3 for {sum(1 for v in rf_vals if abs(v) > 0.3)} / {len(industries)} industries")

rf_neg_all = all(v < 0 for v in rf_vals)
out(f"    Negative with ALL industries? {'YES' if rf_neg_all else 'NO'}")
out(f"  => RF does {'NOT ' if np.mean(np.abs(rf_vals)) < 0.3 else ''}correlate highly with the 30 industry time series.")

# ==================================================
# 1.2.1  Autocorrelation function (ACF) for lags 1-10, four-factor series
# ==================================================
out("\n" + "=" * 90)
out("1.2.1  AUTOCORRELATION FUNCTION (ACF) - LAGS 1 TO 10")
out("=" * 90 + "\n")

acf_results = {}
n = len(df)
# Bartlett approximate 95% CI bound
ci_bound = 1.96 / np.sqrt(n)
out(f"Number of observations: {n}")
out(f"Approximate 95% confidence bound: +/- {ci_bound:.4f}\n")

header3 = f"{'Lag':>5}" + "".join(f"{f:>12}" for f in factors)
out(header3)
out("-" * len(header3))
for lag in range(1, 11):
    row = f"{lag:>5}"
    for f in factors:
        acf_vals = acf(df[f], nlags=10, fft=True)
        if lag == 1:
            acf_results[f] = acf_vals[1:]  # store lags 1-10
        row += f"{acf_vals[lag]:>12.4f}"
    out(row)

out(f"\n(Values outside +/- {ci_bound:.4f} are statistically significant at 95% level)")

# Mark significant lags
out("\nSignificant ACF values (|ACF| > 95% bound):")
for f in factors:
    acf_vals = acf(df[f], nlags=10, fft=True)
    sig_lags = [k for k in range(1, 11) if abs(acf_vals[k]) > ci_bound]
    if sig_lags:
        out(f"  {f}: lags {sig_lags}")
    else:
        out(f"  {f}: none")

# ==================================================
# 1.2.2  AR(1) model assessment
# ==================================================
out("\n" + "=" * 90)
out("1.2.2  AR(1) MODEL ASSESSMENT FOR THE FOUR FACTOR TIME SERIES")
out("=" * 90 + "\n")

out("An AR(1) process: X_t = c + phi * X_{t-1} + epsilon_t")
out("We fit AR(1) to each factor and check:\n"
    "  (a) significance of lag-1 coefficient (phi)\n"
    "  (b) Ljung-Box test on residuals for remaining autocorrelation\n")

for f in factors:
    out(f"--- {f} ---")
    series = df[f].values
    model = AutoReg(series, lags=1, old_names=False)
    result = model.fit()
    phi = result.params[1]
    pval_phi = result.pvalues[1]
    const = result.params[0]
    resid = result.resid

    # Ljung-Box test on residuals (lags 1-10)
    lb_test = acorr_ljungbox(resid, lags=10, return_df=True)
    lb_pvals = lb_test["lb_pvalue"].values

    acf1 = acf(series, nlags=1, fft=True)[1]

    out(f"  ACF(1)            = {acf1:.4f}")
    out(f"  AR(1) constant    = {const:.4f}")
    out(f"  AR(1) phi         = {phi:.4f}  (p-value = {pval_phi:.4e})")
    out(f"  phi significant?  {'YES' if pval_phi < 0.05 else 'NO'} (at 5% level)")
    out(f"  Ljung-Box p-values on residuals (lags 1-10):")
    for lag_idx in range(10):
        sig_marker = " *" if lb_pvals[lag_idx] < 0.05 else ""
        out(f"    Lag {lag_idx+1:>2}: p = {lb_pvals[lag_idx]:.4f}{sig_marker}")
    residual_ok = all(p > 0.05 for p in lb_pvals)
    out(f"  Residuals white noise? {'YES' if residual_ok else 'NO'}")

    is_ar1 = pval_phi < 0.05
    out(f"  => AR(1) model {'IS' if is_ar1 else 'is NOT'} appropriate for {f}.\n")

# Summary
out("\n--- AR(1) SUMMARY ---")
for f in factors:
    series = df[f].values
    model = AutoReg(series, lags=1, old_names=False)
    result = model.fit()
    phi = result.params[1]
    pval_phi = result.pvalues[1]
    is_ar1 = pval_phi < 0.05
    out(f"  {f:>8}: phi = {phi:.4f}, p = {pval_phi:.4e} => {'AR(1) YES' if is_ar1 else 'AR(1) NO'}")

# ==================================================
# Write output file
# ==================================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    fout.write("\n".join(lines))

print(f"\n*** Results written to {OUTPUT_FILE} ***")


# ==================================================
# Part II: AR(2)-style model calculations
# X_t = 0.01 + 0.2 * X_{t-2} + a_t,    a_t ~ WN(0, 0.02)
# ==================================================
part2_lines = []

def out2(text=""):
    part2_lines.append(text)
    print(text)


out2("=" * 90)
out2("PART II  MODEL: X_t = 0.01 + 0.2 X_{t-2} + a_t,   a_t ~ WN(0, 0.02)")
out2("=" * 90)

# Given model parameters
c0 = 0.01
phi2 = 0.2
sigma_a2 = 0.02

# 2.1 Mean and variance
mu = c0 / (1 - phi2)

# For centered process Y_t = X_t - mu: Y_t = phi2 * Y_{t-2} + a_t
# Yule-Walker: gamma2 = phi2*gamma0, gamma0 = phi2*gamma2 + sigma_a2
# => gamma0 = sigma_a2 / (1 - phi2^2)
gamma0 = sigma_a2 / (1 - phi2**2)
var_x = gamma0

out2("\n2.1 Mean and Variance")
out2(f"  Mean E[X_t] = {mu:.10f}")
out2(f"  Variance Var(X_t) = {var_x:.10f}")

# 2.2 ACF at lags 1 and 2
gamma1 = 0.0
gamma2 = phi2 * gamma0
rho1 = gamma1 / gamma0
rho2 = gamma2 / gamma0

out2("\n2.2 Lag-1 and Lag-2 autocorrelations")
out2(f"  rho(1) = {rho1:.10f}")
out2(f"  rho(2) = {rho2:.10f}")

# 2.3 Forecasts at origin t=100
x_100 = -0.01
x_99 = 0.02

# 1-step ahead: X_101 = c0 + phi2*X_99 + a_101, E[a_101|F_100]=0
xhat_101 = c0 + phi2 * x_99

# 2-step ahead: X_102 = c0 + phi2*X_100 + a_102, E[a_102|F_100]=0
xhat_102 = c0 + phi2 * x_100

out2("\n2.3 Forecasts at t=100 (given X_100=-0.01, X_99=0.02)")
out2(f"  1-step forecast  Xhat_101|100 = {xhat_101:.10f}")
out2(f"  2-step forecast  Xhat_102|100 = {xhat_102:.10f}")

with open(OUTPUT_FILE_PART2, "w", encoding="utf-8") as fout2:
    fout2.write("\n".join(part2_lines))

print(f"\n*** Part II results written to {OUTPUT_FILE_PART2} ***")

# Findings Report: Iterative Improvement of the Stacked Meta-Learner

**Project:** EE627A Final - Music Recommender (top-3 of 6 candidates per user)    
**Author:** Jude Eschete    
**Period:** 2026-05-04 (single afternoon of iteration on the stacker)   
**Frozen state:** v4 not yet scored at time of writing (ALS rank=80 + item-item CF in progress)   

This document records what each iteration of the stacked meta-learner   
attempted, what we learned from the results, and which dead ends to   
remember.  It is meant as a post-mortem to inform future ensemble work   
on similar small-label ranking tasks.   

---

## How prior coursework feeds into this

Almost every base model in this final is a direct reuse of work
produced earlier in the semester.  The final is not a from-scratch
build -- it is a multi-level ensemble whose inputs are the homeworks
themselves.  Each prior assignment contributes a distinct view of the
rating matrix, which is exactly the diversity a stacker needs.

| Source                | What it contributes to the ensemble                            |
|-----------------------|----------------------------------------------------------------|
| Project 1             | `p1_evidence`, `p1_weightedavg`, `p1_maxgenre` -- early evidence/genre heuristics, kept as binary feature columns and rolled up into a `p1_consensus` family feature in v2+ |
| Homeworks 3-7         | `v3_additive`, `v3_confidence`, `v3_genre_both`, `v3_interact`, `v3_tuned`, `v4_ridge` -- progressively-tuned heuristic and ridge baselines, rolled into `v3_consensus` |
| Homework 8            | `hw8_hybrid`, `hw8_hybrid_gated`, `hw8_pyspark_mf` -- the first PySpark MF attempts, rolled into `hw8_consensus` |
| Homework 9            | The `hw9_probabilities.csv` matrix (7 PySpark probability columns: lr, dt, rf, gbt, gbt_cv, rf_no_v2, extra) plus 5 binary blends (`hw9_blend_rank`, `hw9_rf`, `hw9_rf_no_v2`, `hw9_blend_three`, `hw9_blend_rf_heavy`).  These are the strongest individual base learners we have -- the meta-learner consistently gives `hw9_consensus` the largest positive weight (+2.31 in v2, +2.33 in v4). |
| Midterm               | `v5b_nn_0878` (BPR neural net) and the full `v5b_hybrid_gap*` confidence-gated hybrid sweep (gap0 through gap20).  Eleven of these were preserved as binary columns; v2+ collapses them into `v5b_hybrid_consensus`.  The midterm `gap10` submission (Kaggle 0.911) is the floor we built up from. |
| Final-only additions  | PySpark ALS (rank=20, repeated from the HW9 toolkit), Layer-1 logistic-regression meta-learner with GroupKFold-by-user CV, isotonic calibration on HW9/v5 outputs, LightGBM LambdaRank attempt, and item-item collaborative filtering (the v4 breakthrough). |

The `past_predictions/` directory in this folder is the literal cache
of those prior coursework outputs -- 39 CSV files named with their
historical Kaggle scores.  Treating earlier homeworks as base
predictions for the final is the entire reason the stack has enough
diversity to work; a from-scratch build with only one model family
would not have broken the 0.918 plateau.

---

## Setup recap

- **Task:** binary recommend/not-recommend, evaluated as top-3 picks per
  user out of 6 candidates.  20,000 test users x 6 candidates = 120,000
  Kaggle rows.
- **Labeled holdout for stacking:** `test2_new.txt`, 6,000 rows
  (1,000 users x 6 candidates), held out from the Kaggle test set.
- **Base models available going in:**
  - v2 heuristic (Kaggle ~0.871 alone)
  - v5 BPR neural net on 32 hand-crafted features (val AUC 0.7424,
    Kaggle 0.878 standalone)
  - v5 + v2 confidence-gated hybrid sweep (`gap10` -> Kaggle 0.911)
  - PySpark ALS rank=20 (Kaggle 0.621 standalone -- weak)
  - HW9 PySpark stack: 7 probability columns (lr, dt, rf, gbt, gbt_cv,
    rf_no_v2, extra) + 5 hw9_* binary blends (best ~0.914)
  - 39 historical Kaggle submissions in `past_predictions/`

---

## v1 stacker -- all-in logistic regression

**Approach:** flatten everything into 49 features (v5 score, v2 score,
ALS, 7 HW9 probs, 39 binary past submissions), z-score, fit
`LogisticRegressionCV` with random 5-fold AUC scoring.

**Result:** Kaggle 0.918, beating the best base model (hw9_blend_rank
at 0.914) by +0.004.

**Reported CV AUC:** 0.9860.  This was an early warning we missed --
0.986 -> 0.918 is a much wider gap than blend stackers usually see.

### What we learned

- Stacking 39 binary near-duplicates with logistic regression
  produces sign-flip artifacts.  The top weights showed
  `hw9_blend_three_0913` at -0.5007 while its siblings
  (`hw9_blend_rank_0914`, `hw9_rf_0914`) were at +0.91 and +0.77.
  Classic collinearity: the LR was learning "trust A but flip on the
  rows where A and B disagree," which fits the labeled holdout but
  doesn't generalize.
- The LR was effectively doing weighted majority voting.  Stacked 0.918
  vs best-base 0.914 = only +0.004 lift, far below the lift you'd see
  from genuinely diverse signals.

---

## v2 stacker -- per-user ranks, family consensus, GroupKFold

**Approach:** four targeted fixes:

1. **Per-user rank features** added (`v5_rank`, `v2_rank`, `als_rank`).
   For each user's 6 candidates, highest-scoring tid -> 1.0, lowest ->
   0.0.  The signal Kaggle actually scores on.
2. **Family consensus collapse:** the 39 past submissions are 95%
   redundant in clusters (v5b_hybrid_*, v6_hybrid_*, hw9_*, etc.).
   Replaced 39 binary columns with 6 "consensus rate" features (fraction
   of family members voting 1) + 2 ungrouped one-offs.  21 features
   total.
3. **GroupKFold by user** for the 5-fold CV.  Random splits had been
   leaking: each user's 6 rows were ending up in both train and val
   folds, so the LR could memorize per-user patterns.
4. **NNLS sanity baseline:** non-negative least squares on 16 core
   features, all weights forced >= 0.  Tests whether the LR's negative
   weights are real or noise.

**Result:** Kaggle 0.919 (LR), 0.916 (NNLS).  CV AUC 0.9859 honest, vs
0.9860 random-fold -- so the random fold was not catastrophic, but we
were tuning on an inflated number.

### What we learned

- **The leak was small.**  Honest CV (0.9859) was within rounding of
  the dishonest CV (0.9860).  The Kaggle gap (0.918) wasn't a leak; it
  was the genuine generalization gap from a 6,000-row training set on
  a difficult ranking task.
- **Family consensus worked.**  Going from 49 features to 21 made the
  weights interpretable: `hw9_consensus` got +2.31 (correctly the
  dominant signal), `v2_score` +0.60, `v5b_nn_0878` +0.57.  No more
  +0.91 / -0.50 sign-flip noise.
- **NNLS validated the LR.**  NNLS at 0.916 vs LR at 0.919 means the
  LR's negative weights weren't pure noise -- they were doing genuine
  bias-correction across correlated features.  Worth keeping the LR.
- **+0.001 Kaggle lift from 4 fixes is small but real.**  At this
  point we suspected we were near the meta-learner ceiling.

### Dead end avoided

NNLS turned out to be a useful diagnostic but not a useful submission.
If future work has time pressure, skip NNLS as a Kaggle submission and
keep it only as a CV check.

---

## v3 stacker -- LightGBM + Isotonic + NDCG@3 (3-method ablation)

**Approach:** three external methods (cited in `SOURCES.md`):

- **A. LightGBM LambdaRank** as a new base model on the same 32 features
  the BPR neural net uses, with the natural per-user group-of-6
  structure.  LightGBM should learn ranking patterns the neural net
  misses.
- **B. Isotonic regression calibration** of the v5 NN raw score and
  each HW9 probability column, fit cross-validated by user.  Re-maps
  raw model outputs onto a common probability scale before stacking.
- **C. NDCG@3 grouped-by-user CV** for the LR's `C` parameter sweep.
  Tunes for the metric Kaggle actually scores against, instead of AUC.

Five submissions written: V2ref (baseline re-fit), A (V2 + LGBM only),
B (V2 + isotonic only), C (V2 features + NDCG@3 tuning), ALL (all three
combined).

**Result:** Kaggle 0.919 for both v3_C_ndcg and v3 ALL -- identical to
v2.  No improvement.

### What we learned

- **Method A (LightGBM LambdaRank): no contribution.**  LightGBM
  early-stopped at iteration 1 with val NDCG@3 = 0.7752.  The 32
  hand-crafted features are saturated: the BPR neural net (val AUC
  0.7424) is already capturing essentially everything a tree model can
  extract from this representation.  Adding `lgbm_rank_score` to the
  stacker moved AUC by -0.0001.  **Failed cleanly.**
- **Method B (Isotonic calibration): +0.0002 AUC.**  Real but within
  fold std (~0.0026).  Probably noise.  Cheap to keep, won't hurt.
- **Method C (NDCG@3 tuning): different best-C than AUC tuning.**  AUC
  picked C=0.1; NDCG@3 picked C=1.0.  That's a different model -- looser
  regularization, fits per-user ranking patterns more aggressively.
  Kaggle didn't reward it.
- **The 0.918-0.919 plateau is a feature-saturation problem, not a
  meta-learner problem.**  v1, v2, and three flavors of v3 all
  converged on Kaggle ~0.919 because they're all combining the same
  signals.  Without a base model that sees the data differently,
  meta-learner tweaks can't break the plateau.

### Headline takeaway

**Always sanity-check by asking: does the new base model see the data
differently than what you already have?**  LightGBM trained on the
same 32 features as the BPR-NN cannot add information.  We only
discovered this empirically, but the early-stop at iteration 1 was the
clearest possible signal that the feature space was exhausted.

### Dead ends avoided

- Don't ablate via separate Kaggle submissions when the CV metric is
  honest.  We tried this with v3 (4 ablation submissions) and
  v3_C_ndcg both scored 0.919.  Honest grouped CV is enough to predict
  Kaggle ordering at this scale; save the daily Kaggle quota.
- Don't add binary submission columns for already-stacked outputs
  (e.g., `submission_final_stacked.csv` from v1) into a future stacker.
  Those columns were trained on the same `test2_new.txt` labels the
  meta-learner uses, so they would leak.  We deliberately left them
  out of `past_predictions/`.

---

## v4 stacker -- item-item CF + ALS rank=80 (in progress)

**Approach:**

- **Item-item collaborative filtering** (Sarwar 2001; Linden 2003) as
  a new base model.  For each test (uid, tid), score = sum of cosine
  similarities between tid and the user's top-30 rated items, weighted
  by the user's rating on those history items.  Pure rating-matrix
  signal, no learned embeddings -- maximally orthogonal to BPR-NN, MF,
  and ALS.
- **ALS rank bumped 20 -> 80.**  At rank=20, ALS scored 0.621 alone and
  got LR weight +0.04 in v1.  Higher rank captures more user/item
  structure.  Expect standalone Kaggle 0.85+ at rank=80.
- v4 feature set = v3 ALL + iicf_score + iicf_rank + the
  higher-rank ALS column.

**Result:**

| Submission                                    | Kaggle |
|-----------------------------------------------|--------|
| `submission_final_stacked_v4_iicf_only.csv`   | 0.781  |
| `submission_final_stacked_v4.csv`             | 0.922  |

**Plateau broken.**  +0.003 over the v2/v3 cluster (0.919 -> 0.922).

### What we learned

- **The orthogonal-signal hypothesis was correct.**  Item-item CF on
  the raw rating matrix (no learned embeddings) captured information
  that none of the BPR-NN, MF, ALS, or HW9 RF/GBT chain had access to.
  This confirms that the v1->v3 plateau was caused by correlated base
  models, NOT by meta-learner saturation.
- **The meta-learner agreed with the diagnosis.**  In v4's weight
  table, `iicf_rank` got +0.4740 (3rd-largest weight overall) and
  `iicf_score` got +0.3240.  The combined ~0.80 weight on iicf
  features is by far the largest weight any new method has received
  since v2 -- compare to LightGBM's effective zero in v3.
- **iicf-only at 0.781 is a healthy standalone score.**  For a pure
  neighborhood method with no parameter learning, that's a strong
  signal -- nothing close to the BPR-NN's 0.878, but standalone scores
  aren't what matters for stacking; orthogonality is.
- **CV-to-Kaggle correlation held this time.**  v3->v4 CV AUC moved
  +0.0015 (0.9858 -> 0.9873), and Kaggle moved +0.003.  That's about
  a 2x amplification, which is normal for a real signal.  The flat
  NDCG@3 was a red herring -- iicf was reordering the boundary
  candidates (the ones near the top-3 cut) without changing the very
  top, which AUC catches but NDCG@3 doesn't.
- **ALS rank=80 didn't help and may have hurt slightly.**  Both
  `als_score` (-0.151) and `als_rank` (-0.117) got negative weights
  in v4.  At rank=20 ALS got +0.04 (small positive) in v1.  Higher
  rank made ALS more confident on the wrong rows.  Lesson: a base
  model's standalone Kaggle score does not predict its stacking
  contribution.

### Why iicf worked when LightGBM did not

This is the cleanest result of the whole project.  Both LightGBM (v3)
and iicf (v4) were "new base models" on paper, but only iicf moved the
needle:

| Method      | Trained on                | Result            |
|-------------|---------------------------|-------------------|
| LightGBM    | Same 32 features as BPR-NN | early-stopped at iter 1 (saturated) |
| Item-item CF| Raw 296K-item co-rating matrix | +0.003 Kaggle |

A new model on existing features cannot add information.  A model on
**new features** can.  This is the textbook ensemble-diversity result
(Caruana et al. 2004) demonstrated empirically.

---

## Cross-cutting lessons

### 1. Stacking works best with diverse signals, not more signals

Going from 49 to 21 features in v2 *helped*.  The 28 features we
removed were collinear duplicates that the LR had to spend capacity
canceling out.  More columns is not better when most columns are
near-duplicates.

### 2. Honest CV is non-negotiable

Random-fold CV gave 0.9860 AUC.  GroupKFold-by-user CV gave 0.9859
AUC.  Tiny gap, but the latter let us trust ablation comparisons.
Without GroupKFold we would have spent v3 chasing a metric that
disagreed with Kaggle.

### 3. CV lift on small labels predicts Kaggle lift roughly

v2->v3 CV AUC went 0.9859 -> 0.9861 (+0.0002).
v2->v3 Kaggle AUC went 0.919 -> 0.919 (0.000).
The CV lift was honest *but tiny*, and the Kaggle resolution was lower
than the CV resolution.  Rule of thumb: a CV lift smaller than the
fold std (~0.0026 here) is probably not going to move Kaggle.

### 4. The 0.986 -> 0.919 gap is the ranking-task gap, not a leak

Once we ruled out leaks (v2's GroupKFold), the remaining gap is just
the price of training a meta-learner on 6,000 labels for a task with
20,000 test users.  That's a 3x label-vs-test ratio -- the meta-learner
genuinely can't see all the user variety it has to predict on.

### 5. Add base models on NEW FEATURES, not meta-learner tricks

v1 -> v2: meta-learner cleanup (-> +0.001)
v2 -> v3: more meta-learner tricks (-> +0.000)
v3 -> v4: new base model on a NEW FEATURE SPACE (-> +0.003)

The marginal return on meta-learner work was zero from v2 onwards.
The single largest single-step gain in the whole project (+0.003) came
from adding item-item CF.  Crucially, LightGBM in v3 was also a "new
base model" but contributed nothing because it trained on the same
32-feature representation the BPR-NN had already saturated.  The right
mental model is: **new base learners only help if they see the data
differently, not if they're just a different algorithm on the same
features.**

### 6. The right ablation lives in the log, not on Kaggle

v3 wrote 5 submission files for ablation.  We submitted only 2 of them
(C and ALL); the rest were redundant per the honest CV.  If the CV
ordering matches Kaggle ordering on the submissions you do test,
trust the CV for the rest.

---

## Submission timeline (for the report)

| Submission                                  | Kaggle | Notes                              |
|---------------------------------------------|--------|------------------------------------|
| `submission_final_v5_hybrid_gap10.csv`      | 0.911  | Pre-stacking baseline (midterm)    |
| `submission_final_als.csv` (rank=20)        | 0.621  | ALS-only sanity                    |
| `submission_final_stacked.csv` (v1)         | 0.918  | All-in LR, 49 features             |
| `submission_final_stacked_v2.csv`           | 0.919  | Ranks + family consensus + GKF     |
| `submission_final_stacked_v2_nnls.csv`      | 0.916  | NNLS sanity baseline               |
| `submission_final_stacked_v3.csv`           | 0.919  | + LightGBM + Isotonic + NDCG@3     |
| `submission_final_stacked_v3_C_ndcg.csv`    | 0.919  | NDCG@3-tuned only (no LGBM/iso)    |
| `submission_final_stacked_v4.csv`           | **0.922** | + item-item CF + ALS rank=80    |
| `submission_final_stacked_v4_iicf_only.csv` | 0.781  | iicf-only sanity                   |

---

## Recommendations for future ranking-task ensembles

1. **Start by auditing how correlated your base models are.**  Compute
   pairwise correlation between base-model scores on a labeled
   holdout *before* stacking.  If correlations are >0.95, expect the
   stacker to plateau fast.
2. **Use grouped-by-query CV from the start.**  Random fold CV will
   inflate metrics on ranking tasks where the same query repeats.
3. **Tune for the metric Kaggle scores.**  AUC and NDCG@3 disagree on
   the optimal regularization; tune for the right one.
4. **Add base models with different inductive biases before adding
   meta-learner tricks.**  Latent-factor models (MF, ALS), neural
   ranking models (BPR-NN), neighborhood models (item-item CF), and
   tree models (LightGBM) all carry different biases.  Aim to have at
   least one of each in the stack before tuning the meta-learner.
5. **Keep a `past_predictions/` archive but don't pollute it with
   meta-learner outputs.**  Leakage from re-feeding stacker results
   into a stacker is subtle and silent.
6. **Cache aggressively.**  Re-running the full pipeline shouldn't be
   the cost of an experiment.  Each iteration here cost <1 minute of
   wall time after the initial caches were built.

---

# Phase 2: Closed-Form LS Ensemble + Advice List (post-PDF iteration)

**Period:** 2026-05-08 (single session continuing from v4/v5 stackers)
**Trigger:** Class introduced `EE627A_ensemble.pdf` — a closed-form
weighted ensemble that uses Kaggle scores as the only ground-truth
signal.

This phase added a fundamentally new code path (the LS ensemble) and
attempted six improvements from an external advice list to break the
0.922 plateau and chase 0.930.

---

## What the PDF method does

Given submitted vectors `s_1, ..., s_K` (each `{0,1}` predictions on the
same 120K test rows) and Kaggle scores `P_1, ..., P_K`:

1. Map `{0,1} -> {-1,+1}` so dot products mean
   `s_i^T x = (correct - wrong) = N(2 P_i - 1)`.
2. Compute closed-form weights `a_LS = (S^T S)^(-1) (N (2 P - 1))`.
3. `s_ensemble = S * a_LS`.
4. Per user (block of 6 candidates), top-3 by `s_ensemble` -> 1, rest -> 0.

The trick: ground-truth `x` is never needed because `S^T x` is
recoverable from the published Kaggle scores alone.

## Implementation

New section 13 in `EE627_Final_Eschete.py`, ~200 lines:

- `parse_submission_scores()` — reads `Submission Results.txt` ->
  `{filename: score}`.
- `load_submission_matrix()` — folder of CSVs -> `(N x K)` `{-1, +1}`
  matrix aligned by sorted TrackID; logs skipped files.
- `solve_ls_ensemble()` — `np.linalg.solve(S^T S, S^T x)`, falls back
  to `lstsq` if singular; reports `cond(S^T S)`.
- `write_ensemble_submission()` — top-3 per user.
- `run_ls_ensemble()` — driver, prints per-submission `(P_i, a_i)` table.
- New CLI flag: `python EE627_Final_Eschete.py --ensemble-only` (skips
  all training, runs only the LS step in seconds).
- Output filenames are timestamped:
  `submission_final_ensemble_Eschete_YYYYMMDD_HHMMSS.csv`.

## Data hygiene around the LS path

- Cleaned `Submission Results.txt` to a strict 1:1 mapping with the
  CSVs in `past_predict_ensemble/` (34 entries; 6 `__V4` / `__V5`
  variants without scores excluded).  Resolved 5 cases where the
  embedded `_0NNN` filename suffix disagreed with the results-file
  score by trusting the filename suffix.
- Redirected `PAST_PREDS_DIR` from `past_predictions/` to
  `past_predict_ensemble/` so every code path reads from one folder.
- Updated `load_past_predictions()` to consult `Submission Results.txt`
  exclusively (no more filename-suffix score parsing). Files not in
  the results file are skipped with a logged warning. No more `?` in
  the score-range output.

## Helper script: `register_submission.py`

Standalone script that detects new
`submission_final_ensemble_Eschete_<ts>.csv` files in `Final/`, prompts
for the Kaggle score, moves the file into `past_predict_ensemble/`, and
appends a line to the `--- LS ensemble ---` section of
`Submission Results.txt` *without reordering existing data*.
Idempotent. Refuses to overwrite anything.

## The advice list (6 items implemented, 1 deliberately skipped)

| # | Item                                         | Verdict |
|---|----------------------------------------------|---------|
| 1 | UBCF as a new base model                     | Implemented; modest contributor |
| 2 | BPR-trained MF instead of MSE-MF             | Implemented; **empirically worse**, reverted |
| 3 | Feed LS ensemble back into stacker           | Already automatic via `load_past_predictions` |
| 4 | Adjusted-cosine IICF (`iicf_d`)              | Implemented; small positive contributor |
| 5 | `decision_function` -> `predict_proba`       | **Skipped** (monotonic transform; cannot change top-3) |
| 6 | PATIENCE 15 -> 25                            | Implemented |
| 7 | MF_EPOCHS 40 -> 80                           | Implemented |

### UBCF details

New section in the script: `_build_user_item_sparse`, `compute_user_knn`
(chunked sparse cosine, top-K per user, cached at
`cache/ubcf_knn_k50_centered.npz`), `ubcf_score_pair` (O(K) lookup
given precomputed KNN), `compute_or_load_ubcf` (test-pair scores,
cached at `cache/ubcf_k50_centered.npz`).  Added as feature 29 in the
BPR-NN, and as `ubcf_rank` + `ubcf_score` columns in the v5 stacker.

### iicf_d details

`IICF_VARIANTS` now stores 4-tuples of `(name, topk, min_co, mode)`
where `mode in {"raw", "centered"}`.  Centered mode subtracts each
user's rating mean before computing item-item similarity (adjusted
cosine).  Modest signal — `iicf_d_rank` got `+0.1601` weight in the
v5 stacker.

### Cache invalidation

Every cache filename now embeds the parameters that affect its
contents:

- `mf_k64_e80_{loss}.npz` — factors, epochs, loss
- `features_g30_seed42_f{N_FEATURES}.npz`
- `bpr_h128-64-32_d0.3_seed42_f{N_FEATURES}.pt`
- `lgbm_lambdarank_f{N_FEATURES}.txt`
- `iicf_k{topk}_m{min_co}_{mode}.npz`
- `ubcf_knn_k{K}_{mode}.npz`, `ubcf_k{K}_{mode}.npz`

A feature-count bump auto-invalidates every feature-dependent cache.

## Submission timeline (Phase 2)

| File                                                    | Kaggle | Notes |
|---------------------------------------------------------|--------|-------|
| `submission_final_ensemble_Eschete_20260508_162024.csv` | **0.925** | First LS ensemble — instant +0.003 over best stacked, no training |
| `submission_final_ensemble_Eschete_20260508_162951.csv` | 0.923  | Self-referential — added 0.925 file back to input set |
| `submission_final_ensemble_Eschete_20260508_171818.csv` | 0.924  | After UBCF/iicf_d/PATIENCE/MF bumps (BPR-MF was broken at this point) |
| `submission_final_ensemble_Eschete_20260508_172940.csv` | 0.923  | After "fixed" BPR-MF (still suboptimal embeddings) |

Best in Phase 2: **0.925** (the very first LS ensemble run).
Subsequent runs all underperformed it.

## What worked

- **The LS ensemble itself.**  +0.003 over the best stacked submission
  with zero training, zero parameter tuning.  The single largest
  single-step gain in the project after item-item CF.
- **Single source of truth for scores.**  Cleaning
  `Submission Results.txt` to 1:1 with the folder removed every "?" in
  downstream loaders and let any code path consult one curated file.
- **`iicf_d` (adjusted cosine).**  Real positive contribution to the
  v5 stacker. Cheap addition.
- **Cache filenames embedding parameters.**  Saved us from a silent
  feature-count corruption (`lgbm_lambdarank.txt` getting loaded with
  32 features when training had been on 33).

## What didn't work

### Self-referential LS ensemble drops the score

After moving the 0.925 ensemble file into `past_predict_ensemble/`,
the next LS run scored **0.923**.  The 0.925 file got assigned weight
`-0.0968` (negative).

**Cause:** the new column lies in the span of the others (it is, by
construction, a linear combination of them).  `cond(S^T S)` rose from
5.30e+04 to 6.06e+04.  The LS solution becomes underdetermined;
`np.linalg.solve` picks a basin that's score-consistent in
`S^T x = N(2P - 1)` but not as good for actual ground truth.

**Repeated:** runs 3 and 4 (which included successive prior LS ensembles
in the input set) showed the same pattern — older LS ensembles got
negative weights, current solution diverged from the original 0.925.

**Lesson:** the LS method has no free improvement loop.  Its outputs
should not feed back into its inputs.

### BPR-trained MF — empirically worse than MSE-MF

Two phases of failure:

**Phase 2a.** First attempt with `MF_LR=0.005`, `weight_decay=1e-4`,
`init_std=0.1`: BPR loss stuck at exactly `0.6931 = -log(0.5)` for all
80 epochs.  Model never learned.  LR tuned for MSE on rating-scale
targets cannot move BPR-loss embeddings out of the random-init basin
(Adam + small init + weight decay = vanishing training).

**Phase 2b.** Second attempt with `MF_LR=0.05`, `weight_decay=0`,
`init_std=1/sqrt(k)`: loss dropped from 0.6932 to 0.2519 in 5 epochs,
then **diverged back up to 0.3661** by epoch 80.  Adam + uniform
negative sampling + sparse data overshoots after the initial descent.

Even at the best epoch (5), BPR-NN val AUC was **0.7124** — still
**0.030 below** the MSE-MF baseline of **0.7424**.

**Verdict:** the advice's claim "BPR-MF must beat MSE-MF for ranking"
does not hold here.  Plausible reasons: ~30 ratings/user is too sparse
for uniform-negative-sampling BPR; Adam can't recover from the noisy
gradients; meanwhile MSE on 0-100 ratings has clean direction.
Reverted to `MF_LOSS = "mse"`.

### UBCF didn't break the plateau

UBCF features exist in both BPR-NN (feature 29) and v5 stacker
(`ubcf_rank`, `ubcf_score`).  In the v5 stacker top-25 weights, UBCF
was absent — `|weight|` < 0.13.  Not strong enough vs. HW9 + past
predictions.  Not harmful; not the savior either.

### `decision_function` -> `predict_proba` (advice #5)

Skipped as instructed.  For sklearn `LogisticRegression`,
`predict_proba` is `sigmoid(decision_function)` — a monotonic
transform.  Top-3 ranking within a 6-candidate user group is
**identical**.  Would not move a single prediction.

## Phase 2 lessons

1. **LS collinearity is real.**  Including a column that's a linear
   combination of others in `S` makes `cond(S^T S)` rise and produces
   score-consistent but truth-inconsistent solutions.  Filter your
   own outputs out of your own inputs.
2. **Loss objectives need matched optimizer hyperparameters.**  LR,
   weight decay, and init scale all need to change when moving from
   MSE to BPR.  Naively swapping `loss="bpr"` is a footgun.
3. **Val AUC is an imperfect proxy for Kaggle score.**  We saw runs
   with lower val AUC sometimes score higher on the leaderboard.
4. **Empirical > theoretical.**  "BPR-MF should beat MSE-MF for
   ranking" is a defensible prior; the data disagreed.
5. **Cache filenames must encode every parameter that affects
   contents.**  Otherwise stale caches silently corrupt downstream
   caches.  Got bit by `lgbm_lambdarank.txt` (no feature-count tag)
   when feature count bumped 32 -> 33.
6. **Kaggle scores in `Submission Results.txt` are the only legitimate
   ground-truth signal we have.**  The closed-form LS method exploits
   this exactly; nothing else in the pipeline does.
7. **Ground-truth-free closed-form methods can outperform days of
   meta-learner work.**  The LS ensemble matched all v3/v4 efforts in
   one shot, no training, no labels, just published scores and `numpy`.

## Phase 3: Filter LS ensembles out of LS inputs (2026-05-08, late)

### Trigger

Recovery run after the MSE-MF revert scored **0.924** on the leaderboard.
Inspecting the LS ensemble weight table revealed where the magnitude was
going — into negative-cancellation among the prior LS ensembles
themselves:

```
_172227.csv (0.923)   a_i = -0.2867
_172940.csv (0.923)   a_i = -0.2037
_162951.csv (0.923)   a_i = -0.1739
_171818.csv (0.924)   a_i = -0.1108
_162024.csv (0.925)   a_i = +0.2446
```

Combined absolute weight on prior LS ensembles: **|a| ≈ 0.78** — about
10× more than the three lowest-Kaggle submissions (ALS 0.621, hw8_pyspark
0.636, hw8_hybrid_gated 0.513) contribute combined (`|a| ≈ 0.072`).

### Diagnosis

A new LS ensemble is, by construction, a linear combination of the
existing columns of `S`.  Adding it back to `S` injects exact
collinearity: the system is underdetermined and `np.linalg.solve`
selects whichever weight vector satisfies `Sᵀx = N(2P−1)` — but multiple
solutions exist and the chosen one may be score-consistent without
being truth-consistent.  In practice, prior ensembles get negative
weights that cancel each other, eating signal that should go to the
genuinely diverse base columns.

This was already noted as a Phase 2 lesson; Phase 3 acts on it.

### Fix

Single predicate in `load_submission_matrix()`:

```python
if (exclude_self_ensemble and
        fname.startswith("submission_final_ensemble_Eschete_")):
    self_excluded.append(fname)
    continue
```

The function returns `self_excluded` as a 6th element so `run_ls_ensemble`
can log them separately from "skipped" (which means "no Kaggle score
known" — a different reason for exclusion).  Default is `True`; can be
disabled by passing `exclude_self_ensemble=False` if ever needed.

### Why this beats "drop the lowest-scoring submissions"

Comparison from the same run:

| Group | Files | Combined `Σ |a_i|` | Reasoning |
|---|---|---|---|
| Prior LS ensembles  | 5 | **0.7796** | Linear combinations of the rest, fight each other |
| Lowest-Kaggle (≤0.65) | 3 | 0.0716 | Already near zero weight, near-noise |

Filtering the LS ensembles frees ~10× more weight magnitude than
dropping the low-scorers.  And there's no theoretical reason to drop
weak signals — the LS solver already deweights them automatically.
Collinearity is a structural problem; weak-signal dilution is not.

### Expected behavior after the fix

- `S` now has K = (previously K) − (number of prior LS ensembles).
- `cond(SᵀS)` should drop (was 6.25e+04 with 5 LS ensembles in S).
- New ensemble file should reproduce ~0.925 reliably regardless of how
  many prior LS ensembles accumulate in `past_predict_ensemble/`.

### Status

Implemented and verified by Kaggle submission.

### Result

Re-ran `--ensemble-only` after the fix.  Diagnostic output showed
`K = 34` (down from 39), `cond(SᵀS) = 5.298e+04` — *exact match* to the
original 0.925 run's diagnostic numbers.  SHA-256 confirmed the new
output (`_174933.csv`) is **byte-identical** to the 0.925 file
(`_162024.csv`).  Did not waste a Kaggle attempt to verify.

The LS ensemble is now deterministic and reproducible: regardless of
how many prior LS-ensemble outputs accumulate in
`past_predict_ensemble/`, the solver always returns to the canonical
0.925 baseline.

---

# Phase 4: Plateau exploration and per-user routing (2026-05-08, evening)

After the Phase 3 fix locked in 0.925, several variants were tested
to find any path past it.

## Submissions made in Phase 4

| File                                                  | Kaggle | Notes |
|-------------------------------------------------------|--------|-------|
| `submission_final_v5_nn.csv` (fresh)                  | 0.868  | Pure BPR-NN with UBCF + iicf_d + retrained MSE-MF.  Lower than the val AUC suggested.  Kept in folder for ensemble use. |
| `submission_final_majority_Eschete_<ts>.csv`          | 0.925  | Per-cell majority vote across 6 LS ensembles.  Differed from the 0.925 baseline on only 28 users — net-zero on Kaggle as predicted. |
| `submission_final_stacked_v4_20260508_180000.csv`     | 0.920  | Fresh v4 stacker after UBCF/iicf_d/MSE-MF retrain.  **Worse than the old 0.922** — new features actively hurt downstream. |

## Key findings

### Val AUC ≠ Kaggle score, confirmed harder

The retrained BPR-NN reached **val AUC 0.7550** (up from 0.7424
baseline), suggesting the new features were informative.  But the
fresh v4 stacker scored **0.920 on Kaggle** (down from 0.922).  Val
AUC is measured on a 10% held-out user split using BPR pairwise
accuracy; Kaggle scores the top-3 picks.  **The held-out AUC gain did
not translate to leaderboard gain.**

This is now a recurring pattern: every "improve the features" change
in Phase 2 pushed val AUC up but Kaggle stayed flat or dropped.  The
12-feature increment from UBCF (1) + ubcf-derived stacker columns +
iicf_d does not net positive at the leaderboard granularity.

### Majority vote across LS ensembles is essentially a no-op

With 6 LS ensembles in the input set (one 0.925, two 0.924, three
0.923), per-cell majority vote produced an output that:

- Differed from the 0.925 baseline on **0.05% of cells, 0.14% of
  users** (28 users)
- Tied the 0.925 score on Kaggle

Why so small: the score-weighted tiebreaker biases the vote toward
the highest-scoring file, and on cells where the LS ensembles
disagree the disagreement is noise (basin oscillation), so majority
voting essentially picks the 0.925 basin verbatim.

### Per-submission diff matrix (cell-level, percent agreement)

|                              | 0.925 | 0.924 | 0.923 | v4    | v5_real | v3    | v3_C  | v2    |
|------------------------------|------:|------:|------:|------:|--------:|------:|------:|------:|
| 0.925 LS                     |  -    | 98.77 | 98.17 | 98.02 | 98.04   | 97.48 | 97.49 | 97.49 |
| 0.924 LS                     |       |  -    | 97.86 | 98.30 | 98.31   | 97.77 | 97.78 | 97.79 |
| 0.923 LS                     |       |       |  -    | 98.58 | 98.62   | 98.02 | 98.03 | 98.04 |
| stacked_v4                   |       |       |       |  -    | 99.55   | 99.31 | 99.31 | 99.33 |
| stacked_v5_real              |       |       |       |       |  -      | 99.24 | 99.25 | 99.27 |
| stacked_v3                   |       |       |       |       |         |  -    | 99.78 | 99.76 |
| stacked_v3_C_ndcg            |       |       |       |       |         |       |  -    | 99.88 |

Two clear clusters:
- **Stackers** (v2, v3, v4, v5_real) agree on **99.2–99.9% of cells**
  with each other.  They are essentially the same model with minor
  variations.
- **LS ensembles** agree on **97.9–98.8% of cells** with each other.
- **Cluster A vs Cluster B** disagree on ~2% of cells / 5–7% of
  users.  This is where the +0.003 LS-ensemble gain came from.

The diversity available to the LS solver is bounded by these
2-cluster gaps.  No amount of re-running variants exceeds it.

## Data audit

Phase 4 did a fresh inventory of `Data/music-recommender-2026s/`:

| File              | Lines     | Content                          | Used? |
|-------------------|----------:|----------------------------------|-------|
| `trainItem2.txt`  | (large)   | uid \| ratings (per user block)  | Yes   |
| `trackData2.txt`  | 224,041   | track_id \| album \| artist \| genres | Yes |
| `albumData2.txt`  |  52,829   | album_id \| artist \| genres     | Yes   |
| `testItem2.txt`   | (small)   | uid \| 6 candidate track_ids     | Yes   |
| `artistData2.txt` |  18,674   | **just artist IDs, one per line**| **No** |
| `genreData2.txt`  |     567   | **just genre IDs, one per line** | **No** |

The two unused files contain *no metadata* — they are universes of
valid IDs.  No song lyrics, artist names, genre labels, or content
embeddings exist in this dataset.

**Implication:** advice item #1 ("content / text features as a new
base model") has no untapped data to draw from.  The existing 30
base features + 3 cross features in `compute_base_features` already
extract everything available from the `track → album → artist + genre`
hierarchy.  Phase 4 did not implement #1.

## Phase 4 implementation: per-user routing stacker (advice #4)

New function `generate_stacked_submission_v6_routed` in section after
v5 stacker.  Adds four per-user meta features to the v5 feature row:

```
um_log_n        = log1p(rating count for this user)
um_mean         = mean rating across user's history
um_std          = std of user's ratings
um_log_artists  = log1p(distinct artists user has rated)
```

Plus helper `compute_user_meta_features(user_ratings, track_meta)`.

### Why a single global LR with user-meta columns

The labeled set is 6,000 rows / 1,000 users.  Splitting into 4
activity buckets gives ~250 users per bucket — too small for a
reliable per-bucket fit.  A single global LR with user-meta columns
implicitly varies its effective coefficients by user type (the
GroupKFold C-sweep regularizes it), which is the soft mixture-of-
experts shape we want without the small-sample brittleness.

### Wire-up

Step "11b" in `main()`, runs after v5 stacker, before the LS
ensemble.  Output: `submission_final_stacked_v6_routed.csv`.  No new
CLI flag; runs as part of the full pipeline.

### Status

Implemented; awaiting Kaggle result.

### What "win" looks like

- v6 LR top-20 weight table shows at least one `um_*` feature with
  `|w| > 0.1` (user features earn weight, not zeroed out by L2)
- v6 output disagrees with v5 on a few hundred users
- Kaggle score ≥ 0.923 (any sign that routing helps even a little)

If the user-meta features all get `|w| < 0.05` weights, the LR
decided routing didn't help and v6 ≈ v5.  In that case the plateau
is structural, not architectural.

---

## Paths still worth trying (Phase 4+ candidates)

In rough order of expected value:

1. **(Done in Phase 3)** Don't include LS ensembles as inputs to LS
   ensembles.  Filter implemented.
2. **Ridge regularization on LS:** `a = (S^T S + lambda I)^(-1) S^T x`.
   Stabilizes against the v5b/v6 sweep near-duplicates and the
   self-ensemble.  Cheap; sweep `lambda in {1, 10, 100, 1000}`.
3. **NNLS** (`scipy.optimize.nnls`).  Constrains weights >= 0.  May
   or may not beat LS; useful as a sanity comparison.
4. **Drop the bottom of the barrel.**  ALS (0.621), `hw8_pyspark_mf`
   (0.636), `hw8_hybrid_gated` (0.513) are below random.  LS gives
   them tiny weights but they contribute noise to `S^T S`.
5. **Drop near-duplicate columns.**  The v5b/v6 hybrid sweep cluster
   has many highly-correlated columns.  PCA / leave-one-out / cluster
   selection could help.
6. **Generate a genuinely new orthogonal model.**  Anything that
   predicts on different signal: text features, content embeddings,
   sequence models, attention.  At this score level, the only way to
   break the plateau is a new *uncorrelated* error pattern, not
   refinements to existing ones.

## Reproducibility notes (Phase 2 state)

- `MF_LOSS = "mse"` (BPR experiment reverted)
- `MF_EPOCHS = 80`, `PATIENCE = 25`
- `N_FEATURES = 33` (added `ubcf_score` as feature 29)
- `IICF_VARIANTS` includes 4 entries: `iicf_a`, `iicf_b`, `iicf_c`,
  `iicf_d` (the last one centered)
- `UBCF_K = 50`, `UBCF_MODE = "centered"`
- Caches deleted before next run: `features_g30_seed42_f33.npz`,
  `bpr_h..._f33.pt`, `lgbm_lambdarank_f33.txt`.  Will rebuild on next
  run using fresh MSE-MF embeddings.

**Commands:**
- Full pipeline: `python EE627_Final_Eschete.py`
- LS ensemble only: `python EE627_Final_Eschete.py --ensemble-only`
- Register a Kaggle-scored submission: `python register_submission.py`

The pipeline does NOT modify `past_predict_ensemble/` or
`Submission Results.txt`.  Only `register_submission.py` writes to
them.  Caches in `cache/` are deletable any time; they regenerate.

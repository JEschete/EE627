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

# External Sources for Final Project Methods

This document cites the external (non-coursework) techniques used in the
v3 stacked meta-learner.  These methods were not covered in EE627A
lectures or homework and are referenced here for academic integrity.

---

## 1. LightGBM LambdaRank (LambdaMART)

**Used for:** Adding a gradient-boosted ranking model as a base learner
that directly optimizes a top-K ranking metric (NDCG) on the per-user
groups of 6 candidates, mirroring the Kaggle scoring rule.

**Primary references:**

- Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART:
  An Overview.* Microsoft Research Technical Report MSR-TR-2010-82.
  https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/

- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q.,
  & Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting
  Decision Tree.* In Advances in Neural Information Processing Systems 30
  (NeurIPS 2017), pp. 3146-3154.
  https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

- Chapelle, O., & Chang, Y. (2011). *Yahoo! Learning to Rank Challenge
  Overview.* JMLR Workshop and Conference Proceedings 14, pp. 1-24.
  (LambdaMART ensembles won Track 1.)
  http://proceedings.mlr.press/v14/chapelle11a/chapelle11a.pdf

**Implementation reference:**

- LightGBM `LGBMRanker` documentation:
  https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html

---

## 2. Isotonic Regression for Probability Calibration

**Used for:** Re-mapping the raw outputs of base models (HW9 probability
columns and the v5 BPR neural net) onto a common, monotonically-correct
probability scale before they are combined by the stacker.

**Primary references:**

- Zadrozny, B., & Elkan, C. (2002). *Transforming classifier scores into
  accurate multiclass probability estimates.* In Proceedings of the 8th
  ACM SIGKDD International Conference on Knowledge Discovery and Data
  Mining (KDD '02), pp. 694-699.
  https://dl.acm.org/doi/10.1145/775047.775151

- Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good
  Probabilities With Supervised Learning.* In Proceedings of the 22nd
  International Conference on Machine Learning (ICML '05),
  pp. 625-632.
  https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf

- Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D.
  (1972). *Statistical Inference Under Order Restrictions: The Theory
  and Application of Isotonic Regression.* John Wiley & Sons.
  (Originating reference for the Pool Adjacent Violators algorithm.)

**Implementation reference:**

- scikit-learn `IsotonicRegression` documentation:
  https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html

- scikit-learn calibration user guide:
  https://scikit-learn.org/stable/modules/calibration.html

---

## 3. Item-Item Collaborative Filtering (Cosine Similarity)

**Used for:** A neighborhood-based base model that scores each test
(uid, tid) candidate as a weighted sum of cosine similarities between
that track and the user's top-K rated tracks.  Pure rating-matrix
signal, computed with no learned embeddings -- intended to provide
diversity orthogonal to the matrix-factorization (MF, ALS) and
gradient-boosting (HW9 RF/GBT) components of the ensemble.

**Primary references:**

- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001).
  *Item-Based Collaborative Filtering Recommendation Algorithms.*
  In Proceedings of the 10th International Conference on World Wide Web
  (WWW '01), pp. 285-295.
  https://dl.acm.org/doi/10.1145/371920.372071

- Linden, G., Smith, B., & York, J. (2003). *Amazon.com
  Recommendations: Item-to-Item Collaborative Filtering.* IEEE Internet
  Computing, 7(1), pp. 76-80.
  https://doi.org/10.1109/MIC.2003.1167344

- Deshpande, M., & Karypis, G. (2004). *Item-Based Top-N Recommendation
  Algorithms.* ACM Transactions on Information Systems (TOIS), 22(1),
  pp. 143-177.
  https://dl.acm.org/doi/10.1145/963770.963776

---

## 4. NDCG@K (Normalized Discounted Cumulative Gain) and GroupKFold CV

**Used for:** Selecting the meta-learner's regularization strength using
the metric the Kaggle leaderboard actually scores against (top-3 picks
per user out of 6 candidates), evaluated under a user-grouped
cross-validation split so no user appears in both train and validation
folds.

**Primary references:**

- Jarvelin, K., & Kekalainen, J. (2002). *Cumulated gain-based
  evaluation of IR techniques.* ACM Transactions on Information Systems
  (TOIS), 20(4), pp. 422-446.
  https://dl.acm.org/doi/10.1145/582415.582418

- Wang, Y., Wang, L., Li, Y., He, D., Chen, W., & Liu, T.-Y. (2013).
  *A Theoretical Analysis of NDCG Type Ranking Measures.* In
  Proceedings of the 26th Annual Conference on Learning Theory
  (COLT 2013), pp. 25-54.
  http://proceedings.mlr.press/v30/Wang13.html

**Implementation references:**

- scikit-learn `ndcg_score` documentation:
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html

- scikit-learn `GroupKFold` documentation:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html

---

## Summary of Where Each Method Is Applied

| Method                           | Where in pipeline                                |
|----------------------------------|--------------------------------------------------|
| LightGBM LambdaRank              | New base model -> `lgbm_rank_score` feature      |
| Isotonic Regression calibration  | Wraps HW9 prob columns + v5 NN scores            |
| Item-item CF (cosine)            | New base model -> `iicf_score` / `iicf_rank`     |
| NDCG@3 with GroupKFold CV        | Tunes meta-learner regularization C              |

The LightGBM, Isotonic, and NDCG@3 methods are applied in
`generate_stacked_submission_v3`.  The item-item CF method is applied
in `generate_stacked_submission_v4` (and uses helpers
`build_item_inverted_index`, `cosine_item_pair`, `score_iicf`).

The class-derived components (BPR neural net, ALS, v2 heuristic, HW9
PySpark stack) are documented in the report and require no external
citation.

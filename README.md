# EE627A --- Data Acquisition and Processing I: Big Data

**Semester:** Spring 2026  
**Student:** Jude Eschete

---

## Course Overview

EE627A covers foundational and applied topics in large-scale data acquisition and processing, including time series analysis, ensemble methods, recommender systems, logistic regression, and feature selection. Coursework centers on Python-based implementations evaluated against real-world datasets.

---

## Repository Structure

### Homework

| Assignment | Topic | Key Files |
|---|---|---|
| **Homework 1** | Higgs Boson Classification (Kaggle) | `Eschete_HW1.tex`, `HiggsBosonCompetition_AMSMetric_rev1.py` |
| **Homework 2** | Exploratory Data Analysis | `AAI627_HW2_Eschete.py`, `Eschete_HW2.tex` |
| **Homework 3** | Multi-Part Statistical Analysis | `AAI627_HW3_Eschete.py`, `AAI627_HW3_Eschete.tex` |
| **Homework 4** | Heuristic & ML Models (Yahoo Music) | `Eschete_HW4_Run_File.py`, `Eschete_HW4_ML_Model.py`, `Eschete_HW4_Heur.py` |
| **Homework 5** | ROC Curves & Logistic Regression | `AAI627_HW5_Eschete.py`, `HW5_Report_Eschete.tex` |
| **Midterm** | Heuristic-Based Music Recommendation | `EE627_Midterm.py`, `EE627_Midterm_Eschete.tex` |

### Lectures

| Week | Date | Topics |
|---|---|---|
| **Week 01** | 01-23 | Course Introduction |
| **Week 02** | 01-30 | XGBoost, Correlation, Autoregressive Models, Time Series |
| **Week 03** | 02-06 | AR Model Selection, ARMA Modeling |
| **Week 04** | 02-13 | Market Basket Analysis, ARIMA, Transition to Recommender Systems |
| **Week 05** | 02-20 | Recommender System Theory |
| **Week 06** | 02-27 | Yahoo Music Recommender Project, Data Structures |
| **Week 07** | 03-06 | Logistic Regression |
| **Week 08** | 03-13 | Feature Selection, Best Subset GLM |
| **Week 09** | 03-26 | Course Conclusion |

---

## Midterm: Heuristic-Based Music Recommendation

The midterm project builds a content-based recommendation system for the Yahoo Music dataset (Kaggle competition). Given six candidate tracks per user, the system recommends three using hierarchy-derived features (Album, Artist, Genre).

**Part 1** implements three heuristic strategies over user-specific feature vectors:
- **Strategy 1 --- Max Genre Score** (AUC 0.829)
- **Strategy 2 --- Weighted Average** (AUC 0.865)
- **Strategy 3 --- Evidence-Weighted Composite** (AUC 0.869)

**Part 2** addresses the cold-start problem with Global Fallback and Dig Deeper (sibling-track search) strategies. Key finding: naive global imputation *decreases* AUC by 0.10--0.12 because non-personalized scores dilute the personalized signal.

---

## Tools & Dependencies

- **Python 3** (standard library + `collections.defaultdict`)
- **LaTeX** (pdflatex with `amsmath`, `booktabs`, `listings`, `tocloft`, `hyperref`)
- **Git LFS** for large data files (`trainItem2.txt`, `trainIdx2_matrix.txt`, `test.csv`, `training.csv`)

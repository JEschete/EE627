# EE627 Homework 4 — Music Recommendation System

A music recommendation system for the Yahoo Music dataset (Kaggle competition). Combines a heuristic content/collaborative-filtering model with a gradient-boosted ML model, blending their ranked outputs for final predictions. Evaluated by AUC-ROC.

## Requirements

**Python 3.8+** with the following packages:

```
numpy
scipy
scikit-learn
```

Install via pip:

```bash
pip install numpy scipy scikit-learn
```

## Data Folder

The `Data/` folder must contain the following files (from the Kaggle competition):

| File | Description |
|------|-------------|
| `trainItem2.txt` | User training histories (UserID, TrackID, Rating) |
| `testItem2.txt` | Test users with 6 candidate tracks each |
| `trackData2.txt` | Track metadata (AlbumID, ArtistID, GenreIDs) |
| `albumData2.txt` | Album metadata |
| `artistData2.txt` | Artist list |
| `genreData2.txt` | Genre list |
| `sample_submission.csv` | Expected output format template |

## How to Run

### Full Pipeline (Recommended)

```bash
python Eschete_HW4_Run_File.py
```

This runs the heuristic model and ML model, blends their scores, and writes `eschete_submission.csv`.

### Quick Smoke Test

```bash
python Eschete_HW4_Run_File.py --quick
```

Uses fewer training users and iterations for a fast sanity check.

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output PATH` | `eschete_submission.csv` | Output CSV path |
| `--binary` | off | Output binary (1/0) labels instead of continuous scores |
| `--quick` | off | Fast mode (2,500 users, 120 iterations) |
| `--heur-content W` | 0.80 | Content weight in heuristic model |
| `--heur-cf W` | 0.20 | Collaborative filtering weight in heuristic model |
| `--ml-n-users N` | 25000 | Number of users sampled for ML training |
| `--ml-neg-per-pos N` | 2 | Negative-to-positive ratio in ML training |
| `--ml-max-iter M` | 450 | Max boosting iterations for ML model |
| `--blend-heur W` | 0.50 | Blend weight for heuristic scores |
| `--blend-ml W` | 0.50 | Blend weight for ML scores |

### Individual Models

Run the heuristic model alone:

```bash
python Eschete_HW4_Heur.py
```

Run heuristic validation (holds out test users, reports AUC):

```bash
python Eschete_HW4_Heur.py --validate
```

Run the ML model alone:

```bash
python Eschete_HW4_ML_Model.py --continuous
```

## Output

The output CSV has two columns — `TrackID` and `Predictor` — with rows formatted as `UserID_TrackID, score`. Each test user has 6 candidate tracks; the top 3 are labeled 1 (recommend) and the bottom 3 are labeled 0.

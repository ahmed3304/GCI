<img width="1460" height="737" alt="percentile" src="https://github.com/user-attachments/assets/7ea76ded-be8f-4c6e-9714-95bdc8aa18b1" />
## Finished within the TOP 1.7% of the competition
# Home Credit Default Risk — All-in-One Pipeline (GPU + CPU, Twin Adjustment)

A single-file, Colab-ready pipeline for the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition. It trains **several models in parallel on one T4 GPU**, blends them with rank-averaged weights, and applies a **second-stage "twin adjustment"** that exploits duplicate applicants appearing under different IDs.

**Measured:** ~**0.77799 leaderboard AUC** (0.778–0.779 depending on seed and blend) on the 32-column application data. The twin stage adds roughly **+0.001 to +0.003 AUC** over the plain blend, validated with group cross-validation.

> **Honest expectations.** On the public leaderboard the CPU-only blend scored about **0.774**; the twin stage added about **+0.0016** on an earlier version. The GPU tree models improve on that; **0.80 is not reachable from these 32 columns** with this pipeline (or, realistically, any pipeline that doesn't use the auxiliary tables).

---

## What's in the box

| Stage | What it does |
|---|---|
| **Features** | Cleaning (`DAYS_EMPLOYED` sentinel, `XNA`, outliers), 40+ ratio / age / enquiry features, cross-fitted **regression imputation of `EXT_SOURCE_1/2/3`**, loan-product clusters from `AMT_CREDIT / AMT_ANNUITY`, and **leakage-free in-fold target encoding** of those clusters (default rate spans ~1%–25% across clusters). |
| **Models** | GPU: `xgb`, `xgb2`, `cat` (CatBoost capped at 40% GPU RAM). CPU: `hgb_f` (sklearn HistGradientBoosting), plus linear/neural models `lr`, `mlpA`, `mlpB`, `mlp3` fed quantile + one-hot + **spline** features. Each model runs in its own process; up to `MAX_PARALLEL` train at once. |
| **Blend** | Rank-average with weights hill-climbed on out-of-fold predictions, starting from equal weights. |
| **Twin adjustment** | Detects the same person under multiple IDs via invariant `DAYS_*` differences (application dates shift all `DAYS_*` fields equally, so `DAYS_REGISTRATION − DAYS_BIRTH` etc. are fingerprints). A second-stage ensemble (LR on core twin features, regularised LR on all twin features, small gradient-boosted model) turns the twin outcomes — **direction-aware**: an earlier default predicts a later loan ~70%, a later default predicts the earlier loan ~23% — into a final probability. ~2.7% of rows have a twin. |
| **Validation** | Stratified K-fold OOF for base models; **group K-fold** (a person never straddles train/validation) for the twin stage. Reports **test-like OOF AUC** (validation re-weighted by a train-vs-test adversarial classifier) alongside plain OOF. |

---

## Results

| Model / stage | Plain OOF AUC | Test-like OOF AUC |
|---|---|---|
| `xgb` | ~0.775 | — |
| `cat` | ~0.776 | — |
| `hgb_f` | ~0.772 | — |
| `lr` / `mlp*` | ~0.75–0.77 | — |
| **Blend** | **~0.775–0.777** | — |
| **Blend + twin adjustment** | **~0.778–0.779** | — |

Numbers vary slightly with seed, fold count, and which models finish. The **test-like** column is the one to compare against the leaderboard; the leaderboard itself is noisy (±0.004 on ~61.5k rows).

---

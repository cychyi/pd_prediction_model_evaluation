# Comparative Evaluation of ML and DL Models for Probability of Default (PD)

Thesis codebase for **Cham Ying Chyi (23076054)**, University of Malaya — a
comparative study of machine-learning and deep-learning models for estimating
the Probability of Default (PD) on the
[American Express Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
dataset.

The pipeline is split into five clean stages, each a standalone script, exactly
mirroring the methodology in the proposal.

| Stage | Script | What it does |
|------|--------|--------------|
| 1 | `src/01_data_landing_eda.py` | Stream the 16 GB CSV to Parquet in 500k-row chunks, then run EDA (missing rates, descriptive stats, feature families, target balance, statements-per-customer). |
| 2 | `src/02_data_preprocessing.py` | Denoise (×100, round), encode categoricals `D_63`/`D_64`, drop features with >90% missing or a single value (incl. `D_87`). |
| 3 | `src/03_feature_engineering.py` | Build **snapshot** features (~1,000, one row/customer) for ML **and** the **sequential** tensor `(N, 13, F)` for DL. |
| 4 | `src/04_feature_selection.py` | Variance → correlation (>0.95) → LightGBM-gain funnel on the snapshot features. |
| 5a | `src/05a_modelling_ml.py` | Logistic Regression, XGBoost, LightGBM. |
| 5b | `src/05b_modelling_dl.py` | LSTM, **GRU**, Transformer Encoder (PyTorch, auto CUDA/MPS/CPU). |
| 5c | `src/05c_modelling_ensemble.py` | **LightGBM+GRU, XGBoost+LSTM, LightGBM+LSTM, XGBoost+GRU** (validation-weighted rank blend). |
| 5d | `src/05d_evaluation.py` | Slide-style comparison tables (train + validate), decile risk-ranking, overfit check, SHAP, Integrated Gradients, attention. |

All models are scored with the **same** metrics module (`utils/metrics.py`):
ROC-AUC, KS, Gini (`2·AUC−1`), log-loss, 10-decile risk-ranking, and the
train-vs-validate **Diff Gini (%)** overfit rule (>30% ⇒ overfit). The official
Kaggle competition metric is also available in `utils/amex_metric.py`.

---

## Quick start

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. place the dataset
#    data/raw/train_data.csv
#    data/raw/train_labels.csv
#    (download from the Kaggle competition page)

# 3. run everything
bash run_all.sh
# ...or run stages individually, e.g.:
python src/01_data_landing_eda.py
```

Outputs land in `data/reports/` (`comparison_train.csv`,
`comparison_validate.csv`, `best_model_deciles.csv`, `shap_top10.csv`,
`ig_top10.csv`, plus all EDA tables).

To smoke-test the whole pipeline without the 16 GB download, generate a tiny
synthetic dataset first:

```bash
python make_synthetic.py && bash run_all.sh
```

---

## Where should I run this? (Mac Mini M4 vs Kaggle)

Short answer: **do the data work once on Kaggle, save the engineered features as
a Kaggle Dataset, then train. Develop/debug locally on the synthetic data.**
Here is the reasoning and the alternatives.

### The constraints
- The raw data is ~16 GB (train) / ~33 GB (test) and **already lives on Kaggle**.
  Downloading 33 GB to a Mac Mini, then re-uploading engineered features, wastes
  hours of bandwidth and disk.
- Your Mac Mini M4 is **excellent for CPU work** (the data wrangling and the
  tree models) and has a capable GPU via **MPS** — but PyTorch MPS is slower than
  a Kaggle T4/P100 for recurrent/attention models and still has occasional op
  gaps. Base M4s also have 16 GB unified memory, which is tight for the full
  16 GB train file.
- Kaggle gives you a **free T4×2 / P100 GPU**, ~30 GB RAM, and the dataset
  pre-mounted — but sessions are capped (~9 h) and `/kaggle/working` output is
  limited (~20 GB), so you cannot keep 33 GB of intermediate Parquet around.

### Recommended workflow (hybrid, fastest in practice)

1. **Develop locally on the Mac.** Run `make_synthetic.py` + `run_all.sh` to
   confirm the code works on tiny data. Commit to GitHub. This is fast and free.
2. **Stage 1–4 on Kaggle (CPU notebook).** The aggregation is CPU-bound and
   memory-heavy; do it once on Kaggle where the raw CSVs are already mounted and
   you have 30 GB RAM. Save `snapshot_selected.parquet` and `sequential.npz`
   (a few GB total) as a **Kaggle Dataset** — see `notebooks/kaggle_data_prep.md`.
3. **Stage 5a (ML) — either place is fine.** XGBoost/LightGBM on ~1,000 snapshot
   features run comfortably on the M4 CPU (minutes–tens of minutes). Set
   `tree_method="gpu_hist"` in `configs/config.py` if you run XGBoost on Kaggle GPU.
4. **Stage 5b (DL) on Kaggle GPU.** Point a **GPU notebook** at the engineered
   Dataset from step 2 and train LSTM/GRU/Transformer there. This is where the
   free GPU pays off — see `notebooks/kaggle_dl_train.md`.
5. **Stages 5c–5d anywhere.** Blending and table-building are lightweight.

The code auto-detects the environment (`configs/config.py`): on Kaggle it reads
`/kaggle/input` and writes `/kaggle/working`; locally it uses `data/`. You do
**not** change any paths by hand.

### Simpler alternatives
- **All on Kaggle:** easiest and fully reproducible for graders — run Stages 1–4
  in a CPU notebook, 5a–5d in a GPU notebook. No local setup at all.
- **All on the Mac:** entirely doable if you have a 24 GB M4 and some patience;
  DL training just takes longer on MPS. Good if you are offline or want full
  control. Use `reduce_mem_usage` (already applied) and consider lowering
  `DL_PARAMS["batch_size"]` if memory is tight.

**Bottom line:** for the thesis I recommend the hybrid (data prep on Kaggle →
features as a Dataset → DL training on Kaggle GPU), with local runs on synthetic
data for development. It is the fastest path and keeps everything reproducible
from GitHub.

---

## Uploading to GitHub

```bash
cd amex-pd-thesis
git init
git add .                      # .gitignore already excludes data/ artifacts
git commit -m "AMEX PD thesis pipeline: ML + DL + ensembles"
git branch -M main
git remote add origin https://github.com/<your-username>/amex-pd-thesis.git
git push -u origin main
```

Do **not** commit the dataset or the `data/parquet|interim|features|models`
folders — they are large and already git-ignored. Keep `data/reports/` if you
want the result tables version-controlled for the thesis.

## Uploading to Kaggle

Two things you may want on Kaggle:

1. **The engineered features as a Dataset** (so the GPU notebook can load them):
   ```bash
   pip install kaggle           # then put your kaggle.json token in ~/.kaggle/
   cd data/features
   kaggle datasets init -p .
   # edit dataset-metadata.json (title, id: <username>/amex-pd-features)
   kaggle datasets create -p .
   ```
2. **The notebooks** in `notebooks/` — paste them into a Kaggle Notebook, attach
   the competition dataset (and your features Dataset), and run.

See `notebooks/kaggle_data_prep.md` and `notebooks/kaggle_dl_train.md` for the
exact cells.

---

## Repository layout

```
amex-pd-thesis/
├── configs/config.py          # all paths, feature groups, hyperparameters
├── utils/
│   ├── common.py              # path bootstrap, memory reduction, ID compression
│   ├── metrics.py             # ROC-AUC, KS, Gini, log-loss, deciles, overfit
│   └── amex_metric.py         # official Kaggle competition metric (optional)
├── src/
│   ├── 01_data_landing_eda.py
│   ├── 02_data_preprocessing.py
│   ├── 03_feature_engineering.py
│   ├── 04_feature_selection.py
│   ├── 05a_modelling_ml.py
│   ├── 05b_modelling_dl.py
│   ├── 05c_modelling_ensemble.py
│   └── 05d_evaluation.py
├── notebooks/                 # Kaggle notebook templates
├── make_synthetic.py          # tiny fake dataset for smoke-testing
├── run_all.sh
├── requirements.txt
└── README.md
```

## Notes for the write-up
- The slide lists only `D_63`/`D_64` as categoricals; the full official AMEX
  categorical list is in `configs/config.py` (`CAT_COLS_FULL`) if your supervisor
  wants the complete treatment — flip `CAT_COLS` to use it.
- The overfit threshold, missing threshold, sequence length, and all
  hyperparameters live in `configs/config.py` so every choice is documented and
  reproducible.
- For the final thesis numbers, increase `n_estimators`, `epochs`, and tune the
  hyperparameters; the defaults are tuned for a sane first run, not the leaderboard.

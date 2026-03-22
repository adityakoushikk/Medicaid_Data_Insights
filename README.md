# Medicaid Provider Anomaly Detection

# Setup

### 1. Install dependencies

Create and activate a virtual environment, then install requirements:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment template and fill in your values:

```bash
cp .env.example .env
```

`.env.example`:
```
WANDB_API_KEY=  # not needed if not using wandb
```

### 2. Set up Weights & Biases (optional)

wandb is not required but recommended for result visualization — the CSV result logger works as well (see usage for details). To use wandb:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from **Settings → API keys**
3. Add it to your `.env` file as `WANDB_API_KEY`, or log in directly:

```bash
wandb login
```

---

### 3. Download datasets

Place all three files in `data/datasets/` under the exact filenames listed below.

---

#### Medicaid Provider Spending — `medicaid-provider-spending.csv`
**Source:** [opendata.hhs.gov/datasets/medicaid-provider-spending](https://opendata.hhs.gov/datasets/medicaid-provider-spending/)

Raw Medicaid billing data (~10 GB). One row per (provider, month, HCPCS code). Primary input for all feature engineering steps.

---

#### LEIE Exclusion Labels — `LEIElabels.csv`
**Source:** [oig.hhs.gov/exclusions/exclusions_list.asp](https://oig.hhs.gov/exclusions/exclusions_list.asp)

HHS Office of Inspector General List of Excluded Individuals/Entities — providers sanctioned for fraud or abuse. Used as **weak supervised labels**: LEIE-listed NPIs serve as positive fraud signals during model evaluation and label construction.

---

#### NPPES Provider Registry — `nppes.csv`
**Source:** [download.cms.gov/nppes/NPI_Files.html](https://download.cms.gov/nppes/NPI_Files.html)

National Plan and Provider Enumeration System full replacement file (~10 GB). Contains NPI, entity type (individual vs. organization), state, and taxonomy codes. Used to assign each provider a cohort based on `(state, entity_type)`.

# Data Model Summary


1. **Build cohorts** (`build_provider_cohorts.py`) — joins Medicaid NPIs against NPPES to assign each provider a `(state, entity_type)` cohort label (e.g. `TX_individual`).

2. **Build labels** (`build_labels.py`) — cross-references providers against LEIE to produce a binary label per NPI (`1` = excluded, `0` = not). These are weak supervised labels used for evaluation only, not training.

3. **Provider-month features** (`create_provider_month_dataset.py`) — aggregates raw billing to one row per `(provider, month)`. A provider-month is a single snapshot of one provider's activity in one calendar month - Features include things like `paid_per_claim_t`, `hcpcs_entropy` (diversity of codes billed), and `top_code_paid_share` (concentration in a single code). See `provider_month_data_dictionary.csv` for the full list.

4. **Provider-level features** (`create_provider_level_from_month.py`) — collapses the provider-month table to one row per provider. Captures temporal patterns: billing volatility, MoM growth rates, spike behavior, code diversity over time, structural breaks (PELT changepoints). See `provider_level_data_dictionary.csv` for details.

**Model** — configured via [Hydra](https://hydra.cc) (`configs/`), trained with [PyTorch Lightning](https://lightning.ai). The default model is an autoencoder trained on provider-level features unsupervised. Results are tracked with wandb or the CSV logger.

---

# Model Architecture

Default autoencoder (`configs/model/autoencoder.yaml`). Trained on label=0 providers only.

```
n_features → 64 → 32 → 16 → [8] → 16 → 32 → 64 → n_features
                              ^^^
                           bottleneck
```

Each layer: Linear + BatchNorm + ReLU. Dropout=0.1.

**Anomaly score** = `MSE(x, x̂)` — reconstruction error per provider.

---

# Usage

One cohort at a time — e.g. `NV_organization`, `TX_individual`. *(Multi-cohort batch support planned.)*

### 1. Build provider-month dataset

Preprocessing pipeline for chosen cohort (label construction, cohort assignment, feature aggregation):

```bash
./run_provider_month.sh NV_organization
```

Produces `data/outputs/provider_month_NV_organization.csv`.

---

### 2. Run anomaly detector

#### First run — provider-level features computed from scratch

```bash
./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv
```

Output → `data/outputs/<timestamp>/`:
- `provider_level.csv` — one row per provider, all engineered features
- `scored_providers.csv` — ranked by anomaly score (MSE)
- `feature_auroc.csv` — per-feature AUROC vs LEIE labels
- `lift_table.csv` — lift/precision/hits at each percentile cutoff, with expected hits under random selection
- `metrics.csv` — train/val loss per epoch
- `checkpoints/best.ckpt` — best checkpoint

#### Reuse existing provider-level file

Provider-level computation is the slowest step — can be skipped if features from previous run are to be reused:

```bash
./run_anomaly.sh \
  data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
  data.provider_level_csv=data/outputs/2026-03-21_22-57-39/provider_level.csv
```

---

### 3. Comparing cohorts in wandb

Tag each cohort run, compare side-by-side in the same wandb project:

```bash
# NV organizations
./run_anomaly.sh \
  data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
  data.provider_level_csv=data/outputs/2026-03-21_22-57-39/provider_level.csv \
  logger=wandb logger.project=medicaid-insights \
  "tags=[NV_organization,baseline]"

# TX individuals
./run_anomaly.sh \
  data.provider_month_csv=data/outputs/provider_month_TX_individual.csv \
  logger=wandb logger.project=medicaid-insights \
  "tags=[TX_individual,baseline]"
```

---

### 4. CSV vs wandb logger

**CSV (default)** — no account needed, writes locally to `data/outputs/<timestamp>/metrics.csv`:

```bash
./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv
```

**wandb** — online dashboards, run comparison, artifact tracking. Requires `WANDB_API_KEY` or `wandb login`:

```bash
./run_anomaly.sh \
  data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
  data.provider_level_csv=data/outputs/2026-03-21_22-57-39/provider_level.csv \
  logger=wandb logger.project=medicaid-insights logger.name=NV_org_run1 \
  "tags=[NV_organization,no_feature_selection]"
```


---

### 5. Feature engineering tuning

```bash
# Rolling z-score sensitivity + window
data.provider_level_features.rolling_flags.robust_z_threshold=2.0
data.provider_level_features.rolling_flags.window_obs=6

# PELT changepoint penalty (lower = more changepoints)
data.provider_level_features.changepoints.penalty=1.0

# Min months of history required
data.min_months=12

# Disable feature selection (use all features, not just AUROC > 0.65)
data.feature_selection.auroc_threshold=null
```

See `parameter_dictionary.csv` for full list.

---

### 6. Autoencoder tuning

```bash
# Architecture
"model.net.encoder_dims=[128, 64, 32]"
model.net.bottleneck_dim=16
model.net.activation=leaky_relu
model.net.dropout_rate=0.2

# Optimizer
model.optimizer.lr=0.0005
model.optimizer.weight_decay=1e-4

# Training
train.max_epochs=200
data.batch_size=128
```

---

### 7. Multirun sweep

`--multirun` runs every combination. Outputs → `data/outputs/multirun/<timestamp>/0/`, `/1/`, ...

```bash
./run_anomaly.sh --multirun \
  data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
  data.provider_level_csv=data/outputs/2026-03-21_22-57-39/provider_level.csv \
  logger=wandb logger.project=medicaid-insights "tags=[NV_organization,sweep]" \
  data.feature_selection.auroc_threshold=0.55,0.65,null \
  data.provider_level_features.changepoints.penalty=1.0,2.0
```


# An Interpretable Pipeline for Imbalanced Industrial Anomaly Detection

**VAE-GAN Augmentation · CatBoost Classification · TreeSHAP Interpretation**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00)](https://catboost.ai/)
[![SHAP](https://img.shields.io/badge/XAI-TreeSHAP-8A2BE2)](https://github.com/shap/shap)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code for the MSc Robotics dissertation by **Dong-Jun Chen**,
University of Bristol & University of the West of England, August 2025.
📄 **[Read the dissertation (PDF)](docs/Dissertation.pdf)**

Anomalous wafers make up only about 6% of the UCI-SECOM semiconductor dataset, and
supervised detectors trained on it hardly ever see a defect. Black-box detectors are also hard
for process engineers to trust. This project builds one **end-to-end, interpretable pipeline**
that:

1. **reduces** 590 noisy sensor features to a compact, informative subset,
2. **generates realistic synthetic anomalies** with a **VAE-GAN** (vanilla VAE as baseline) to rebalance the classes,
3. **detects** anomalies with a Bayesian-optimised **CatBoost** classifier,
4. **explains** every detection with **TreeSHAP**, globally and per instance, and
5. **serves** the model through a **FastAPI** endpoint that scores streamed raw records in real time.

<p align="center">
  <img src="docs/figures/fig01_system_flow.png" width="560" alt="System flow chart: offline training and online validation">
</p>

---

## Contents

- [Highlights](#highlights)
- [Research aims](#research-aims)
- [Dataset](#dataset)
- [Method](#method)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Real-time API](#real-time-api)
- [Implementation notes](#implementation-notes)
- [Limitations and future work](#limitations-and-future-work)
- [Citation](#citation)
- [License and acknowledgements](#license-and-acknowledgements)

---

## Highlights

| | |
|---|---|
| **Feature reduction** | 590 → 33 sensor features (a 94.4% reduction) through EDA-driven filtering |
| **Generation quality** | VAE-GAN anomalies are closer to the real anomalies than VAE anomalies on **all four** distance metrics (KL, JS, Wasserstein, MMD) |
| **Detection** | VAE-GAN + CatBoost: **Precision 0.988 · Recall 0.953 · F1 0.970 · ROC-AUC 0.985 · PR-AUC 0.989**, ahead of prior SMOTE-based tree models on UCI-SECOM |
| **Interpretability** | TreeSHAP bar, beeswarm and waterfall plots; feature **511** has the largest mean impact |
| **Deployment** | Online simulation with 10 held-out raw records: all 7 normal records pass, 2 of 3 anomalies are flagged and explained automatically |

## Research aims

| # | Aim | How it is evaluated |
|---|-----|---------------------|
| 1 | Identify the critical features in high-dimensional sensor data | EDA: missing values, variance, label correlation, collinearity |
| 2 | Verify that VAE-GAN captures the real anomaly distribution better than a VAE | KL / JS divergence, Wasserstein distance, MMD, KDE & KernelPCA plots |
| 3 | Quantify how synthetic-data quality affects detection | Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix |
| 4 | Find the best oversampling ratio | CatBoost trained with 0%, 30%, 50%, 80%, 100% synthetic anomalies |
| 5 | Provide global and local explanations | TreeSHAP bar, beeswarm and waterfall plots |
| 6 | Validate online detection | Raw records streamed to a FastAPI service |

## Dataset

[**UCI-SECOM**](https://archive.ics.uci.edu/dataset/179/secom) ([Kaggle mirror](https://www.kaggle.com/datasets/paresh2047/uci-semcom)) contains
**1,567** production records from a semiconductor manufacturing line. Each record has **590** sensor
measurements, a timestamp and a `Pass/Fail` label (`-1` = pass/normal, `1` = fail/anomaly).
Only **104** records (≈ 6.6%) are anomalies, and many features are constant, sparse or highly collinear.

The raw file is included at [`data/raw/uci-secom.csv`](data/raw/uci-secom.csv). The dataset is
published by the UCI Machine Learning Repository under CC BY 4.0.

<p align="center">
  <img src="docs/figures/fig02_class_distribution.png" width="380" alt="Proportion of normal vs anomalous records">
  <img src="docs/figures/fig03_kpca_normal_vs_anomaly.png" width="380" alt="KernelPCA of normal vs anomalous records">
</p>

The KernelPCA projection (right) shows that normal and anomalous records overlap heavily. That
breaks the main assumption of unsupervised detectors, which is why this work uses a **supervised**
detector combined with data augmentation.

## Method

```mermaid
flowchart LR
    A[Raw UCI-SECOM<br/>1567 × 590] --> H[Hold out 10 raw records<br/>7 normal + 3 anomalous]
    A --> B[Preprocessing<br/>590 → 33 features]
    B --> C[Real anomalies<br/>101]
    C --> D1[VAE]
    C --> D2[VAE-GAN]
    D1 --> E[Synthetic anomalies<br/>1355]
    D2 --> E
    B --> F[Combined balanced set<br/>1456 normal + 1456 anomalous]
    E --> F
    F --> G[CatBoost<br/>Bayesian optimisation]
    G --> S[TreeSHAP]
    H --> API[FastAPI /predict]
    G --> API
    S --> API
    API --> W[Score + waterfall plot]
```

### 1. Preprocessing (`scripts/01_EDA.py`)

Before any cleaning, 10 raw records (7 normal, 3 anomalous) are held out for the online validation.
The rest goes through the following steps:

| Step | Rule |
|------|------|
| Hold out raw records | 7 normal + 3 anomalous kept aside |
| Drop timestamp | |
| Missing values | drop features with > 20% missing, mean-impute the rest |
| Constant features | drop features with a single unique value |
| Low variance | drop features with variance < 0.05 |
| Label correlation | drop features with \|Pearson r\| < 0.05 with the label |
| Collinearity | for each pair with \|r\| ≥ 0.75, keep only the most informative feature |
| Standardisation | z-score, with the scaler fitted on normal records only |

The feature dimensionality drops from **590 to 33 (a 94.4% reduction)**.

The fitted scaler and the indices of the kept features are saved in `artifacts/preprocessing/`, so the
API applies exactly the same transformation to a raw 590-value record.

### 2. Synthetic anomaly generation (`scripts/02_GeneratorVAE.py`, `scripts/03_GeneratorVAEGAN.py`)

Both generators are trained **only on the 101 real anomalies** and share the same architecture, so any
difference in output comes from the training objective alone:

- **Encoder**: Dense 512 → 256 → 128 → 64 → 32 (ReLU, BatchNorm, L2) → (μ, log σ²)
- **Decoder / generator**: the mirror image, with a linear output layer
- **Discriminator** (VAE-GAN only): Dense 512 → 256 → 128 → 64 → 32 → sigmoid

**VAE** ([`models/VAE.py`](src/anomaly_detection/models/VAE.py)) minimises the β-weighted negative ELBO:

$$\mathcal{L}_{VAE} = \beta \, D_{KL}\big(q(z\mid x)\,\|\,p(z)\big) + \mathcal{L}_{recon}$$

**VAE-GAN** ([`models/VAEGAN.py`](src/anomaly_detection/models/VAEGAN.py), Larsen et al. 2016)
replaces the pixel-wise reconstruction loss with a **feature-matching loss** on an intermediate
discriminator layer, which counteracts the over-smoothing (“blurring”) of a VAE. Encoder, generator
and discriminator are each trained with their own optimiser:

$$\mathcal{L}_{Enc} = \beta\,\mathcal{L}_{prior} + \mathcal{L}^{Dis_l}_{recon}, \qquad
\mathcal{L}_{Gen} = \gamma\,\mathcal{L}^{Dis_l}_{recon} - \mathcal{L}_{GAN}, \qquad
\mathcal{L}_{Dis} = \mathcal{L}_{GAN}$$

$$\mathcal{L}_{GAN} = \log D(x) + \log\big(1 - D(\tilde{x})\big) + \log\big(1 - D(G(z))\big)$$

where $\tilde{x}$ is a reconstruction and $G(z)$ is a sample decoded from the prior. By default, enough
anomalies are generated to balance the two classes (1,355 samples); the factor in
`synthetic_num * 1` (`03_GeneratorVAEGAN.py`) scales that number.

### 3. Generation quality (`scripts/04_Measure.py`)

Real and generated anomalies are compared feature by feature with four complementary distances:
**KL divergence** and **JS divergence** (probability mass, histogram-based), plus **Wasserstein
distance** and **MMD** with an RBF kernel (geometry and feature-space structure). Lower values mean
the generated anomalies are closer to the real ones.

### 4. CatBoost detector (`scripts/05_CatShap.py`)

The combined dataset is split 80/20 into training and test sets. CatBoost uses **ordered boosting**, which
avoids target leakage, with `Logloss` as the loss and F1 as the evaluation metric. The
hyperparameters were selected by **Bayesian optimisation** (Gaussian-process `gp_minimize`),
which maximises the mean 5-fold stratified cross-validated F1:

$$\theta^{*} = \arg\min_{\theta} \; -\frac{1}{5}\sum_{k=1}^{5} F1_k(\theta)$$

| max_depth | iterations | learning_rate | l2_leaf_reg | min_data_in_leaf |
|---:|---:|---:|---:|---:|
| 4 | 73 | 0.1 | 12 | 6 |

These are the values reported in the dissertation (Table II); the hyperparameters currently used
in `05_CatShap.py` are listed under [Implementation notes](#implementation-notes).
Scores ≥ 0.5 are classified as anomalies.

### 5. TreeSHAP explanations (`shap_explain()` in `scripts/05_CatShap.py`)

TreeSHAP computes exact Shapley values for tree ensembles in $O(TLD^2)$ instead of $O(2^M)$.
The background set contains **1,000 normal training records**, so each attribution measures how far
a record deviates from normal behaviour. Bar and beeswarm plots give the global picture, and a
waterfall plot explains a single detection.

### 6. Online validation (`src/anomaly_detection/API.py`)

A FastAPI service loads the model, the preprocessing artifacts and the SHAP explainer. The 10
held-out **raw** records are streamed to `/predict` one by one. Each record goes through the same
preprocessing and is scored by CatBoost. When it is flagged, a waterfall plot is rendered for
the domain expert.

## Results

All numbers and figures in this section are those reported in the [dissertation](docs/Dissertation.pdf).

### Generated anomaly quality

| VAE (baseline) | VAE-GAN |
|:---:|:---:|
| <img src="docs/figures/fig04_kde_vae.png" width="400" alt="KDE of real vs VAE-generated anomalies"> | <img src="docs/figures/fig06_kde_vaegan.png" width="400" alt="KDE of real vs VAE-GAN-generated anomalies"> |
| <img src="docs/figures/fig05_kpca_vae.png" width="400" alt="KernelPCA of real vs VAE-generated anomalies"> | <img src="docs/figures/fig07_kpca_vaegan.png" width="400" alt="KernelPCA of real vs VAE-GAN-generated anomalies"> |

VAE samples collapse into a narrow peak and cover only a small corner of the real anomalies in
KernelPCA space. VAE-GAN samples follow the real distributions much more closely.

| Total distance over all features ↓ | VAE | VAE-GAN |
|---|---:|---:|
| KL divergence | 433.655 | **26.756** |
| JS divergence | 13.051 | **4.046** |
| Wasserstein distance | 21.401 | **12.457** |
| MMD | 10.466 | **4.975** |

### Detection performance

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| SMOTE + DT (prior study) | 0.869 | 0.958 | 0.911 | – | – |
| SMOTE + AdaBoost (prior study) | 0.919 | 0.926 | 0.965 | – | – |
| SMOTE + GBT (prior study) | 0.968 | 0.958 | 0.963 | – | – |
| SMOTE + XGB (prior study) | 0.975 | 0.958 | 0.966 | – | – |
| VAE + CatBoost (baseline) | 0.981 | 0.949 | 0.965 | **0.986** | **0.989** |
| **VAE-GAN + CatBoost (proposed)** | **0.988** | 0.953 | **0.970** | 0.985 | **0.989** |

| VAE + CatBoost | VAE-GAN + CatBoost |
|:---:|:---:|
| <img src="docs/figures/fig09_confusion_matrix_vae.png" width="340" alt="Confusion matrix with VAE augmentation"> | <img src="docs/figures/fig10_confusion_matrix_vaegan.png" width="340" alt="Confusion matrix with VAE-GAN augmentation"> |

With VAE-GAN augmentation, CatBoost makes fewer false negatives (11 vs 14) and fewer false
positives (4 vs 5).

### Oversampling ratio

| OS ratio | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| 0% | 0.000 | 0.000 | 0.000 | 0.740 | 0.130 |
| 30% | 0.964 | 0.788 | 0.867 | 0.922 | 0.904 |
| 50% | 0.984 | 0.871 | 0.924 | 0.922 | 0.962 |
| 80% | 0.986 | 0.939 | 0.962 | 0.974 | 0.980 |
| **100%** | **0.988** | **0.953** | **0.970** | **0.985** | **0.988** |

Without augmentation CatBoost never predicts an anomaly. Performance rises steeply with the
oversampling ratio, and 80% is already close to the fully balanced 100%. The chart is produced by
[`results/OSRate.py`](results/OSRate.py).

### Interpretability

| Global feature importance | Feature value vs. SHAP value |
|:---:|:---:|
| <img src="docs/figures/fig12_shap_importance.png" width="380" alt="Mean absolute SHAP value per feature"> | <img src="docs/figures/fig13_shap_beeswarm.png" width="400" alt="SHAP beeswarm plot"> |

- Feature **511** has the largest mean |SHAP|, followed by 59 and 40.
- The beeswarm plot shows *how* a feature pushes towards an anomaly. For example, low values of
  feature 14 or high values of feature 21 increase the anomaly score.

<p align="center">
  <img src="docs/figures/fig14_shap_waterfall.png" width="460" alt="Waterfall explanation of an anomaly detected online">
</p>

During the online simulation, the waterfall plot above explains a streamed anomaly (score 0.620).
Feature 59 (value 1.962) contributes the most, which is consistent with the beeswarm plot.

## Repository structure

```
.
├── data/
│   ├── raw/uci-secom.csv                 # original dataset
│   ├── processed/                        # uci-secom-cleaned.csv, anomaly_data.csv
│   ├── streaming/test_data.csv           # 10 held-out raw records for the API demo
│   └── synthetic/                        # synthetic_{vae,vaegan}.csv, combined_data_{vae,vaegan}.csv
├── artifacts/
│   ├── preprocessing/                    # scaler.pkl, all_cols.npy, idx.npy
│   └── model/                            # model.cbm, threshold.npy, bg_1000.npy (SHAP background)
├── scripts/                              # pipeline steps, run in order
│   ├── 01_EDA.py                         # preprocessing + EDA figures
│   ├── 02_GeneratorVAE.py                # VAE anomaly generation
│   ├── 03_GeneratorVAEGAN.py             # VAE-GAN anomaly generation
│   ├── 04_Measure.py                     # KL / JS / Wasserstein / MMD
│   └── 05_CatShap.py                     # CatBoost training, evaluation, TreeSHAP
├── src/anomaly_detection/                # importable package (installed in editable mode)
│   ├── models/VAE.py                     # VAEGenerator
│   ├── models/VAEGAN.py                  # VAE, Discriminator, VAEGAN
│   ├── Preprocess.py                     # transform(): raw 590-value record → model input
│   └── API.py                            # FastAPI service + streaming simulation
├── results/
│   ├── OSRate.py                         # oversampling-ratio chart (Fig. 11)
│   └── plots_output/                     # SHAP waterfall plots written by the API
├── docs/
│   ├── Dissertation.pdf
│   └── figures/                          # figures used in this README
├── pyproject.toml                        # dependencies and package configuration
├── uv.lock                               # exact, tested dependency versions
└── .python-version                       # Python 3.11
```

## Getting started

### Installation

Requires Python 3.10–3.12 (tested on 3.11).

**Option A – uv (recommended).** `uv sync` creates `.venv`, downloads Python 3.11 if needed,
installs the exact versions in `uv.lock` and installs `src/anomaly_detection` in editable mode:

```bash
git clone https://github.com/JimCH123/Master_Dissertation.git
cd Master_Dissertation
uv sync
```

**Option B – venv + pip.**

```bash
git clone https://github.com/JimCH123/Master_Dissertation.git
cd Master_Dissertation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # dependencies + the anomaly_detection package
```

The editable install is what lets the scripts in `scripts/` import `anomaly_detection`.

### Run the pipeline

Run each step from the repository root. With uv, prefix the commands with `uv run`
(e.g. `uv run python scripts/01_EDA.py`); with venv + pip, activate `.venv` first.

| Step | Command | Reads | Writes |
|------|---------|-------|--------|
| 1. Preprocess & EDA | `python scripts/01_EDA.py` | `data/raw/` | `data/processed/`, `data/streaming/`, `artifacts/preprocessing/` |
| 2. VAE generation | `python scripts/02_GeneratorVAE.py` | `data/processed/` | `data/synthetic/*_vae.csv` |
| 3. VAE-GAN generation | `python scripts/03_GeneratorVAEGAN.py` | `data/processed/` | `data/synthetic/*_vaegan.csv` |
| 4. Generation quality | `python scripts/04_Measure.py` | `data/processed/`, `data/synthetic/` | printed distances + comparison chart |
| 5. Detect & explain | `python scripts/05_CatShap.py` | `data/synthetic/combined_data_vaegan.csv` | metrics, confusion matrix, ROC/PR curves, `artifacts/model/` |
| 6. Oversampling chart | `python results/OSRate.py` | – | oversampling-ratio chart |
| 7. Online validation | `python src/anomaly_detection/API.py` | `artifacts/`, `data/streaming/` | `results/plots_output/` |

Things to know when running the scripts:

- **Figures open in windows.** Each script calls `plt.show()`, so execution pauses until the
  window is closed.
- **Step 5 overwrites the model.** `05_CatShap.py` saves a new `model.cbm`, `threshold.npy` and
  `bg_1000.npy` to `artifacts/model/`, which is what the API loads.
- **Optional parts of step 5.** The Bayesian search (`bayesian_search_params`) and the TreeSHAP
  plots (`shap_explain`) are commented out in `main()`; uncomment them to run them. To train
  on the VAE data instead, change `combined_data_vaegan.csv` to `combined_data_vae.csv`.
- **Oversampling ratio.** `03_GeneratorVAEGAN.py` generates `synthetic_num * 1` samples (100%).
  Change the factor (e.g. `0.3`) to reproduce another ratio.

## Real-time API

```bash
python src/anomaly_detection/API.py              # serve + stream the 10 held-out records
uvicorn anomaly_detection.API:app --port 8000    # serve only (no streaming)
```

When `API.py` is run directly, it starts the server on `http://127.0.0.1:8000`, waits until
`/health` responds, then posts the 10 records in `data/streaming/test_data.csv` to `/predict`,
one every 3 seconds. Host, port and interval are set at the top of `API.py`.

| Endpoint | Description |
|----------|-------------|
| `GET /health` | liveness check |
| `POST /predict` | body `{"line": "<590 comma-separated raw values>"}` (empty fields are allowed) → `{"score": float, "is_anomaly": bool, "plot_file": str \| null}` |
| `GET /plot/{file}` | the SHAP waterfall PNG of a detected anomaly (saved in `results/plots_output/`) |

Invalid input returns HTTP 400, an inference failure HTTP 500. Interactive docs are available at
`http://127.0.0.1:8000/docs`. Example client:

```python
import pandas as pd, requests

row = pd.read_csv("data/streaming/test_data.csv").iloc[-1]          # an anomalous record
line = ",".join("" if pd.isna(v) else str(v) for v in row.drop(["Pass/Fail", "Time"]))
print(requests.post("http://127.0.0.1:8000/predict", json={"line": line}).json())
```

Each streamed record is logged with its true label and the API response:

```
Index 0 (true: normal) -> {'score': ..., 'is_anomaly': False, 'plot_file': None}
...
Index 7 (true: anomaly) -> {'score': ..., 'is_anomaly': True, 'plot_file': '<timestamp>.png'}
Streaming finished: 10/10 records sent. Server is still running, press Ctrl+C to stop.
```

The scores depend on the model in `artifacts/model/`. With the model used in the dissertation,
all 7 normal records passed and 2 of the 3 anomalies were flagged (scores 0.620 and 0.516).

## Implementation notes

The code in this repository has been revised since the dissertation experiments. The points below
explain where it differs from the dissertation text and what to keep in mind when reading it:

- **Generator hyperparameters.**

  | Parameter | Code | Dissertation (Table I) |
  |---|---|---|
  | VAE β | 0.01 (`VAE.train_step`) | 0.1 |
  | VAE-GAN β / γ | 1 / 1 (`VAEGAN.train_step` defaults) | 0.1 / 2 |

  Keras calls `train_step(data)` without extra arguments, so the VAE-GAN β and γ can only be
  changed by editing the defaults in `train_step`.
- **VAE-GAN decoder update.** The decoder step recomputes the discriminator outputs inside its own
  gradient tape, so the adversarial term of Eq. 9 now updates the decoder. In the code used for the
  dissertation results, those outputs were reused from the discriminator step and the decoder was
  trained through the feature-matching loss only. Anomalies generated with the current code
  therefore differ from those behind the reported results.
- **VAE-GAN output scaling.** `VAEGAN.generate()` z-scores each generated feature;
  `VAEGenerator.generate()` does not.
- **CatBoost hyperparameters.** `categoricalboost()` in `05_CatShap.py` trains with
  `max_depth=10, iterations=1000, learning_rate=0.1, l2_leaf_reg=20, min_data_in_leaf=24,
  scale_pos_weight=3` and ordered boosting, from a Bayesian search over max_depth 1–10,
  iterations 10–1000, learning_rate 1e-4–0.1, l2_leaf_reg 1–20 and min_data_in_leaf 1–50.
  The dissertation (Table II) reports `max_depth=4, iterations=73, learning_rate=0.1,
  l2_leaf_reg=12, min_data_in_leaf=6`. The `best_params` argument of `categoricalboost()` is
  currently not used.
- **CatBoost early stopping.** The final model is fitted without an evaluation set, so
  `early_stopping_rounds` has no effect. During the Bayesian search, the early-stopping set is a
  subset of the training fold.
- **Missing values online.** `transform()` fills missing raw values with 0 before scaling, while
  the offline pipeline mean-imputes them.
- **Randomness.** No random seeds are set for TensorFlow or for the SHAP background sample, so
  re-running steps 2–5 gives different data, models and numbers each time. The files committed
  in `data/` and `artifacts/` were regenerated with the current code and differ from the
  dissertation snapshot.

## Limitations and future work

From the dissertation's discussion:

- **Preprocessing** relies on linear correlations and fixed thresholds, so rare but critical
  non-linear features may be discarded. Non-linear reduction methods (e.g. t-SNE, UMAP) are worth exploring.
- **Mode collapse.** VAE-GAN samples tend to concentrate around some real anomaly modes. Lowering
  β (1.0 → 0.1) and raising γ (1 → 2) improves coverage.

  <img src="docs/figures/fig15_mode_collapse.png" width="380" alt="Tendency of VAE-GAN mode collapse">
- **Augmentation vs. detection.** Despite its better sample quality, VAE-GAN improves CatBoost only
  marginally over VAE, so on small datasets a VAE may be sufficient.
- **Oversampling ratio.** 100% gives the best metrics but costs more compute. An adaptive ratio
  could balance efficiency and generalisation.
- **Expert validation.** The dataset is anonymised, so the SHAP explanations could not be checked
  by domain experts.
- **Online miss.** One anomaly was missed online, probably because CatBoost overfits a small
  dataset. More careful regularisation and tuning may help.
- **Next steps.** Diffusion models as generators, and deployment on other production lines to test
  generalisation.

## Citation

```bibtex
@mastersthesis{chen2025interpretable,
  title  = {An Interpretable Pipeline for Imbalanced Industrial Anomaly Detection:
            VAE-GAN Augmentation, CatBoost Classification, and TreeSHAP Interpretation},
  author = {Chen, Dong-Jun},
  school = {University of Bristol and University of the West of England},
  type   = {MSc Robotics Dissertation},
  year   = {2025},
  month  = aug
}
```

## License and acknowledgements

The code is released under the [MIT License](LICENSE). The UCI-SECOM dataset belongs to its
original authors and is distributed by the UCI Machine Learning Repository under CC BY 4.0.

Thanks to my supervisor **Carwyn Ward**, and to the School of Engineering Mathematics and
Technology (University of Bristol) and the School of Engineering (University of the West of England).

Key references: Kingma & Welling (VAE), Larsen et al. 2016 (VAE-GAN), Goodfellow et al. 2014 (GAN),
Prokhorenkova et al. 2018 (CatBoost), Lundberg & Lee 2017 and Lundberg et al. 2018 (SHAP / TreeSHAP).
See the dissertation for the full bibliography.

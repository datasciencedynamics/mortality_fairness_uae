# Kidney Function and Mortality in the United Arab Emirates

Machine learning pipeline for all-cause mortality prediction and sex-stratified
fairness auditing in a public UAE cardiovascular risk cohort (Al-Shamsi et al.,
n = 1,186).

The project trains four classifiers (logistic regression, random forest,
XGBoost, CatBoost), evaluates them under a common protocol, and runs a fairness
audit comparing model behavior across sex — including an ablation that removes
sex from the feature set entirely.

---

## Data

| Item | Value |
|---|---|
| Source | Al-Shamsi et al., public deposit (Mendeley) |
| License | CC BY 4.0 |
| Records | 1,186 patients |
| Raw file | `data/raw/` (Excel) |
| Outcome | All-cause mortality (`outcome`) |

Data is not redistributed in this repository. Download it from the original
deposit and place it in `data/raw/` before running the pipeline.

---

## Requirements

- Python 3.11
- See `requirements.txt`

```bash
make create_venv
source kidney_venv/bin/activate
make requirements
```

Or with conda:

```bash
conda create -n conda_kidney python=3.11
conda activate conda_kidney
make requirements
```

---

## Quick start

```bash
make create_folders          # scaffold data/, models/, notebooks/
make preproc_train_eval      # full pipeline: preprocess -> train -> evaluate
make mlflow_ui               # inspect runs at http://localhost:5501
```

---

## Project layout

```
kidney_uae/
├── core/                        # shared utilities
├── data/
│   ├── raw/                     # source Excel (not tracked)
│   ├── interim/
│   ├── processed/               # parquet artifacts + logs
│   │   └── inference/
│   └── external/
├── modeling/
│   ├── train.py                 # model training + hyperparameter search
│   └── evaluation.py            # held-out evaluation, calibration, metrics
├── preprocessing/
│   ├── data_gen.py              # raw Excel -> df.parquet
│   ├── preprocessing.py         # cleaning, zero-handling
│   └── feat_gen.py              # feature construction -> X.parquet, y.parquet
├── notebooks/
│   ├── Kidney_UAE_Preprocessing.ipynb
│   ├── performance_assessment.ipynb
│   ├── bias_fairness.ipynb
│   └── bias_fairness_sex_ablated.ipynb
├── models/
│   ├── results/<outcome>/       # training logs per model + pipeline
│   └── eval/<outcome>/          # evaluation logs
└── Makefile
```

---

## The Makefile

The Makefile is the orchestration layer. Every stage is a target, targets
compose into pipelines, and all stdout is teed to a log file so runs are
reproducible after the fact.

### Configuration variables

Set at the top of the file:

| Variable | Purpose |
|---|---|
| `PROJECT_NAME` | Project identifier |
| `PYTHON_VERSION` | Interpreter version (3.11) |
| `VENV_DIR` | Virtualenv directory (`kidney_venv`) |
| `CONDA_ENV_NAME` | Conda environment name |
| `RAW_DATA` | Path to source Excel |
| `PROCESSED_DATA` | Path to `df.parquet` |
| `OUTCOMES` | Outcome variables to loop over |
| `PIPELINES` | Sampling variants (`orig`, `smote`, `orig_no_sex`) |
| `SCORING` | Tuning metric (`average_precision`) |
| `PRETRAINED` | `0` = train from scratch, `1` = calibrate an existing model |

`RAW_DATA` and `PROCESSED_DATA` use `?=`, so they can be overridden at the
command line without editing the file:

```bash
make data_gen RAW_DATA=/path/to/other.xlsx
```

### Environment and setup

| Target | Does |
|---|---|
| `init_config` | Interactive rename of project + variables (cross-platform sed) |
| `check_vars` | Prints the variables that still need real values |
| `create_venv` | Builds the virtualenv |
| `activate_venv` | Prints the activation command |
| `clean_venv` | Removes the virtualenv |
| `requirements` | Upgrades pip, installs `requirements.txt` |
| `create_folders` | Scaffolds the full directory tree, per-outcome subdirs |
| `clean` | Deletes `.pyc` files and `__pycache__` |
| `clean_dir` | Removes `data/` entirely |
| `mlflow_ui` | MLflow UI on port 5501, backend `mlruns` |

### Preprocessing

| Target | Input | Output |
|---|---|---|
| `data_gen` | Raw Excel | `data/processed/df.parquet` |
| `data_prep_preprocessing_training` | `df.parquet` | `df_sans_zero.parquet` |
| `feat_gen_training` | `df_sans_zero.parquet` | `X.parquet`, `y.parquet` |
| `preproc_pipeline` | — | All three, in order |
| `clean_data` | — | Removes the processed parquet + csv |

`data_gen` is the one true file-based rule in the Makefile:

```make
$(PROCESSED_DATA): $(RAW_DATA) $(DATA_GEN_SCRIPT)
```

It only re-runs when the raw data or the generation script is newer than the
output. Everything downstream is phony and re-runs on every invocation.

### Training

One target per algorithm, each looping over `OUTCOMES` × `PIPELINES`:

| Target | Model |
|---|---|
| `train_logistic_regression` | `--model-type lr` |
| `train_random_forest` | `--model-type rf` |
| `train_xgboost` | `--model-type xgb` |
| `train_catboost` | `--model-type cat` |
| `train_all_models` | All four |

Logs land in `models/results/<outcome>/<model>_<pipeline>.txt`. When
`PRETRAINED=1`, the suffix `_prefit` is appended so calibration runs don't
overwrite training runs:

```bash
make train_catboost PRETRAINED=1
```

### Evaluation

| Target | Model |
|---|---|
| `eval_logistic_regression` | lr |
| `eval_random_forest` | rf |
| `eval_xgboost` | xgb |
| `eval_catboost` | cat |
| `eval_all_models` | All four |

Logs land in `models/eval/<outcome>/<model>_eval_<pipeline>.txt`.

### Fairness ablation

```make
cat_no_sex:
	$(MAKE) train_catboost PIPELINES=orig_no_sex
	$(MAKE) eval_catboost PIPELINES=orig_no_sex
```

This is the sex-ablated arm of the fairness analysis: CatBoost retrained with
sex removed from the feature set, then evaluated under the same protocol. The
recursive `$(MAKE)` call overrides `PIPELINES` for the sub-invocation only, so
the parent configuration is untouched.

Paired with `notebooks/bias_fairness_sex_ablated.ipynb`, this supports the
counterintuitive result that dropping the sensitive attribute can *widen*
disparity rather than close it.

### Explainability

| Target | Does |
|---|---|
| `model_explainer` | Selects the best model by K-Fold Average Precision |
| `model_explanations_training` | SHAP values on training data, top 5 features |
| `model_explaining_training` | Both, in order |
| `model_explanations_inference` | SHAP values on inference data |

### Inference

| Target | Does |
|---|---|
| `data_prep_preprocessing_inference` | Preprocessing in inference mode |
| `feat_gen_inference` | Feature generation in inference mode |
| `predict` | Best-model predictions to CSV |
| `preproc_pipeline_inf` | All three, in order |

### Composite pipelines

| Target | Chain |
|---|---|
| `preproc_pipeline` | data_gen → preprocessing → feat_gen |
| `train_eval_pipeline` | train_all → eval_all → cat_no_sex |
| `preproc_train_eval` | preproc → train_all → eval_all |
| `preproc_pipeline_inf` | inference preprocessing → feat_gen → predict |

### Help

`.DEFAULT_GOAL := help`, so a bare `make` prints available rules. The help text
is generated by a Python regex over the Makefile itself, picking up any target
preceded by a `##` comment line.

---

## Reproducing the analysis

```bash
make create_folders
make preproc_pipeline
make train_eval_pipeline
```

Then run the notebooks in order:

1. `Kidney_UAE_Preprocessing.ipynb` — cohort construction, EDA
2. `performance_assessment.ipynb` — discrimination, calibration, decision curves
3. `bias_fairness.ipynb` — sex-stratified audit, Chouldechova decomposition
4. `bias_fairness_sex_ablated.ipynb` — ablation comparison

---

## Citation

If you use this code, please cite the source dataset:

> Al-Shamsi S, et al. [dataset]. Mendeley Data. CC BY 4.0.

---

## License

Code released under MIT. Source data is CC BY 4.0 and belongs to its original
authors.
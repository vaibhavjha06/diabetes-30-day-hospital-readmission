# 30-Day Hospital Readmission Prediction

A modular Python machine-learning pipeline for predicting whether a patient will be readmitted within 30 days of discharge. The project emphasizes leakage-aware evaluation by keeping each patient's records within a single train, validation, or test split.

## What it does

- Loads and configures a hospital readmission dataset
- Builds clinical and administrative features, including utilization, diagnoses, laboratory procedures, medications, admission details, and demographic fields
- Imputes missing values, standardizes numeric variables, and one-hot encodes categorical variables
- Uses patient-level grouped splits to create 60% training, 20% validation, and 20% held-out test data
- Compares logistic regression and random forest classifiers
- Selects logistic regression and applies a fixed probability threshold of 0.35 for final 30-day readmission predictions
- Reports average precision and classification metrics for validation and test sets

## Project structure

```text
src/readmission/
├── config.py          # Load YAML configuration
├── data.py            # Load raw data
├── features.py        # Feature selection, target definition, preprocessing
├── splitting.py       # Patient-level train/validation/test splits
├── models.py          # Model pipelines and model fitting
├── predict.py         # Probability scoring and thresholded predictions
├── evaluation.py      # Metrics and reporting
└── train_model.py     # End-to-end training workflow
```

## Modeling approach

The target is a binary indicator of **readmission within 30 days**. Numeric features are median-imputed and standardized; categorical features are imputed with a `Missing` category and one-hot encoded. The pipeline evaluates logistic regression and random forest models on a validation set, then retrains the selected logistic-regression model using the combined development data before one final evaluation on the untouched test set.

A grouped split by patient ID helps prevent the same patient from appearing in both training and evaluation data, reducing the risk of overly optimistic performance estimates.

## Run

Configure paths and model settings in `configs/default.yaml`, then run:

```bash
python -m src.readmission.train_model
```

## Example results

In the current workflow, logistic regression outperformed the random forest on validation average precision. On the held-out test set, the logistic-regression model achieved an average precision of approximately **0.209**. Because 30-day readmission is an imbalanced outcome, the project emphasizes average precision and recall for the readmitted class rather than accuracy alone.

## Tech stack

Python, pandas, NumPy, scikit-learn, YAML configuration, and modular ML pipelines.

## Notes

This repository is intended as an educational and portfolio project. It is not a clinical decision-support tool and should not be used to make patient-care decisions.

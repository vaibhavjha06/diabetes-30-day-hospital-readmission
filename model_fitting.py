import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, precision_score,
                             recall_score, f1_score, PrecisionRecallDisplay,
                             confusion_matrix)


"""
Read in data
"""

df = pd.read_csv(
    "/Users/vaibhavjha/Documents/sklearn/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv",
    na_values='?'
    )

"""
Inspect data
"""

print(df.shape)  # 101,766 x 50
print(df.info())
print(df.head())

# Define types of features
numeric_features = ['time_in_hospital',
                    'num_lab_procedures',
                    'num_procedures',
                    'num_medications',
                    'number_outpatient',
                    'number_emergency',
                    'number_inpatient',
                    'number_diagnoses',
                    ]
categorical_features = ['race',
                        'gender',
                        'age',
                        'weight',
                        'admission_type_id',
                        'discharge_disposition_id',
                        'admission_source_id',
                        'payer_code',
                        'medical_specialty',
                        'max_glu_serum',
                        'A1Cresult',
                        'metformin',
                        'repaglinide',
                        'nateglinide',
                        'chlorpropamide',
                        'glimepiride',
                        'glipizide',
                        'glyburide',
                        'pioglitazone',
                        'rosiglitazone',
                        'insulin',
                        'change',
                        'diabetesMed']

# Define label
label = df['readmitted']

"""
Missingness EDA
"""

missing_features = {}
numeric_missing_found = False

# Identify numeric features with missing values
print("\n---Numeric features---")
for feature in numeric_features:
    n_missing = df[feature].isna().sum()

    if n_missing > 0:
        numeric_missing_found = True
        missing_features[feature] = int(n_missing)
        pct_missing = n_missing / len(df) * 100

        print(
            f"The number of missing values for {feature} is: "
            f"{n_missing:,} ({pct_missing:.2f}%)"
        )

if not numeric_missing_found:
    print("No numeric features have missing values.")

# Identify categorical features with missing values
categorical_missing_found = False

print("\n---Categorical features---")
for feature in categorical_features:
    n_missing = df[feature].isna().sum()

    if n_missing > 0:
        categorical_missing_found = True
        missing_features[feature] = int(n_missing)
        pct_missing = n_missing / len(df) * 100

        print(
            f"The number of missing values for {feature} is: "
            f"{n_missing:,} ({pct_missing:.2f}%)"
        )

if not categorical_missing_found:
    print("No categorical features have missing values.")

print(missing_features)

# Look into features with <60% missingness: race, payer_code, medical_specialty
print(df['race'].value_counts())
print(df['payer_code'].value_counts())
print(df['medical_specialty'].value_counts())

# Look into label
print(label.value_counts())


"""
Label pre-processing
"""
# Redefine label categories
label = label.replace({
    "<30": "Readmitted_within_30_days",
    ">30": "No_30_day_readmission",
    "NO": "No_30_day_readmission"
})
print(label.value_counts())


"""
Missingness pre-processing
"""
# Remove features that are missing > 60% values
numeric_features = [
    feature
    for feature in numeric_features
    if missing_features.get(feature, 0) / len(df) < 0.60
]

categorical_features = [
    feature
    for feature in categorical_features
    if missing_features.get(feature, 0) / len(df) < 0.60
]


"""
60-20-20 train-validation-test split
"""
# Make split by grouping by patient id
feature_columns = numeric_features + categorical_features
X = df[feature_columns].copy()
y = label.copy()
groups = df["patient_nbr"].copy()

# Step 1: 80% development data, 20% untouched test data
outer_splitter = GroupShuffleSplit(n_splits=1,
                                   test_size=0.20,
                                   random_state=22)
dev_idx, test_idx = next(outer_splitter.split(X, y, groups=groups))

X_dev = X.iloc[dev_idx].copy()
X_test = X.iloc[test_idx].copy()

y_dev = y.iloc[dev_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_dev = groups.iloc[dev_idx].copy()
groups_test = groups.iloc[test_idx].copy()

# Step 2: Split the 80% development data into:
# 60% train overall and 20% validation overall
inner_splitter = GroupShuffleSplit(n_splits=1, 
                                   test_size=0.25, 
                                   random_state=22)
train_idx, val_idx = next(inner_splitter.split(X_dev, y_dev, groups=groups_dev))

X_train = X_dev.iloc[train_idx].copy()
X_val = X_dev.iloc[val_idx].copy()

y_train = y_dev.iloc[train_idx].copy()
y_val = y_dev.iloc[val_idx].copy()


"""
Model pre-processing
"""
# Build transformers
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(
            strategy="median"
        )),
        ('scaler', StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(
            strategy="constant",
            fill_value="Missing"
        )),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# Pull features into feature matrix
preprocessor = ColumnTransformer(
    transformers=[('numeric', numeric_transformer, numeric_features),
                  ('categorical', categorical_transformer, categorical_features)]
)


"""
Model fitting
"""
baseline_model = LogisticRegression(max_iter=2000,
                                    class_weight="balanced",
                                    solver="saga",
                                    random_state=22)
second_model = RandomForestClassifier(n_estimators=200,
                                      class_weight="balanced",
                                      random_state=22,
                                      n_jobs=1)

logistic_regression = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('logistic_regression', baseline_model)
    ]
)
random_forest = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('rf', second_model)
    ]
)

logistic_regression.fit(X_train, y_train)
random_forest.fit(X_train, y_train)


"""
Model predictions with validation set
"""
positive_class = "Readmitted_within_30_days"

lr_pos_idx = list(logistic_regression.classes_).index(positive_class)
rf_pos_idx = list(random_forest.classes_).index(positive_class)

lr_pred = logistic_regression.predict(X_val)
lr_prob = logistic_regression.predict_proba(X_val)[:, lr_pos_idx]

rf_pred = random_forest.predict(X_val)
rf_prob = random_forest.predict_proba(X_val)[:, rf_pos_idx]

y_val_binary = (y_val == positive_class).astype(int)


"""
Model evaluation to identify best model
"""
lr_ap = average_precision_score(y_val_binary, lr_prob)
rf_ap = average_precision_score(y_val_binary, rf_prob)

print(f"\nLogistic regression ROC-AUC: {roc_auc_score(y_val_binary, lr_prob):.3f}")
print(f"Logistic regression PR-AUC: {lr_ap:.3f}")
print(f"Random forest ROC-AUC: {roc_auc_score(y_val_binary, rf_prob):.3f}")
print(f"Random forest PR-AUC: {rf_ap:.3f}")
print("\n---Logistic Regression---")
print(classification_report(y_val, lr_pred))
print("\n---Random Forest---")
print(classification_report(y_val, rf_pred))


"""
Plot precision-recall AUC curve
"""
no_skill_baseline = y_val_binary.mean()
fig, ax = plt.subplots(figsize=(8, 6))

PrecisionRecallDisplay.from_predictions(
    y_val_binary,
    lr_prob,
    name=f"Logistic Regression (AP = {lr_ap:.3f})",
    ax=ax
)

PrecisionRecallDisplay.from_predictions(
    y_val_binary,
    rf_prob,
    name=f"Random Forest (AP = {rf_ap:.3f})",
    ax=ax
)

ax.axhline(
    y=no_skill_baseline,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=f"No-skill baseline = {no_skill_baseline:.3f}"
)

ax.set_title("Precision–Recall Curves: 30-Day Readmission")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="best")

plt.tight_layout()
plt.show()


"""
Tuning threshold for logistic regression model using validation set
"""
thresholds = np.arange(0.05, 0.51, 0.05)

for threshold in thresholds:
    pred = np.where(
        lr_prob >= threshold,
        "Readmitted_within_30_days",
        "No_30_day_readmission"
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision_score(y_val, pred, pos_label='Readmitted_within_30_days'):.3f} | "
        f"Recall: {recall_score(y_val, pred, pos_label='Readmitted_within_30_days'):.3f} | "
        f"F1: {f1_score(y_val, pred, pos_label='Readmitted_within_30_days'):.3f}"
    )


"""
Logistic regression ROC-AUC: 0.668
Logistic regression PR-AUC: 0.222

Random forest ROC-AUC: 0.648
Random forest PR-AUC: 0.207

---Logistic Regression---
                           precision    recall  f1-score   support

    No_30_day_readmission       0.92      0.68      0.78     17903
Readmitted_within_30_days       0.18      0.56      0.28      2309

                 accuracy                           0.67     20212
                macro avg       0.55      0.62      0.53     20212
             weighted avg       0.84      0.67      0.73     20212


---Random Forest---
                           precision    recall  f1-score   support

    No_30_day_readmission       0.89      1.00      0.94     17903
Readmitted_within_30_days       0.83      0.00      0.01      2309

                 accuracy                           0.89     20212
                macro avg       0.86      0.50      0.47     20212
             weighted avg       0.88      0.89      0.83     20212

Threshold: 0.05 | Precision: 0.116 | Recall: 1.000 | F1: 0.208
Threshold: 0.10 | Precision: 0.116 | Recall: 1.000 | F1: 0.209
Threshold: 0.15 | Precision: 0.117 | Recall: 0.998 | F1: 0.209
Threshold: 0.20 | Precision: 0.117 | Recall: 0.995 | F1: 0.209
Threshold: 0.25 | Precision: 0.118 | Recall: 0.987 | F1: 0.211
Threshold: 0.30 | Precision: 0.121 | Recall: 0.968 | F1: 0.215
Threshold: 0.35 | Precision: 0.128 | Recall: 0.913 | F1: 0.225
Threshold: 0.40 | Precision: 0.144 | Recall: 0.827 | F1: 0.245
Threshold: 0.45 | Precision: 0.164 | Recall: 0.700 | F1: 0.265
Threshold: 0.50 | Precision: 0.184 | Recall: 0.559 | F1: 0.277
"""


"""
Evaluate logistic regression model with a threshold of 0.35 and test set
"""
# 1. Lock the threshold selected on validation data
chosen_threshold = 0.35
positive_class = "Readmitted_within_30_days"
negative_class = "No_30_day_readmission"

# 2. Refit the chosen model on ALL development data:
#    training + validation, but never test data
logistic_regression.fit(X_dev, y_dev)

# 3. Identify the probability column for the positive class explicitly
test_positive_idx = list(logistic_regression.classes_).index(positive_class)

# 4. Obtain predicted probability of 30-day readmission
test_prob = logistic_regression.predict_proba(X_test)[:, test_positive_idx]

# 5. Apply the locked threshold
test_pred = np.where(
    test_prob >= chosen_threshold,
    positive_class,
    negative_class
)

# Binary true test outcome for ROC-AUC and AP
y_test_binary = (y_test == positive_class).astype(int)

# 6. Threshold-independent ranking metrics
print("\n--- Test performance: Logistic Regression ---")
print(f"Threshold: {chosen_threshold:.2f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test_binary, test_prob):.3f}")
print(
    "Test Average Precision / PR-AUC: "
    f"{average_precision_score(y_test_binary, test_prob):.3f}"
)

# 7. Threshold-dependent classification metrics
print("\nClassification report at threshold = 0.35:")
print(
    classification_report(
        y_test,
        test_pred,
        labels=[negative_class, positive_class],
        zero_division=0,
        digits=3
    )
)

test_precision = precision_score(
    y_test,
    test_pred,
    pos_label=positive_class,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_pred,
    pos_label=positive_class,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_pred,
    pos_label=positive_class,
    zero_division=0
)

print(f"Positive-class precision: {test_precision:.3f}")
print(f"Positive-class recall:    {test_recall:.3f}")
print(f"Positive-class F1:        {test_f1:.3f}")
print(f"Patients flagged:         {(test_pred == positive_class).sum():,}")
print(f"Flag rate:                {(test_pred == positive_class).mean():.1%}")

# 8. Confusion matrix, with labels fixed in a known order
cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[negative_class, positive_class]
)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion matrix:")
print(cm)
print(f"TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")


"""
--- Test performance: Logistic Regression ---
Threshold: 0.35
Test ROC-AUC: 0.667
Test Average Precision / PR-AUC: 0.209

Classification report at threshold = 0.35:
                           precision    recall  f1-score   support

    No_30_day_readmission      0.955     0.194     0.322     17997
Readmitted_within_30_days      0.125     0.926     0.220      2239

                 accuracy                          0.275     20236
                macro avg      0.540     0.560     0.271     20236
             weighted avg      0.863     0.275     0.311     20236

Positive-class precision: 0.125
Positive-class recall:    0.926
Positive-class F1:        0.220
Patients flagged:         16,586
Flag rate:                82.0%

Confusion matrix:
[[ 3485 14512]
 [  165  2074]]
TN=3,485, FP=14,512, FN=165, TP=2,074
"""

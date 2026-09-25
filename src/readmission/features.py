from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.compose import ColumnTransformer


"""
Pull features
"""


def get_feature_columns(df, config):
    numeric_features = ['time_in_hospital',
                        'num_lab_procedures',
                        'num_procedures',
                        'num_medications',
                        'number_outpatient',
                        'number_emergency',
                        'number_inpatient',
                        'number_diagnoses']
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
    candidate_features = numeric_features + categorical_features
    # Define threshold for missing values
    max_missing_fraction = config["features"]["max_missing_fraction"]

    # Get missingness proportions for features
    missing_fraction = df[candidate_features].isna().mean()

    # Define features based on missingness condition
    numeric_features = [
        feature
        for feature in numeric_features
        if missing_fraction[feature] < max_missing_fraction
    ]
    categorical_features = [
        feature
        for feature in categorical_features
        if missing_fraction[feature] < max_missing_fraction
    ]

    # Define feature matrix and groups
    feature_columns = numeric_features + categorical_features
    X = df[feature_columns].copy()
    groups = df[config["data"]["patient_id_column"]].copy()

    return X, groups, numeric_features, categorical_features


"""
Define label
"""


def define_y(df, config):
    y = df[config["features"]["target_column"]]

    # Redefine label categories
    y = y.replace({
        "<30": "Readmitted_within_30_days",
        ">30": "No_30_day_readmission",
        "NO": "No_30_day_readmission"
    })

    return y


"""
Build preprocessor
"""


def build_preprocessor(numeric_features, categorical_features):
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

    return preprocessor


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


"""
Fit specified model
"""


def build_pipeline(preprocessor, config):
    # Logistic regression
    lr_model = LogisticRegression(max_iter=config["model"]["logistic_regression"]["max_iter"],
                                  class_weight=config["model"]["logistic_regression"]["class_weight"],
                                  solver=config["model"]["logistic_regression"]["solver"],
                                  random_state=config["split"]["random_state"])

    logistic_regression = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('logistic_regression', lr_model)
        ]
    )

    # Random forest
    rf_model = RandomForestClassifier(n_estimators=config["model"]["random_forest"]["n_estimators"],
                                      class_weight=config["model"]["random_forest"]["class_weight"],
                                      random_state=config["split"]["random_state"],
                                      n_jobs=config["model"]["random_forest"]["n_jobs"])

    random_forest = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('rf', rf_model)
        ]
    )

    return logistic_regression, random_forest


"""
Fit models
"""


def fit_models(preprocessor,
               config,
               X_train,
               y_train):
    # Fit models
    logistic_regression, random_forest = build_pipeline(preprocessor, config)

    fitted_lr = logistic_regression.fit(X_train, y_train)
    fitted_rf = random_forest.fit(X_train, y_train)

    return fitted_lr, fitted_rf


"""
Fit final model of choice with development data (training + validation)
Use threshold = 0.35
"""


def fit_final_model(preprocessor,
                    X_dev,
                    y_dev,
                    config):
    # Fit models
    logistic_regression, random_forest = build_pipeline(preprocessor, config)
    final_model = logistic_regression.fit(X_dev, y_dev)

    return final_model

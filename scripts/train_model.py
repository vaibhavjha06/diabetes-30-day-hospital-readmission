from src.readmission.config import load_config
from src.readmission.data import load_raw_data
from src.readmission.features import (get_feature_columns,
                                      define_y,
                                      build_preprocessor)
from src.readmission.splitting import split_by_patient
from src.readmission.models import (fit_models,
                                    fit_final_model)
from src.readmission.predict import (validation_predictions,
                                     test_predictions)
from src.readmission.evaluation import (val_evaluate_model,
                                        test_evaluate_model,
                                        print_results)

"""
Script for workflow
"""


def main():
    # Call on configuration
    config = load_config("configs/default.yaml")

    # Load data
    df = load_raw_data(config)

    # Pull features
    X, groups, numeric_features, categorical_features = get_feature_columns(df, config)

    # Define label
    y = define_y(df, config)

    # Split data into train/validation/test, 60-20-20
    (X_train, X_val, X_dev, X_test,
        y_train, y_val, y_dev, y_test,
        groups_dev, groups_test) = split_by_patient(X, y, groups, config)

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Fit initial models
    fitted_lr, fitted_rf = fit_models(preprocessor, config, X_train, y_train)

    # Make predictions with validation set
    lr_pred, lr_prob, rf_pred, rf_prob = validation_predictions(fitted_lr,
                                                                fitted_rf,
                                                                X_val,
                                                                config)

    # Output classification reports
    results = val_evaluate_model(y_val, lr_pred, lr_prob, rf_pred, rf_prob, config)

    # Fit final model of choice using development data (training + validation)
    final_model = fit_final_model(preprocessor, X_dev, y_dev, config)

    # Make predictions with test set
    test_pred, test_prob = test_predictions(final_model, X_test, config)

    # Output classification report
    final_results = test_evaluate_model(y_test, test_pred, test_prob, config)

    # Return points of interest
    return {
        'message': 'The better performing model was the logistic regression model.',
        'validation_results': results,
        'test_results': final_results
    }


"""
Run it!
"""


if __name__ == "__main__":
    outputs = main()
    config = load_config("configs/default.yaml")
    positive_class = config["evaluation"]["positive_class"]

    print_results(
        "VALIDATION RESULTS",
        outputs["validation_results"],
        "Validation set average_precision",
        "Validation set classification_report",
        positive_class,
    )

    print(f"\n{outputs['message']}")

    print_results(
        "FINAL TEST RESULTS",
        outputs["test_results"],
        "Test set average_precision",
        "Test set classification_report",
        positive_class,
    )


"""
VALIDATION RESULTS
==================

Logistic Regression
  Average precision:     0.222
  Readmission precision: 18.4%
  Readmission recall:    55.7%
  Readmission F1:        0.277
  Readmission cases:     1,844
  Overall accuracy:      66.8%

Random Forest
  Average precision:     0.204
  Readmission precision: 54.5%
  Readmission recall:    0.3%
  Readmission F1:        0.006
  Readmission cases:     1,844
  Overall accuracy:      88.6%

The better performing model was the logistic regression model.

FINAL TEST RESULTS
==================

Logistic Regression
  Average precision:     0.209
  Readmission precision: 12.5%
  Readmission recall:    92.7%
  Readmission F1:        0.220
  Readmission cases:     2,239
  Overall accuracy:      27.4%
"""

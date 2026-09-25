import numpy as np

"""
Use models to make predictions with validation set
"""


def validation_predictions(fitted_lr, fitted_rf, X_val, config):
    positive_class = config["evaluation"]["positive_class"]

    lr_pos_idx = list(fitted_lr.classes_).index(positive_class)
    rf_pos_idx = list(fitted_rf.classes_).index(positive_class)

    lr_pred = fitted_lr.predict(X_val)
    lr_prob = fitted_lr.predict_proba(X_val)[:, lr_pos_idx]

    rf_pred = fitted_rf.predict(X_val)
    rf_prob = fitted_rf.predict_proba(X_val)[:, rf_pos_idx]

    return lr_pred, lr_prob, rf_pred, rf_prob


"""
Use chosen final model to make predictions for test set
"""


def test_predictions(final_model, X_test, config):
    positive_class = config["evaluation"]["positive_class"]
    chosen_threshold = config["evaluation"]["locked_threshold"]
    negative_class = config["evaluation"]["negative_class"]

    # Identify the probability column for the positive class explicitly
    test_positive_idx = list(final_model.classes_).index(positive_class)
    # Obtain predicted probability of 30-day readmission
    test_prob = final_model.predict_proba(X_test)[:, test_positive_idx]
    # Apply the locked threshold
    test_pred = np.where(
        test_prob >= chosen_threshold,
        positive_class,
        negative_class
    )

    return test_pred, test_prob

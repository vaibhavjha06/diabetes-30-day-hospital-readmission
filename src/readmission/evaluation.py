from sklearn.metrics import (average_precision_score, classification_report)


"""
Model evaluations with validation set
"""


def val_evaluate_model(y_val, lr_pred, lr_prob, rf_pred, rf_prob, config):
    y_val_binary = (y_val == config["evaluation"]["positive_class"]).astype(int)

    # Return results
    results = {
        "logistic_regression": {
            "Validation set average_precision": average_precision_score(y_val_binary, lr_prob),
            "Validation set classification_report": classification_report(
                y_val, lr_pred, output_dict=True
            ),
        },
        "random_forest": {
            "Validation set average_precision": average_precision_score(y_val_binary, rf_prob),
            "Validation set classification_report": classification_report(
                y_val, rf_pred, output_dict=True
            ),
        },
    }

    return results


"""
Model evaluations with test set
"""


def test_evaluate_model(y_test, test_pred, test_prob, config):
    y_val_binary = (y_test == config["evaluation"]["positive_class"]).astype(int)

    # Return results
    final_results = {
        "logistic_regression": {
            "Test set average_precision": average_precision_score(y_val_binary, test_prob),
            "Test set classification_report": classification_report(
                y_test, test_pred, output_dict=True
            ),
        }
    }

    return final_results


"""
Make output from CLI more user-friendly
"""


def print_results(title, results, score_key, report_key, positive_class):
    print(f"\n{title}")
    print("=" * len(title))

    for model_name, metrics in results.items():
        report = metrics[report_key]
        readmission = report[positive_class]

        print(f"\n{model_name.replace('_', ' ').title()}")
        print(f"  Average precision:     {metrics[score_key]:.3f}")
        print(f"  Readmission precision: {readmission['precision']:.1%}")
        print(f"  Readmission recall:    {readmission['recall']:.1%}")
        print(f"  Readmission F1:        {readmission['f1-score']:.3f}")
        print(f"  Readmission cases:     {int(readmission['support']):,}")
        print(f"  Overall accuracy:      {report['accuracy']:.1%}")
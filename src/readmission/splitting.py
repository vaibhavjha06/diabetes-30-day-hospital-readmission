from sklearn.model_selection import GroupShuffleSplit

"""
Define train-validation-test split of data
"""


def split_by_patient(X, y, groups, config):

    # Define configuration parameters
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]

    # Step 1: 80% development data, 20% untouched test data
    outer_splitter = GroupShuffleSplit(n_splits=1,
                                       test_size=test_size,
                                       random_state=random_state)
    # Identify where to get development and test data
    dev_idx, test_idx = next(outer_splitter.split(X, y, groups=groups))
    # Pull development and test data
    X_dev = X.iloc[dev_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_dev = y.iloc[dev_idx].copy()
    y_test = y.iloc[test_idx].copy()
    # Pull group data
    groups_dev = groups.iloc[dev_idx].copy()
    groups_test = groups.iloc[test_idx].copy()

    # Step 2: Split the 80% development data into:
    # 60% train overall and 20% validation overall
    inner_splitter = GroupShuffleSplit(n_splits=1,
                                       test_size=test_size,
                                       random_state=random_state)
    # Identify where to get development and test data
    train_idx, val_idx = next(inner_splitter.split(X_dev, y_dev, groups=groups_dev))
    # Pull train and validation data
    X_train = X_dev.iloc[train_idx].copy()
    X_val = X_dev.iloc[val_idx].copy()
    y_train = y_dev.iloc[train_idx].copy()
    y_val = y_dev.iloc[val_idx].copy()

    return X_train, X_val, X_dev, X_test, y_train, y_val, y_dev, y_test, groups_dev, groups_test

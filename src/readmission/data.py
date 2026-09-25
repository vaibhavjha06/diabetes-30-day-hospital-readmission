import pandas as pd


"""
Load data
"""


def load_raw_data(config):
    df = pd.read_csv(config["data"]["path"],
                     na_values='?'
                     )
    return df

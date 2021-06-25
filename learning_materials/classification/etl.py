"""
This script it used to load the extract, transform and load the raw data into the desired subset (if applicable.)

Usage:
    python3 ./src/data/etl.py

"""


import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.preprocessing import remove_duplicate_info, simplify_value
from src.utilities.utility import parse_config, set_logger, load_data


@click.command()
@click.argument("config_file", type=str, default="./src/config.yml")
def etl(config_file):
    """ETL function that loads raw data and select appropriate subset, while generating train and test set.

    Args:
        config_file ([str]): Path to the config file.

    Returns:
        None
    """

    # configure logger
    logger = set_logger("../log/etl.log")

    # load config from config file
    logger.info(f"Load config from {config_file}.")
    config = parse_config(config_file)

    raw_data_file = config["etl"]["raw_data_file"]
    processed_path = Path(config["etl"]["processed_path"])
    test_size = config["etl"]["test_size"]
    random_state = config["etl"]["random_state"]
    logger.info(f"Config from {config['etl']}.")

    # data transformation
    logger.info("-------------------Start ETL Process-------------------")
    features, target = load_data(raw_data_file, "Churn Label")

    # encode target values
    target = target.map({"Yes": 1, "No": 0})

    # selecting subset of cols
    cols_to_keep = [
        "CustomerID",
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
    ]

    features = features[cols_to_keep]
    features.columns = [("").join(col.split(" ")) for col in features.columns]

    # data wrangling
    features = remove_duplicate_info(features)  # replace duplicated info
    features = simplify_value(features)  # simplify column values
    features = features[features["TotalCharges"].notnull()]
    logger.info("End ETL Process")

    # concat features and targets
    data = pd.concat([features, target], axis=1)

    # generating train/test data
    logger.info("-------------------Train test split & Export-------------------")
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    # exporting data
    logger.info(f"Write data to {processed_path}")
    train.to_csv(processed_path / "train.csv", index=False)
    test.to_csv(processed_path / "test.csv", index=False)
    logger.info(f"Train: {train.shape}")
    logger.info(f"Test: {test.shape}")
    logger.info("\n")


if __name__ == "__main__":
    etl()

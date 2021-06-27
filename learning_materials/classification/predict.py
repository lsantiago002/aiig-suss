"""
This script is used to do prediction based on trained model

Usage:
    python3 predict.py
"""

import logging
from pathlib import Path
from datetime import datetime

from pickle import load

import click
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.features.preprocessing import make_pipeline, create_sampler, retrieve_columns
from src.utilities.utility import parse_config, load_data, set_logger


@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def predict(config_file):
    """
    Main function that runs predictions.

    Args:
        config_file [str]: path to config file

    Returns:
        None
    """

    # configure logger
    logger = set_logger(f"log/predict-{datetime.today().strftime('%b-%d-%Y')}.log")

    # load config from config file
    logger.info(f"Load config from {config_file}.")
    config = parse_config(config_file)

    model_path = Path(config["predict"]["model_path"])
    transformer_path = Path(config["train"]["transformer_path"])
    processed_test = config["predict"]["processed_test"]
    predicted_file = config["predict"]["predicted_file"]
    export_result = config["predict"]["export_result"]

    logger.info(f"Config: {config['predict']}")

    # load model and test file
    logger.info(f"-------------------Load the trained model-------------------")
    with open(model_path, "rb") as f:
        trained_model = load(f)

    logger.info(f"Load the test data from {processed_test}.")
    X, y = load_data(processed_test, "Churn Label", type="csv")
    logger.info(f"Cols: {list(X.columns)}")
    logger.info(f"X: {X.shape}")
    logger.info(f"y: {y.shape}")

    # transform test dataset
    _, _, _, X = retrieve_columns(X)
    with open(transformer_path, "rb") as f:
        transformer = load(f)
    preprocess_data = transformer.transform(X)

    # make prediction and evaluate
    logger.info(f"-------------------Predict and evaluate-------------------")
    y_pred = trained_model.predict(preprocess_data)
    logger.info(
        f"Classification report: \n {classification_report(y, y_pred, digits=5)}"
    )
    output = pd.DataFrame(y)
    output["prediction"] = y_pred
    if export_result:
        output.to_csv(predicted_file, index=False)
        logger.info(f"Export prediction to : {predicted_file}")


if __name__ == "__main__":
    predict()

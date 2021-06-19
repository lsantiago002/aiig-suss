import numpy as np
import pandas as pd

from src.features import preprocessing
from src.models import train_model
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


def main():

    # Retrieve data
    file_path = 'data/interim/train_interim.csv'
    housing_prices = pd.read_csv(file_path)

    # Seperating predictors and target
    input_feats, output_feats = preprocessing.make_dataset(housing_prices, 'SalePrice')

    # Subsetting columns of interest
    feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
                     'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'HouseStyle']
    features = input_feats[feature_names]

    # Data processing
    preprocess_pipeline = preprocessing.preprocess_pipeline(features)

    # Generating pipeline for model
    model = make_pipeline(
        preprocess_pipeline,
        LinearRegression()
    )

    # Train the model
    model, predictions, actual = train_model.train_model(features, output_feats, model)

    # Evaluating the model
    train_model.evaluate_model(predictions, actual, model)


if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd

from src.features import preprocessing
from src.models import train_model
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


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
    pipeline = make_pipeline(
        preprocess_pipeline,
        KNeighborsRegressor()
    )

    # Defining a params for grid-search
    params = {'kneighborsregressor__n_neighbors': range(2, 21),
              'kneighborsregressor__weights': ['uniform', 'distance']}

    model = GridSearchCV(pipeline, params, cv=10, scoring='neg_mean_squared_error')

    # Train the model
    model, predictions, actual = train_model.train_model(features, output_feats, model)
    # check the best parameters that was chosen
    print(f"Best parameters chosen: {model.best_params_}")
    # Evaluating the model
    train_model.evaluate_model(predictions, actual, model)


if __name__ == '__main__':
    main()
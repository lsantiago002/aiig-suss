import datetime as datetime
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class ImputeNans(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        categorical_columns = [col for col in X.columns if X.dtypes[col] == 'object']
        numerical_columns = [col for col in X.columns if col not in categorical_columns]

        for col in categorical_columns + numerical_columns:
            if col in categorical_columns:
                imputer = X[col].dropna().mode()[0]
            if col in numerical_columns:
                imputer = X[col].dropna().median()
            X[col] = X[col].fillna(imputer)

        return X


class GetAge(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        current_year = int(datetime.datetime.now().year)

        X['Age'] = current_year - X['YearBuilt']
        X.drop(['YearBuilt'], axis=1 , inplace=True)

        return X

def make_dataset(df, y):
    output_feature = df[y]
    input_features = df.drop(y, axis=1)

    return input_features, output_feature


def preprocess_pipeline(X):
    categorical_columns = [col for col in X.columns if X.dtypes[col] == 'object']
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    return make_column_transformer(
        (make_pipeline(ImputeNans(), GetAge(), MinMaxScaler()), numerical_columns),
        (make_pipeline(ImputeNans(), OneHotEncoder()), categorical_columns)
    )
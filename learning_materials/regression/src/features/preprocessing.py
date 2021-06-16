import datetime as datetime
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class ImputeNans(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        categorical_columns = [col for col in X.columns if X.dtypes[col] == 'object']
        numerical_columns = [col for col in X.columns if col not in categorical_columns]
        # sanity check: categorical + numerical = total
        assert(len(categorical_columns) + len(numerical_columns) == len(X)))

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
"""
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import RandomOverSampler, SMOTE


def remove_duplicate_info(data):
    """Convert duplicates information that can be found in other columns, into 'No'.

    Args:
        data ([pd.DataFrame]): Dataframe to be wrangled.
    """

    # replace 'No phone service'
    data["MultipleLines"] = data["MultipleLines"].replace({"No phone service": "No"})

    # replace 'No internet service'
    for col in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        data[col] = data[col].replace({"No internet service": "No"})

    return data


def simplify_value(data):
    """Simply values in columns that is made up of phrases.
    E.g., `Fiber Optic` -> `FiberOptic`

    Args:
        data ([pd.DataFrame]): Dataframe to be wrangled.
    """

    # Simplify the values made up of phrases
    data["PaymentMethod"] = data["PaymentMethod"].replace(
        {
            "Bank transfer (automatic)": "transfer",
            "Credit card (automatic)": "creditcard",
            "Electronic check": "echeck",
            "Mailed check": "mcheck",
        }
    )

    data["InternetService"] = data["InternetService"].replace(
        {"Fiber optic": "FiberOptic"}
    )

    data["Contract"] = data["Contract"].replace(
        {"Month-to-month": "M2M", "One year": "OneYear", "Two year": "TwoYear"}
    )

    return data


def retrieve_columns(X):
    """Retrieve list of columns by data types.

    Args:
        X ([pd.DataFrame]): DataFrame.
    """

    num_features = [
        key for key in dict(X.dtypes) if dict(X.dtypes)[key] in ["int64", "float64"]
    ]
    cat_features = ["Gender", "InternetService", "Contract", "PaymentMethod"]
    bin_features = [col for col in X.columns if col not in cat_features + num_features]

    # NOTE: refactor when necessarily to shift encoding to its individual function
    for col in bin_features:
        X[col] = X[col].map({"Yes": 1, "No": 0})

    return num_features, cat_features, bin_features, X


def create_sampler(type, config):

    if type == "SMOTE":
        return SMOTE(**config)
    else:
        return RandomOverSampler(**config)

"""
"""

import numpy as np
import pandas as pd


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

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error


def train_model(predictors, target, model_class):

    X_train, X_test, y_train, y_test = train_test_split(
        predictors, target, test_size=0.2, random_state=42
    )

    model_class.fit(X_train, y_train)
    y_pred = model_class.predict(X_test)

    # Cross-valiation score
    print("Training model...")

    return model_class, y_pred, y_test


def evaluate_model(predictions, actual, model_class):

    # Generating evaluation metrics
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    print("Results on Test Data")
    print("====================")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.5f}")

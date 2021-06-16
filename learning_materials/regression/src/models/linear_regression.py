# AI Singapore
# Regression 2 Exercise
# Exercise: Building a Regression job template

# 1. Import required libraries
import numpy as np
import pandas as pd
import datetime as d

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import joblib

# Information on Data
# https://www.kaggle.com/c/home-data-for-ml-course/data

# Custom Classes and Functions
def display_df_info(df, df_name, verbose=False):
    """Convenience function to display information about a dataframe

        Args:
            df [pd.DataFrame]: DataFrame for display
        Returns:
            [None]: DataFrame's Summary Information
    """

    print(f"Dataset name {df_name}")
    print(f"Shape {df.shape[0]}, {df.shape[1]}")
    print("Printing first five rows...")
    print(df.head())

    # Optional: Display other optional information with the verbose flag
    if verbose:
        print("\nDataframe Info:")
        print(df.info())

class GetAge(BaseEstimator, TransformerMixin):
    """Custom Transformer: Calculate age (years only) relative to current year.

        Note that the col values will be replaced but the original col name remains.
        When the transformer is used in a pipeline, this is not an issue as the names are not used.
        However, if the data from the pipeline is to be converted back to a DataFrame, then the col name change should 
        be done to reflect the correct data content.
    """

    def fit(self, X, y=None):
        return self

    def transform(self,X):
        current_year = int(d.datetime.now().year)

        """TASK: Replace the 'YearBuilt' column values with the calculated age (subtract the
        current year from the original values).
        """

        X['YearBuilt'] = current_year - X['YearBuilt']

        return X

def main():

    # DATA INPUT
    ############
    file_path = "../../data/interim/train_eda.csv" #TASK: Modify to path of file
    input_data = pd.read_csv(file_path)
    display_df_info(input_data, "Interim Input")

    # Seperate out the outcome variable from the loaded dataframe
    output_var_name = 'SalePrice'
    output_var = input_data[output_var_name]
    input_data.drop(output_var_name, axis=1, inplace=True)

    # DATA ENGINEERING / MODEL DEFINITION
    #####################################

    # Subsetting the columns: define features to keep
    feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
                     'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'HouseStyle']# TASK: Define the names of the columns to keep
    features = input_data[feature_names]
    display_df_info(features, 'Features before Transform')

    # Create the pipeline ...
    # 1. Pre-processing
    # Define variables made up of lists. Each list is a set of columns that will go through the same data transformations.
    numerical_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'] # TASK: Define numerical column names
    categorical_features = ['HouseStyle'] # TASK: Define categorical column names

    """TASK:
    Define the data processing steps (transformers) to be applied to the numerical features in the dataset.

    At a minimum, use 2 transformers: GetAge() and one other. Combine them using make_pipeline() or Pipeline()
    """
    preprocess = make_column_transformer(
        (make_pipeline(GetAge(), SimpleImputer(), MinMaxScaler()), numerical_features),
        (OneHotEncoder(), categorical_features))

    # 2. Combine pre-processing with ML algorithm
    model = make_pipeline(
        preprocess,
        LinearRegression())

    # TRAINING
    ##########
    # Train/Test Split
    """TASK:
    Split the data in test and train sets by completing the train_test_split function below. Define a random_state value so that
    the experiment is repeatable.
    """
    x_train, x_test, y_train, y_test = train_test_split(features, output_var, test_size=0.2, random_state=42)

    # Train the pipeline
    model.fit(x_train, y_train)

    # Optional: Train with cross-validation and/or parameter grid search

    # SCORING/EVALUATION
    ####################
    # Fit the model on the test data
    y_pred = model.predict(x_test)

    # Display the results of the metrics
    """TASK:
    Calculate the RMSE and Coeff of Determination between the actual and predicted sale prices.
    Name your variables rmse and r2 respectively.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(model, x_train, y_train, scoring='r2', cv=10)
    print("Results on Test Data")
    print("####################")
    print(f"CV R2 Score: {np.mean(cv):.2f}")
    print("RMSE: {:.2f}".format(rmse))
    print("R2 Score: {:.5f}".format(r2))

    # Compare actual vs predicted values
    """TASK:
    Create a new dataframe which combines the actual and predicted Sale Prices from the test dataset. You
    may also add columns with other information such as difference, abs diff, %tage difference etc.

    Name your variable compare
    """
    compare = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Diffence': y_test - y_pred
    })
    display_df_info(compare, 'Actual vs Predicted Comparison')

    # Save the model
    with open('../../models/my_model_lr.joblib', 'wb') as fo:
        joblib.dump(model, fo)


if __name__ == '__main__':
    main()

classification
==============================

Project Repo containing code for [SUP-4: Classification](https://learn.aisingapore.org/courses/ai-for-industry-part-2/lessons/sup-4-classification/).

In this lesson, you will continue your knowledge about Classification by applying to a new dataset, the *Telco Churn* dataset where you will attempt to predict whether if a customer will [churn](https://www.europeanbusinessreview.com/how-costly-is-customer-churn-in-the-telecom-industry/).

At the end of this lesson, you should not only have a better grasp of key concepts revolving around classification problems, some key algorithms like `K-Nearest Neighbors`, `Decision Trees`, `Logistic Regression`, and even advanced ensemble algorithms like, `Xgboost`. You will also have the experience of dealing with [imbalance](https://machinelearningmastery.com/what-is-imbalanced-classification/) dataset.

Overview
------------
During this course, you will be exposed to the following:

1. Practise conducting EDA, extending your experience and your knowledge acquired in [SUP-3: Regression](https://github.com/wtlow003/aiig-suss/tree/main/learning_materials/regression).
2. You will learn how to build simple classification models from `K-Nearest Neighbors` to self-customized `Ensemble` models.
3. Learning to deal with imbalance dataset through sampling methods such as `Random Oversampling` and `Synthetic Minority Oversampling Technique (SMOTE)`.
4. Tuning your model to achieve better performance using `Grid Search`.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `01-lwt-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── preprocessing.py
    │   │
    │   │
    │   └── utilities  <- Scripts used for miscellaneous purposes in running the project
    │       └── utility.py
    │
    ├── etl.py           <- Run the extraction, transformation and loading of raw data for modeling
    │
    ├── train.py         <- Train model using preprocessed dataset
    │
    └── predict.py       <- Run predictions using trained model generated from `train.py`


Setup
------------
Before you get started, first clone the repo:
```bash
git clone https://github.com/wtlow003/aiig-suss.git
```
Change the current directory to `learning_materials/classfication`:
```bash
cd learning_materials/classfication
```
Create a virtual environment (`virtualenv`) and install required dependencies and packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
You can define the project's settings before running the appropriate scripts via `src/config.yml`:
```
etl:
  raw_data_file: "data/raw/Telco_customer_churn.xlsx"
  processed_path: "data/interim/"
  test_size: 0.3
  random_state: 42

# Training Decision Tree
train:
  processed_train: "data/interim/train.csv"
  model: "DecisionTreeClassifier"
  model_config: {random_state: 42}
  model_path: "models/dt_classifier.pkl"
  transformer_path: "models/transformer.pkl"
  sampler: "SMOTE"
  sampler_config: {random_state: 42}

# Tune Decision Tree
tune:
  param_grid: {max_depth: [3, 4, 5, 6],
               min_samples_leaf: [0.04, 0.06, 0.08],
               max_features: [0.2, 0.4, 0.6, 0.8]}
  scoring: "balanced_accuracy"
  tune_model: True

predict:
  model_path: "models/dt_classifier.pkl"
  transformer_path: "models/transformer.pkl"
  processed_test: "data/interim/test.csv"
  predicted_file: "data/processed/results.csv"
  export_result: True
```
Changes must be made to the following fields in order to apply a new classification algorithm for training:

1. `model` – Model to be used for training.
2. `model_path` – Path to save your model, usually under `models/{model}_classifier.pkl`

If hyperparameter tuning is required:

1. `param_grid` – Parameters settings to be evaluated in Grid Search.
2. `scoring` – Specify scoring strategy used in Grid Search evaluation.
3. `tune_model` – Specify as True, if tuning is required.

Usage
------------
To go through the entire project, you must run the project in three parts:

1. Extract, Transform, Load (ETL) of raw data
2. Training of model
3. Predicting results using trained model

```bash
python3 etl.py
python3 train.py
python3 predict.py
```

You may execute each of the individual scripts separately if, for example, you simply want to train the model and not predict anything. As a result, you can just execute the `train` script:
```bash
python3 train.py
```

In addition to the commands listed above, a shell script is provided that will execute the complete project for you:
```bash
./main.sh
```

Results
------------
All training and prediction results are created and logged, as defined in `log/`.

For instance, upon generating predictions, the results are logged in `log/predict.log`:
```
-------------------Predict and evaluate-------------------
2021-06-26 21:00:31,906 : INFO : src.utilities.utility : Classification report:
               precision    recall  f1-score   support

           0    0.88832   0.68988   0.77663      1522
           1    0.49138   0.77551   0.60158       588

    accuracy                        0.71374      2110
   macro avg    0.68985   0.73270   0.68911      2110
weighted avg    0.77771   0.71374   0.72785      2110

2021-06-26 21:00:31,910 : INFO : src.utilities.utility : Export prediction to : data/processed/results.csv
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

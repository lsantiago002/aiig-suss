Regression
==============================

Project Repo containing code for AI4I SUP-3: Regression module.

Setup
------------
Before you begin, ensure that all necessary dependencies are installed.

You can do so via running the following command on your command prompt:
```bash
bin/install_deps_locally.sh
```
A virtual environment will be created and required dependencies in `requirements.txt` will be installed.


Getting Started
------------
To train the models, you can run the following commands.

`Linear Regression`:
```python3
source .venv/bin/activate
python3 linear_regression.py
```

`K-Nearest Neighbors Regressor`:
```python3
source .venv/bin/activate
python3 knn_regressor.py
```

Results will be displayed as follows:
```
Linear Regression:

Training model on 10-Fold Cross Validation...
CV Score: 0.68

Results on Test Data
====================
RMSE: 44733.72
R2 Score: 0.73911

------------------------------

K-Nearest Neighbors Regressor:

Training model on 10-Fold Cross Validation...
CV Score: 0.70

Best parameters chosen: {'kneighborsregressor__n_neighbors': 4, 'kneighborsregressor__weights': 'distance'}
Results on Test Data
====================
RMSE: 35010.05
R2 Score: 0.84020
```

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
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-lwt-initial-exploratory-data-analysis`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
    │
    |
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── preprocessing.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

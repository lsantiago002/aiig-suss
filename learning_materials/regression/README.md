Regression
==============================

Project Repo containing code for AI4I [SUP-3: Regression module](https://learn.aisingapore.org/courses/ai-for-industry-part-2/lessons/sup-3-regression/).

In this lesson, you will continue building your knowledge about Regression by applying it to a new dataset, the Ames Housing dataset where you will attempt to predict housing prices.

At the end of this lesson, you should know not only have a better grasp of key concepts but also have created Python scripts that you can reuse in the future to quickly build more Machine Learning projects.

Overview
------------
During this course, you will exposed to the following:

1. Practise conducting an *Exploratory Data Analysis* on the given dataset based on the knowledge acquired through [EDA-1: Introduction to Data Visualisation in Python](https://learn.datacamp.com/courses/introduction-to-data-visualization-with-seaborn).
2. You will learn to develop a research methodology, which aims to help you answer some possible questions you have wrt.the dataset.
3. After understanding your dataset, you will be exposed to developing a simple `Linear Regression` model outside of the Jupyter Notebook interface. This aims to solidify your existing python programming knowledge as well as leveraging in some shell scripting that you picked up in  [LATO-3: Introduction to Shell](https://learn.datacamp.com/courses/introduction-to-shell).
4. In addition, you will learn the practise of usability in code by reusing code developed for the *Linear Regression* model, to modify and develop a `K-Nearest Neighbors Regressor` model.
5. Beside simply developing a model, you will also be exposed to simple *hyperparameter* tuning – which aims to maximise the performance of your model.

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

Setup
------------
Before you begin, ensure that all necessary dependencies are installed.

You can do so via running the following command on your command prompt:
```bash
bin/install_deps_locally.sh
```
A virtual environment will be created and required dependencies in `requirements.txt` will be installed.

<<<<<<< HEAD
<<<<<<< HEAD
You may also start by creating a `virtualenv`. Activate the virtualenv and `pip install` the packages and dependencies after that.

**Mac:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**Windows:**
```
pip install virtualenv
py -m virtualenv .env
source env/Scripts/activate
pip install -r requirements.txt
```
=======
=======
For Windows users:
------------
The 'Setup' above have to be done differently. 
1. Open Git Bash terminal. 
2. ```bash
    cd
    ``` 
    to your regression folder
3. Execute the following code to clone the repo:
    ```bash
    git clone https://github.com/wtlow003/aiig-suss
    ```
    Do this only if you haven't downloaded or cloned this repo.
    Else, follow the next step.

Caution:
If you try to run the code line:
```bash
./bin/install_deps_locally.sh
```
from the said Setup above, (it does not work at least for me-Loreine) you may get this error:
```bash
Python was not found; run without arguments to install from the Microsoft Store,
 or disable this shortcut from Settings > Manage App Execution Aliases.
```
You can refer to this: https://stackoverflow.com/a/48588878 for solution, though the error is very clear.

4. Before you can do "Getting Started" below, you need to install virtual env using the following command:
```bash
pip install virtualenv
```
5. Create your virtual env:
```bash 
py -m virtualenv .env
```
6. Activate the virtual env:
```bash 
source env/Scripts/activate
```
7. Install the requirements for this project.
```bash
pip install -r requirements.txt
```

Getting Started
------------
To train the models, you can run the following commands.

`Linear Regression`:
```bash
python3 linear_regression.py
```

`K-Nearest Neighbors Regressor`:
```bash
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

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

The Beer Type Prediction Project
==============================

The Beer Type Prediction project uses a trained Machine Learning Model to accurately predict a type of beer based on a finite set of review criteria.


Installation process
------------
The main steps required for installing and executing this project are as follows:
1) Clone the git repo
2) Download the training data set
3) Build and run the custom Docker image to train models and run predictions
4) Execute Jupyter notebook to train models and run predictions
5) Build and run the custom Docker image to build an API to interact with the model
6) Interact with API

Setup the local Git repository
------------

Create a new folder to store this repository, eg: ~/Projects/nba-career-predict:

<pre>
cd ~
mkdir Projects
cd Projects
</pre>

To download all the necessary files and folders (apart from the datasets) run command <code>git clone</code>.

<pre>git clone https://github.com/seanbwilliams/beer_type_prediction.git</pre>


Install Train Data
------------

Within your local repository main folder, create sub-folder 'data', and three other sub-folders within 'data':
<pre>
    |
    ├── data
    |   ├── external       <- The final prediction CSV output files for submission to Kaggle
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The raw Train & Test raw datasets.
</pre>

<pre>
cd beer_type_prediction
mkdir data
cd data
mkdir external
mkdir processed
mkdir raw
cd ..
</pre>

Download dataset from <a href="https://drive.google.com/file/d/1vYyJL_IB6KjKCxuk9kg4vIMPGTtoX8Ek/view?usp=sharing">Google Drive</a> and store in the local repository 'data\raw' folder.


Build custom pytorch-notebook Docker image and run container
------------

This solution is using a customer docker image that ensures the required libraries and their versions are ready to go.

Build the Docker image

<pre>
docker build -t pytorch-notebook:latest .
</pre>


Run the Docker image

<pre>
docker run  -dit --rm --name beer_type_prediction -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v {PWD}:/home/jovyan/work pytorch-notebook:latest 
</pre>


Execute Notebooks
---------------

Locate the URL in the Docker log, and paste it into a browser to launch Jupyter Lab

<pre>docker logs --tail 50 beer_type_prediction</pre>

Execute the notebooks in the following order:

1. dataprep
2. modelling
3. pipeline


Build custom beer-type-pred-fastapi Docker image and run container
------------

This solution is using a customer docker image to serve an API that interacts with the ML Model.

Build the Docker image

<pre>
cd app
docker build -t beer-type-pred-fastapi:latest .
</pre>


Run the Docker image

<pre>
docker run  -dit --rm --name beer_type_pred_fastapi -p 8080:80 beer-type-pred-fastapi:latest
cd ..
</pre>


Interact with API
---------------

Click on the link below to interact with the ML model:

<a href="http://127.0.0.1:8080/">Beer Type Prediction Project</a>


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
    │                         `1.0-jqp-initial-data-exploration`.
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
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

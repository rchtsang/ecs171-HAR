# ECS171 Group 19 Final Project Repository

Welcome to our final project repository!

Please note that there are both `.ipynb` and `.py` files present in this repository. In general, notebooks will contain our model training and testing code for development and should all work once the repo environment is properly set up. `.py` files will generally contain support scripts. See below for the repo structure and file descriptions.

## Repo Details

### Models

All models can be found in the `models` directory. They are subdivided into the different models that we implemented, as well as a folder `phone_accel`, which contains the subset of data from the WISDM dataset that we used for model testing and training. More information about each model as well as their results can be found in the corrosponding directories and/or notebooks. 

- Artificial Neural Network
- k-Nearest Neighbors
- Logistic Regression
- Random Forest

Note: spectral-nn was an attempt to preprocess the raw data into spectral and cepstral features. The feature extraction worked, but unfortunately did not perform well under simple sklearn MLPClassifier, likely due to problems with the dataset sampline rate. This has been noted in the report as well.

### Web App

The `webapp` folder contains the relevent code to run our web app demo. The `README.md` inside that folder contains instructions for running it. It does not implement all of the models and has a separate, modified copy of the ANN implementation for integration with a javascript backend. 

### File Tree
```
.
|-- README.md
|-- environment.yml
|-- models/
|   |-- ann/
|   |   |-- ann.ipynb
|   |   `-- wisdm_preprocess.py
|   |-- knn/
|   |   `-- knn.ipynb
|   |-- logistic/
|   |   `-- logistic.ipynb
|   |-- phone_accel/
|   |-- random-forest/
|   |   `-- random-forest.ipynb
|   |-- spectral-nn/
|   |   |-- README_freq.md
|   |   |-- bins.py
|   |   |-- freq-model.ipynb
|   |   |-- freq-model.py
|   |   |-- freq-preprocessing.ipynb
|   |   `-- freq-preprocessing.py
|   `-- wisdm-dataset/
|-- setup-wisdm.py
|-- setup.sh*
`-- webapp/
```

See `webapp` folder for its file tree.

## Setup

Some setup is necessary to run code in this repository. If you are running a unix-based system (Darwin or Linux), there is a bash script `setup.sh` that can be run once that will create a conda environment and install any dependencies.
Otherwise, things will need to be done manually as follows.

PLEASE NOTE: anaconda is notoriously difficult to install, so **we assume that you have a working distribution of `conda` installed and added to PATH variable**. If you don't have that, we recommend installing Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html) for a quick start up. 
**we also assume you have a working distribution of `node` added to PATH**. This can be installed directly as outlined on the [official node install guide](https://nodejs.org/en/download/) or with a [package manager](https://nodejs.org/en/download/package-manager/).

1. Create a new conda environment from `environment.yml`. This should set you up immediately with all python dependencies. Don't forget to activate it before running any scripts!

2. Download the WISDM Dataset from the UCI ML Repo. (For this we have provided a poratble python script `setup-wisdm.py` that will setup the dataset folders as our scripts expect them. Simply run `python3 setup-wisdm.py`)

3. To run the webapp, `node` must be installed.

## Repo Rules (For Contributors)

1. Unless we've all agreed, **never edit the master or dev branch**. Make your own branch, push it upstream, and push to that. I recommend setting up your own branch as soon as you clone the repo so you don't forget and need to fix things later.
1. Have a .gitignore file! You don't want to clutter the repo even on your own branch in case merges need to happen. 

## Merge Notes

6/2: During merge, we may have missed some filepath corrections, so if there are any `file/dir does not exist` errors, it is likely because the path name is wrong and needs to be corrected.

## Helpful Links:

- [WISDM Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)

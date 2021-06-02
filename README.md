# ECS171 Group 19 Final Project Repository

Welcome to our final project repository!

Please note that there are both `.ipynb` and `.py` files present in this repository. In general, notebooks will contain our model training and testing code for development and should all work once the repo environment is properly set up. `.py` files will generally contain support scripts. See below for the repo structure and file descriptions.


## Setup

Some setup is necessary to run code in this repository. If you are running a unix-based system (Darwin or Linux), there is a bash script `setup.sh` that can be run once that will create a conda environment and install any dependencies.
Otherwise, things will need to be done manually as follows.

PLEASE NOTE: anaconda is notoriously difficult to install, so **we assume that you have a working distribution of `conda` installed already**. If you don't have that, we recommend installing Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html) for a quick start up. 

1. Create a new conda environment from `environment.yml`. This should set you up immediately with all python dependencies.

2. Download the WISDM Dataset from the UCI ML Repo. (For this we have provided a python script `setup-wisdm.py` that will setup the dataset folders as our scripts expect them. Simply run `python3 setup-wisdm.py`)

## Repo Rules (For Contributors)

1. Unless we've all agreed, **never edit the master branch**. Make your own branch, push it upstream, and push to that. I recommend setting up your own branch as soon as you clone the repo so you don't forget and need to fix things later.
1. Have a .gitignore file! You don't want to clutter the repo even on your own branch in case merges need to happen. 

## Merge Notes

6/2: During merge, we may have missed some filepath corrections, so if there are any `file/dir does not exist` errors, it is likely because the path name is wrong and needs to be corrected.

## Helpful Links:

- [WISDM Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)

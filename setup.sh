#!/bin/bash

# this script should only be run once
# to keep track of this, it creates a hidden empty
# file as a nonce. if you really need to rerun this, 
# delete the nonce and try again.
NONCE=.nonce
if [[ -f "$NONCE" ]]; then
	echo "$NONCE exists! This script has already been run!"
else
	ENV_NAME=ecs171-proj

	# check if conda is installed
	if ! [[ -x "$(command -v conda)" ]]; then
		echo "Error: Couldn't find conda."
		exit 1
	fi

	# check for conda environment and python setup files
	if [[ ! -f "environment.yml" || ! -f "setup-wisdm.py" ]]; then
		echo "Error: Couldn't find environment.yml and/or setup-wisdm.py."
		exit 1
	fi 

	# init conda for bash
	conda init bash

	# create and activate conda environment
	echo "Creating conda environment $ENV_NAME"
	conda env create -f environment.yml --name $ENV_NAME -y
	echo "Activating environment"
	source $(conda info --base)/etc/profile.d/conda.sh
	conda activate $ENV_NAME

	# download and setup WISDM Dataset
	echo "Running setup-wisdm.py"
	python3 setup-wisdm.py

	# create the nonce
	touch $NONCE
	FINISHED="Done! Note: Before you run any scripts,"
	FINISHED="$FINISHED you must activate the new conda environment"
	FINISHED="$FINISHED by running 'conda activate $ENV_NAME'"
	echo $FINISHED
fi
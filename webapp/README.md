# Webapp

## Prerequisites

We use a node.js server, so node must be installed. You should be able to navigate to the 'webapp' directory and run 'npm install' to install all dependencies (I believe we actually just use express).

## Usage

Navigate to 'webapp' and run 'node server.js'. Then open your web browser to 'localhost:3000' and browse for a datafile (examples in models/activities).

WARNING: The code is not robust. If you submit something other than a datafile or a datafile that is wonky just restart the server or refresh the page or both.

## Summary

This webapp is designed to ultimately allow users to input a data file, which is then analyzed by our machine learning model to predict what activity the data file is associated with.

Sample datafiles (data corresponding to 3 minutes of a single activity) are provided in models/activities. Use these to test the website.

## Features

Currently the app is a single page, prompting users to upload a file. Once a file is uploaded, a POST request sends the file to the node.js backend, which then uses the NN model to make a prediction on what activity the datafile represents. That prediction is sent in the response of the POST request and the Output is updated. The whole process takes about 10 seconds, which may be a bit long, but is reasonable.

This page is powered by a node.js backend, which serves all files in the 'public' folder. We use the built-in 'child-process' package to run python files (translated from the jupyter notebooks using nbconvert), whose stdout (print statements) is then captured by the node server. 

Right now the only model in use is Jack's NN. It was the highest priority as it is our most accurate model. To implement more models, we may need to reformat them as a class the way Jack did in 'nnmodel.ipynb'.

## To do

The most obvious improvement would be to include the other models (NN would be default, but user could choose to try a different model). This is currently planned.

The design could be improved, if we figure out more content to include or a better background or whatever. This is lower priority.

Right now the only files we can use on the website come from the data the website was trained on. An unnecessary but extremely cool addition would be to use a gyroscope/accelerometer app to record our own datafiles and see if our models can handle original data as well.
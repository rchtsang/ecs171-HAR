# Webapp

## Intro

This webapp is designed to ultimately allow users to input a data file, which is then analyzed by our machine learning model to predict what activity the data file is associated with.

## Features

Currently the app is a single page, prompting users to upload a file. The contents of the file will then be printed to the console. This is temporary - eventually the file will be sent to the server and processed using our models.

This page is powered by a node.js backend, which serves all files in the 'public' folder. We use the built-in 'child-process' package to run python files (translated from the jupyter notebooks using nbconvert), whose stdout (print statements) is then captured by the node server. 

This is a bit of a problem since print statements are commonly used in the jupyter notebooks to visualize the modeling process. The current goal is to save the result of the models (using something like pickle) so that they can be used by a separate function that takes as input two arguments: the type of data (phone/watch gyro/accel), and the input data from which to make a prediction.

This is all subject to change as we may run into roadblocks, but this is designed to be a simple rundown of what we have implemented and what we are working on.
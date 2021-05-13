from nnmodel import Reader, Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import pandas as pd
import pickle
import sys

# Apply the Reader class:
df = Reader("models/data.arff",mode='f').df # read datafile into dataframe
df[0].columns = df[1] #assign column names
df = df[0]

# Apply the Model class:
# Evaluate model with 70/30 train/test split

# list attributes and label of choice
attributes = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
       'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Z0', 'Z1',
       'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'XAVG', 'YAVG', 'ZAVG',
       'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV',
       'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'XVAR', 'YVAR', 'ZVAR', 'RESULTANT']
label = 'ACTIVITY'
testsize = 0.3 # use a 70/30 train/test split

X = np.array(df[attributes])

# Instantiate the model as you please (we are not going to use this)
estimator2 = KerasClassifier(build_fn=Sequential(), epochs=10, batch_size=10, verbose=1)

# This is where you load the actual saved model into new variable.
estimator2.model = load_model('models/nn.h5')
estimator2.classes_ = pickle.load(open( "models/nn_classes.p", "rb" ))
encoder = pickle.load(open( 'models/nn_encoder.p', 'rb' ))

# Making predictions?
activity_num = 0
left = activity_num * 18
right = activity_num * 18 + 18
arr = estimator2.predict(X[left:right])
letter = encoder.inverse_transform([np.argmax(np.bincount(arr))])[0]

classes =  {"A":"Walking","B":"Jogging","C":"Stairs","D":"Sitting",
            "E":"Standing","F":"Typing","G":"Brushing teeth","H":"Eating soup",
            "I":"Eating chips","J":"Eating pasta","K":"Drinking from a cup",
            "L":"Eating sandwich","M":"Kicking (soccer ball)","O":"Playing catch (with tennis ball)",
            "P":"Dribbling (basketball)", "Q":"Writing","R":"Clapping","S":"Folding clothes"}

print(classes[letter])

sys.stdout.flush()
sys.stderr.flush()
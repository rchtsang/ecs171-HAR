#!/usr/bin/env python
# coding: utf-8

# # Neural Network Model for Human Activity Recognition of WISDM Dataset
# ## by Jack Goon
# 
# ## A. sklearn's MLPClassifier
# ### 1. Read data

# In[1]:

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class Reader:
    def __init__(self, path, mode='f'):
        
        self.df = self.wrapper(path,mode)
    
    # read arff file
    def readarff(self, filename,collect=True): #collect if you need to collect attribute names
        with open(filename) as f:
            content = f.read().splitlines()
        data = False
        metalist = [] # storets metadata as list of rows
        datalist = [] # store data as list of rows
        
        # read data line-by-line
        for line in content:
            if data == True:
                line = line.split(",")
                datalist.append(line)
            elif line == "@data":
                data = True # read lines before "@data" as metadata and after as data
            else:
              # clean up metadata header
              if collect:
                line = line.replace(' "', ".")
                line = line.replace('" ', ".")
                line = line.replace(" ","")
                line = line.split(".")
                if len(line)==3: #ignore first two lines of file
                    line = line[1:3] #remove repetitive "@attribute"
                    metalist.append(line)
        
        # create dataframes from lists of rows
        if not collect:
            dataframe = pd.DataFrame(datalist,dtype=float)
            return dataframe
        else:
            dataframe = pd.DataFrame(datalist,dtype=float)
            metaframe = pd.DataFrame(metalist,columns=["attribute","description"])
            attributes = metaframe["attribute"].rename("SAMPLE")
            return dataframe, attributes
    
    def readdirectory(self, path,quiet=False): # make sure path ends in a slash
        alldata = []
        count = 0
        for filename in os.listdir(path):
            if filename.endswith(".arff"):
                if count == 0: #only collect attributes once
                    if not quiet:
                          print("processing "+filename+"; collecting attribute names")
                    dataframe, attributes = self.readarff(path+filename)
                    alldata.append(dataframe)
                else:
                    if not quiet:
                        print("processing "+filename)
                    dataframe = self.readarff(path+filename,collect=False)
                    alldata.append(dataframe)
                count += 1
                continue
            else:
                continue
        if not quiet:
            print("Concatenating data")
        alldata = pd.concat(alldata).reset_index(drop=True) #reset indices so it is continuous
        alldata.columns = attributes #assign column names
        return alldata
    
    def wrapper(self, path, mode='f'):
        if mode == 'f':
            try:
                df =  self.readarff(path, collect = True)
                return df
            except:
                print("make sure you inputted the correct arff FILE path")
        elif mode == 'd':
            try:
                df =  self.readdirectory(path, quiet=True)
                return df
            except:
                print("make sure you inputted the correct arff DIRECTORY path, ending with a slash")
        else:
            print("mode must either be 'f' or 'd'")
            return 0


# In[2]:


phone_accel = Reader("models/data/phone_accel/",mode='d').df
phone_accel


# ### 2. Preprocess data

# In[3]:


# FORMAT DATA
# remove ACTIVITY, RESULTANT, MFCC values, COS values, and CORRELATION values (same as publication)
X = phone_accel[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
       'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Z0', 'Z1',
       'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'XAVG', 'YAVG', 'ZAVG',
       'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV',
       'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'XVAR', 'YVAR', 'ZVAR', 'RESULTANT']]
columns = X.columns
# FORMAT LABELS
# TODO, group similar activities
conversions = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 
               'J':9, 'K':10, 'L':11, 'M':12, 'O':13, 'P':14, 'Q':15, 'R':16, 'S':17}
y_raw = phone_accel['ACTIVITY']
y = []
# convert letters to 1's and 0's in a matrix
for i in y_raw:
    j = conversions[i] # convert activity letter to a number, j
    row = [0]*18 # initialize an array of zeros
    row[j]=1 # set the j'th item in the row to 1, indicating that the sample belongs to the j'th category
    y.append(row)
y = pd.DataFrame(y)
y.columns = conversions.keys()
print("Class labels:")
y.head()


# In[4]:


from sklearn.preprocessing import MinMaxScaler

# Scale data
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
X.columns = columns
print("Attribute values:")
X.head()


# ### 3. Train model with neural network
# Using [Scikit-learn's MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)

# In[5]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# shuffle and split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, random_state=1)

# define classifier to use stochastic gradient descent, sigmod activation function, regularization constant 1e-5, etc
clf = MLPClassifier(solver='sgd', activation='relu', alpha=1e-3, learning_rate_init = 0.01, 
                    hidden_layer_sizes=(100,100,100,25), random_state=1, max_iter = 1000) # 4000 for best performance
clf.fit(X_train, y_train)


# ### 4. Evaluate model performance

# In[6]:


print("Training score:",clf.score(X_train, y_train))
print("Testing score:",clf.score(X_test, y_test))


# In[7]:


sample = X.iloc[30]
sample = np.array(sample).reshape(1, -1) #reshape because single sample
print("sample probability vector: ",clf.predict_proba(sample))
print("sample prediction vector: ",clf.predict(sample))
print("actual sample class vector: ",np.array(y.iloc[30]).reshape(1,-1))


# In[8]:


import matplotlib.pyplot as plt
loss_values = clf.loss_curve_
plt.plot(loss_values)
plt.show()


# ## B. Keras

# ### 1. Read data

# In[9]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

phone_accel = Reader("phone_accel/",mode='d').df

# read attributes
X = phone_accel[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
       'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Z0', 'Z1',
       'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'XAVG', 'YAVG', 'ZAVG',
       'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV',
       'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'XVAR', 'YVAR', 'ZVAR', 'RESULTANT']]
X = np.array(X)

# read and convert class labels to binary form
Y = phone_accel['ACTIVITY']
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = pd.get_dummies(Y) # convert to one hot encoded Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify=Y, random_state=1)


# ### 2. Create Keras model

# In[10]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.optimizers import SGD
from keras.optimizers import Adam


# Below, the "layer call" action `x = dense(inputs)` is like drawing an arrow from "inputs" to this dense layer you created. You're "passing" the inputs to the dense layer, and you get x as the output.

# In[11]:


p_input = 0
p_hidden = 0.1 # fraction of the inputs to drop
eta = 0.00001 # learning rate, not in use
m = 0.95 # momentum constant, not in use

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='relu')) # input layer
    model.add(Dropout(p_input)) # dropout applied to input layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(p_hidden)) # dropout applied to first hidden layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(p_hidden)) # dropout applied to second hidden layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p_hidden)) # dropout regularization layer
    #model.add(Dense(32, activation='relu', kernel_regularizer='l2')) # default regularization constant 0.01
    model.add(Dense(Y.shape[1], activation = 'softmax')) # output layer = softmax for probabilities
    # Compile model
    
    sgd = SGD(lr=eta, momentum=m)
    adam = Adam()
    
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

# from https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    epochs = range(1,len(history.history[loss_list[0]]) + 1) # x values on graph
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# ### 3. Train and evaluate model

# #### First, test model on entire test set and try different batch sizes

# In[12]:


import time
start_time = time.time()
estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=200, verbose=0)
kfold = KFold(n_splits=4, shuffle=True)

# uncomment this for cross validation
# results = cross_val_score(estimator, X, y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# print("%s seconds" % (time.time() - start_time))

history = estimator.fit(X_train, Y_train, verbose = 0)
plot_history(history)
print("Test score: "+str(estimator.score(X_test, Y_test)))


# In[13]:


import time
start_time = time.time()
estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=400, verbose=0)

# uncomment this for cross validation
# kfold = KFold(n_splits=4, shuffle=True)
# results = cross_val_score(estimator, X, y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# print("%s seconds" % (time.time() - start_time))

history = estimator.fit(X_train, Y_train, verbose = 0)
plot_history(history)
print("Test score: "+str(estimator.score(X_test, Y_test)))


# #### Now, let's use the model to evaluate accuracies for each individual class

# In[14]:


classes = {"A":"Walking","B":"Jogging","C":"Stairs","D":"Sitting",
           "E":"Standing","F":"Typing","G":"Brushing teeth","H":"Eating soup",
           "I":"Eating chips","J":"Eating pasta","K":"Drinking from a cup",
           "L":"Eating sandwich","M":"Kicking (soccer ball)","O":"Playing catch (with tennis ball)",
           "P":"Dribbling (basketball)", "Q":"Writing","R":"Clapping","S":"Folding clothes"}

print("Model prediction accuracies based on phone acceleration data:")
print("="*61)
for i in encoder.classes_: # iterate through classes
    j = encoder.transform([i])
    idx = Y_test[j[0]] # create reference array to only choose samples from the i'th (aka j'th) class
    Y1 = Y_test[idx==1]
    X1 = X_test[idx==1]
    print(classes[i], "test score:","-"*(40-len(classes[i])),str(estimator.score(X1, Y1)))


# ## C. Repeat Keras with Watch Accel

# In[15]:


# repetitive imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

watch_accel = Reader("watch_accel/",mode='d').df

# read attributes
Xw = watch_accel[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
       'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Z0', 'Z1',
       'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'XAVG', 'YAVG', 'ZAVG',
       'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV',
       'XSTANDDEV', 'YSTANDDEV', 'ZSTANDDEV', 'XVAR', 'YVAR', 'ZVAR', 'RESULTANT']]
Xw = np.array(Xw)

# read and convert class labels to binary form
Yw = watch_accel['ACTIVITY']
encoder = LabelEncoder()
encoder.fit(Yw)
Yw = encoder.transform(Yw)
Yw = pd.get_dummies(Yw) # convert to one hot encoded Y

Xw_train, Xw_test, Yw_train, Yw_test = train_test_split(Xw, Yw, test_size = 0.3, stratify=Yw, random_state=1)


# In[16]:


import time

def baseline_modelw():
    # create model
    model = Sequential()
    model.add(Dense(Xw.shape[1], input_dim=Xw.shape[1], activation='relu')) # input layer
    model.add(Dropout(p_input)) # dropout applied to input layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(p_hidden)) # dropout applied to first hidden layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(p_hidden)) # dropout applied to second hidden layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p_hidden)) # dropout regularization layer
    #model.add(Dense(32, activation='relu', kernel_regularizer='l2')) # default regularization constant 0.01
    model.add(Dense(Yw.shape[1], activation = 'softmax')) # output layer = softmax for probabilities
    # Compile model
    
    sgd = SGD(lr=eta, momentum=m)
    adam = Adam()
    
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

start_time = time.time()
estimator = KerasClassifier(build_fn=baseline_modelw, epochs=500, batch_size=400, verbose=0)

# uncomment this for cross validation
# kfold = KFold(n_splits=4, shuffle=True)
# results = cross_val_score(estimator, X, y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# print("%s seconds" % (time.time() - start_time))

history = estimator.fit(Xw_train, Yw_train, verbose = 0)
plot_history(history)
print("Test score: "+str(estimator.score(Xw_test, Yw_test)))


# In[17]:


classes = {"A":"Walking","B":"Jogging","C":"Stairs","D":"Sitting",
           "E":"Standing","F":"Typing","G":"Brushing teeth","H":"Eating soup",
           "I":"Eating chips","J":"Eating pasta","K":"Drinking from a cup",
           "L":"Eating sandwich","M":"Kicking (soccer ball)","O":"Playing catch (with tennis ball)",
           "P":"Dribbling (basketball)", "Q":"Writing","R":"Clapping","S":"Folding clothes"}

print("Model prediction accuracies based on watch acceleration data:")
print("="*61)
for i in encoder.classes_: # iterate through classes
    j = encoder.transform([i])
    idx = Yw_test[j[0]]
    Y1w = Yw_test[idx==1]
    X1w = Xw_test[idx==1]
    print(classes[i], "test score:","-"*(40-len(classes[i])),str(estimator.score(X1w, Y1w)))

# Provide prints to node js
sys.stdout.flush()
# In[ ]:





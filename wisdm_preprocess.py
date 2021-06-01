import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# class to read arff files from WISDM dataset
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

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# function to preprocess data by normalizing numerical attributes and one-hot encoding categorical labels
def preprocess(df, attributes, class_label, normalize=True):
        """
        attributes = list of desired attribute names
        class_label = string of the desired class label
        """
        # define X
        X = df[attributes]
        
        # Normalize data
        if normalize:
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X))
            
        # define Y
        Y = df[class_label]
        encoder = LabelEncoder() # encoder stores conversion between class values (str) and identifiers (int)
        encoder.fit(Y)
        Y = encoder.transform(Y)
        Y = pd.get_dummies(Y) # convert to one hot encoded form
        
        # output combined df
        df = pd.concat([X, Y], axis=1)
        return df, X, Y, encoder


from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

# function to find outliers using LocalOutlierFactor and IsolationForest
def findOutliers(df):
    """
    Function that takes in a dataframe and returns outliers that are identified by both 
    the LocalOutlierFactor and Isolation Forest methods.
    """
    # identify outliers with LocalOutlierFactor
    lof = LocalOutlierFactor()
    pred_lof = lof.fit_predict(df)

    # identify outliers with IsolationForest
    isoforest = IsolationForest()
    pred_isoforest = isoforest.fit_predict(df)
    
    # identify outliers at the intersection of both methods
    outliers = np.intersect1d(np.where(pred_lof == -1), np.where(pred_isoforest == -1))
    
    return outliers

import seaborn as sns
import matplotlib.pyplot as plt
import math

# function to plot feature correlation
def heatmap(df):
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(df.corr(),annot=False, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm',square=True,ax=ax)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import os
import sys
import multiprocessing as mp
from multiprocessing import Manager, Pool, Lock
from glob import glob
from time import time
# set multiprocessing start method to 'fork' for macOS
# if running on jupyter notebook
# you can only run this once
# mp.set_start_method('fork')

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

# GLOBALS

DATA_DIR_PATH = "../data/"


def main():
    # read data from files for phone accel
    print("Reading data from files...")

    filepaths = glob("_spectral/*_phone_accel.txt")

    def read_file_csv(filepath):
        # read data from csv files
        df = pd.read_csv(filepath, engine='c')

        return df

    # using multiprocessing again to speed up reads
    with Pool(processes=2) as pool:
        dataframes = pool.map(read_file_csv, filepaths)
        
    pa_df = pd.concat(dataframes,
                      # columns=dataframes[0].columns,
                      ignore_index=True,
                      copy=False)

    # treat sid as object
    pa_df['sid'] = pa_df['sid'].astype('object')

    # Label Encoding
    # encoder = LabelEncoder()
    # encoder.fit(pa_df['label'])

    # feat_names=list(encoder.classes_)
    # label_df = pd.DataFrame(encoder.transform(pa_df['label'].to_numpy().reshape(-1, 1)),
    #                         columns=['label']).astype('int')

    # pa_df_enc = pd.concat([pa_df.drop(['label'], axis=1), label_df], axis=1)

    # OneHotEncoding for the label (for learning, keep data separate)
    pa_df_enc = pd.get_dummies(pa_df, columns=['label'])
    feat_names = [f'label_{l}' for l in pa_df['label'].unique()]

    # failed attempt to use sklearn version
    # encoder = OneHotEncoder(handle_unknown='ignore')
    # encoder.fit(pa_df['label'].unique().reshape(-1, 1))

    # feat_names = encoder.get_feature_names()
    # ohe = encoder.transform(pa_df['label'].to_numpy().reshape(-1, 1)).toarray()
    # label_df = pd.DataFrame(ohe, columns=feat_names).astype('int')

    # pa_df_enc = pd.concat([pa_df.drop(['label'], axis=1), label_df], axis=1)

    print("Original Dataframe:")

    print(pa_df.info())
    print(pa_df.head())

    print("Encoded Dataframe:")

    print(pa_df_enc.info())
    print(pa_df_enc.head())

    # outliers analysis
    print("\n\nOutliers Analysis")

    ocsvm = OneClassSVM(nu=0.01)
    pred_ocsvm = ocsvm.fit_predict(pa_df.select_dtypes(np.number))
    ocsvm_ct = pd.crosstab(pred_ocsvm, columns=['count'])
    print("OCSVM Results:", ocsvm_ct)

    isof = IsolationForest(n_jobs=2)
    pred_isof = isof.fit_predict(pa_df.select_dtypes(np.number))
    isof_ct = pd.crosstab(pred_isof, columns=['count'])
    print("ISOF Results:", isof_ct)

    outliers = np.intersect1d(np.where(pred_isof == -1), np.where(pred_ocsvm == -1))

    print("Num Outliers:", len(outliers))
    print("OCSVM && ISOF:", list(outliers))

    # we will not be dropping the outliers for now

    # skipping visualization of correlation matrix

    # scaling
    to_scale = list(pa_df_enc.select_dtypes(np.number).columns)
    to_scale = list(set(to_scale).difference(set(feat_names)))

    # print(to_scale)
    scalers = {k:StandardScaler() for k in to_scale}

    for col in to_scale:
        pa_df_enc.loc[:,col] = scalers[col].fit_transform(np.array(pa_df_enc.loc[:,col]).reshape(-1, 1))
    # print(pa_df_enc.head())

    # try learning with just the cepstral coefficients first
    drop = ['xmag0.5', 'xmag1.0', 'xmag1.5', 'xmag2.0', 'xmag2.5', 'xmag3.0', 'xmag3.5', 'xmag4.0', 
            'xmag4.5', 'xmag5.0', 'xmag5.5', 'xmag6.0', 'xmag6.5', 'xmag7.0', 'xmag7.5', 'xmag8.0', 
            'ymag0.5', 'ymag1.0', 'ymag1.5', 'ymag2.0', 'ymag2.5', 'ymag3.0', 'ymag3.5', 'ymag4.0', 
            'ymag4.5', 'ymag5.0', 'ymag5.5', 'ymag6.0', 'ymag6.5', 'ymag7.0', 'ymag7.5', 'ymag8.0', 
            'zmag0.5', 'zmag1.0', 'zmag1.5', 'zmag2.0', 'zmag2.5', 'zmag3.0', 'zmag3.5', 'zmag4.0', 
            'zmag4.5', 'zmag5.0', 'zmag5.5', 'zmag6.0', 'zmag6.5', 'zmag7.0', 'zmag7.5', 'zmag8.0',
            'xpow0.5', 'xpow1.0', 'xpow1.5', 'xpow2.0', 'xpow2.5', 'xpow3.0', 'xpow3.5', 'xpow4.0', 
            'xpow4.5', 'xpow5.0', 'xpow5.5', 'xpow6.0', 'xpow6.5', 'xpow7.0', 'xpow7.5', 'xpow8.0', 
            'ypow0.5', 'ypow1.0', 'ypow1.5', 'ypow2.0', 'ypow2.5', 'ypow3.0', 'ypow3.5', 'ypow4.0', 
            'ypow4.5', 'ypow5.0', 'ypow5.5', 'ypow6.0', 'ypow6.5', 'ypow7.0', 'ypow7.5', 'ypow8.0', 
            'zpow0.5', 'zpow1.0', 'zpow1.5', 'zpow2.0', 'zpow2.5', 'zpow3.0', 'zpow3.5', 'zpow4.0', 
            'zpow4.5', 'zpow5.0', 'zpow5.5', 'zpow6.0', 'zpow6.5', 'zpow7.0', 'zpow7.5', 'zpow8.0',
            'xcc10', 'xcc11', 'xcc12', 'xcc13', 'xcc14', 'xcc15', 
            'ycc10', 'ycc11', 'ycc12', 'ycc13', 'ycc14', 'ycc15', 
            'zcc10', 'zcc11', 'zcc12', 'zcc13', 'zcc14', 'zcc15']

    print("\n70:30 Train Test Split")

    # 70:30 split
    train, test = train_test_split(pa_df_enc.drop(['sid'] + drop, axis=1), test_size=0.3)
    train = train.copy()
    test = test.copy()

    # begin learning

    # shuffle the training set
    train = train.sample(frac=1)

    # split into independent and dependent variables
    y_train, x_train = train[feat_names].copy(), train.drop(feat_names, axis=1)
    y_test, x_test = test[feat_names].copy(), test.drop(feat_names, axis=1)

    # construct neural network
    nn = MLPClassifier(hidden_layer_sizes=(40, 40),
                       activation='logistic',
                       solver='adam',
                       max_iter=1000000,
    #                    early_stopping=True,
                       random_state=1
                    )

    # measuring training time
    t = time()
    print("Begin Training...")

    # train model
    nn.fit(x_train, y_train)

    # final accuracy scores
    train_score = nn.score(x_train, y_train)
    test_score = nn.score(x_test, y_test)

    # final recall scores
    train_recall = recall_score(y_train, nn.predict(x_train),
                                zero_division=0,
                                average='macro')
    test_recall = recall_score(y_test, nn.predict(x_test),
                                zero_division=0,
                                average='macro')

    # final precision score
    train_precision = precision_score(y_train, nn.predict(x_train),
                                zero_division=0,
                                average='macro')
    test_precision = precision_score(y_test, nn.predict(x_test),
                                zero_division=0,
                                average='macro')


    print(f"Elapsed Time: {time()-t} s")
    print(f"Final Weights: {[w[0] for w in nn.coefs_[-1]]}\n")
    print(f"Final Training Accuracy {round(train_score, 5)}")
    print(f"Final Testing Accuracy {round(test_score, 5)}")
    print(f"Final Training Recall {round(train_recall, 5)}")
    print(f"Final Testing Recall {round(test_recall, 5)}")
    print(f"Final Training Precision {round(train_precision, 5)}")
    print(f"Final Testing Precision {round(test_precision, 5)}")

if __name__ == "__main__":
    main()
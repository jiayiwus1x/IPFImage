import os, random, warnings
import plotly.express as px
import warnings
import numpy as np
import pandas as pd
import autosklearn.classification

from glob import glob


import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest

def truncate_SVD(Xi, nor=True, n_components=5, n_iter=50, random_state=42):
    if nor:
        X = np.array(Xi)/np.linalg.norm(np.array(Xi), axis = 0)
    else:
        X = Xi
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state)
    re =svd.fit(X)
    print(sum(svd.explained_variance_ratio_))
    n5=svd.fit_transform(X)
    return n5

def SVC_GS(X, y, parameters = {'kernel':('linear', 'rbf', 'sigmoid', 'poly'), \
                         'C': list(range(1, 10)), 'gamma':np.arange(0.1, 1.5, 0.1)},\
           class_weight='balanced', cv = 5, refit='Accuracy'):
    
    
    svc = SVC(class_weight=class_weight, probability=True)
    clf = GridSearchCV(svc, parameters, cv = cv,scoring = {'AUC': 'roc_auc', 'Accuracy': 
                                                            make_scorer(sklearn.metrics.accuracy_score)}, 
                       refit= refit,return_train_score=True)
#     clf =  GridSearchCV(svc, parameters, cv = cv, n_jobs = 1)
    results = clf.fit(X,y)
#     print('Best Mean Accuracy: %.3f' % results.best_score_)
#     print('Best Config: %s' % results.best_params_)
    # summarize all
    means_accuracy = results.cv_results_['mean_test_Accuracy']
    means_AUC = results.cv_results_['mean_test_AUC']
    params = results.cv_results_['params']
#     for i,j, param in zip(means_accuracy, means_AUC, params):
#         print("accuracy >%.3f, AUC >%.3f with: %r" % (i, j, param))
    return results


def binary(ys):
    
    color = np.zeros(len(ys))
    color[np.where(ys==list(set(ys))[0])] = 1
    return color

def str2int(x):
    labels = set(x)
#     print(labels)
    xint = np.zeros(len(x))
    for i, label in enumerate(labels):
        xint[np.where(x==label)] = i
    return xint
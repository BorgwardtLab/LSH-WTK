'''
Contains various utility functions for data handling, cross-validation,
and so on.
'''
import os
from os.path import basename, join
import numpy as np
from numpy.linalg import eigh
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import SVC

import sys
sys.path.append('../src')

from kernels import pairwise_subsequence_kernel
from kernels import wasserstein_kernel


def strip_suffix(s, suffix):
    '''
    Removes a suffix from a string if the string contains it. Else, the
    string will not be modified and no error will be raised.
    '''

    if not s.endswith(suffix):
        return s
    return s[:len(s)-len(suffix)]


def read_ucr_data_txt(filename):
    '''
    Loads an UCR data set from a txt file, returns the samples and the
    respective labels. Also extracts the data set name such that one
    may easily display results.
    '''

    data = np.loadtxt(filename)
    Y = data[:, 0]
    X = data[:, 1:]

    # Remove all potential suffixes to obtain the data set name. This is
    # somewhat inefficient, but we only have to do it once.
    name = os.path.basename(filename)
    name = name[:len(name)-4]
    name = strip_suffix(name, '_TRAIN')
    name = strip_suffix(name, '_TEST')

    return X, Y, name


def get_ucr_dataset(data_dir: str, dataset_name: str):
    '''
    Loads train and test data from a folder in which
    the UCR data sets are stored.
    '''

    X_train, y_train, _ = read_ucr_data_txt(os.path.join(data_dir, dataset_name, f'{dataset_name}_TRAIN.txt'))
    X_test, y_test, _ = read_ucr_data_txt(os.path.join(data_dir, dataset_name, f'{dataset_name}_TEST.txt'))

    return X_train, y_train, X_test, y_test


def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5):
    '''
    Performs k-fold CV on a parameter grid. Returns the fitted model with the
    highest CV accuracy.
    '''
    # Custom model for an array of precomputed kernels
    # 1. Stratified K-fold
    cv = StratifiedKFold(n_splits=cv, shuffle=False)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = [] # list of dict, its the same for every split
        # run over the kernels first
        for K_idx, K in enumerate(precomputed_kernels):
            # Run over parameters
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), 
                        train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc)
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    #import pdb; pdb.set_trace()
    # select the best results
    best_idx = np.argmax(fin_results)
    # Return the fitted model and the best_parameters
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]


def ensure_psd(K, tol=1e-8):
    '''
    Helper function to remove negative eigenvalues
    '''
    w,v = eigh(K)
    if (w<-tol).sum() >= 1:
        neg = np.argwhere(w<-tol)
        w[neg] = 0
        Xp = v.dot(np.diag(w)).dot(v.T)
        return Xp
    else:
        return K


def svm_grid_search(name: str,
        X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        pctl: float,
        w_grid: list, 
        param_grid: dict={'C': np.logspace(-3, 5, num=9)},
        gammas: np.ndarray=np.logspace(-4,1,num=6)):
    '''
    Performs k-fold CV on a parameter grid using an SVM classifier. Selects the
    best performing kernelized SVM and calculates its test accuracy.
    '''

    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    logger.info('Starting analysis')
    
    kernel_matrices_train = []
    kernel_matrices_test = []
    kernel_params = []

    for w in w_grid:   
        D_train, D_test = pairwise_subsequence_kernel(
            X_train,
            X_test,
            w,
            pctl,
            wasserstein_kernel
            )

        for g in gammas:
            M_train = np.exp(-g*D_train) 
            M_test = np.exp(-g*D_test)
            # Add psd-ensuring conditions
            M_train = ensure_psd(M_train)
    
            kernel_matrices_train.append(M_train)
            kernel_matrices_test.append(M_test)
            kernel_params.append({'w': w, 'gamma': g})

    svm = SVC(kernel='precomputed')

    # Gridsearch
    gs, best_params = custom_grid_search_cv(svm, param_grid, kernel_matrices_train, y_train, cv=5)
    # Store best parameters
    best_w = kernel_params[best_params['K_idx']]['w']
    best_gamma = kernel_params[best_params['K_idx']]['gamma']
    best_C = best_params['params']['C']
    print(f"Percentile threshold: {pctl}")
    print(f"Best w: {best_w}")
    print(f"Best gamma: {best_gamma}")
    print(f"Best C: {best_C}")
    
    y_pred = gs.predict(kernel_matrices_test[best_params['K_idx']])
    accuracy = accuracy_score(y_test, y_pred)

    logger.info('Accuracy = {:2.2f}'.format(accuracy * 100))
    

def heatmap(data,  fname, show_label=True, row_labels=None, col_labels=None, step=1,
            show_text=True, rotate_xlabel=False, title=None, 
            figsize=(10,10), **kwargs):
    
    '''
    Visualize matrices using heatmap colors.
    '''
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data)
    
    if show_label:
        if row_labels is None:
            row_labels = np.arange(data.shape[0], step=step)
            ax.set_yticks(row_labels)
            ax.set_yticklabels(row_labels, **kwargs)
        else:
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels, **kwargs)
        if col_labels is None:
            col_labels = np.arange(data.shape[1], step=step)
            ax.set_xticks(col_labels)
            ax.set_xticklabels(col_labels, **kwargs)
        else:
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(col_labels, **kwargs)
            # rotate the x labels and set their alignment
            if rotate_xlabel:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
    else:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
            
    if show_text:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(x=j, y=i, s=int(data[i, j]), ha="center", va="center", 
                        color="w")
    if title is not None:
        ax.set_title(title)
        
    fig.tight_layout(pad=0)
    fig.savefig(fname)
    


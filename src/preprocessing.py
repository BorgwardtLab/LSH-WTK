# -*- coding: utf-8 -*-
import numpy as np

    
def mean_center(inp_mtx):
    '''
    Mean-center the rows of the input array.
    
    Parameters
    ----------
    inp_mtx : ndarray
        2-dim array where each row corresponds to a subsequence.
        
    Returns
    -------
    ndarray
        2-dim array whose rows are mean-centered.
    '''
    # row-wise mean
    row_means = np.mean(inp_mtx, axis=1)
    return (inp_mtx.transpose() - row_means).transpose()

def preprocess(inp_mtx):
    '''
    Preprocess subsequences before using LSH.
    
    Parameters
    ----------
    inp_mtx : ndarray
        2-dim array where each row corresponds to a subsequence.
    
    Returns
    -------
    ndarray
        2-dim array whose rows are mean-centered and each entry is
        non-negative.
        
    '''
    mean_centered = mean_center(inp_mtx)
    # shift subsequences into the positive orthant
    return mean_centered - np.min(mean_centered)

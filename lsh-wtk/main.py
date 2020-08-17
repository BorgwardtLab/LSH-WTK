# -*- coding: utf-8 -*-
import sys
sys.path.append('../src')

import numpy as np 
import argparse
from utilities import get_ucr_dataset, svm_grid_search

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_name',
        type=str)
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='../data/UCR'
        )
    parser.add_argument(
        '--pctl',
        type=float, 
        default=95, 
        help='Percentile threshold for the clustering algorithm'
        )
    parser.add_argument(
        '--w_grid',
        nargs='*',
        type=float, 
        default=[0.5, 0.3, 0.1], 
        help='List of subsequence lengths as fractions of time series length'
        )

    args = parser.parse_args()

    X_train, y_train, X_test, y_test = get_ucr_dataset(args.data_dir, args.dataset_name)
    
    # Time series length
    m = X_train.shape[1]
    w_grid = np.rint(m*np.array(args.w_grid)).astype(int)
        
    svm_grid_search(
        args.dataset_name,
        X_train, 
        X_test, 
        y_train, 
        y_test,
        args.pctl,
        w_grid
        )
    

    
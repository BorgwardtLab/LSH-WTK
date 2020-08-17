# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fdir',
        type=str, 
        default='figures/', 
        help='Directory to save the figures'
        )
    
    args = parser.parse_args()

    # Reads in the results
    res = pd.read_excel('results.xlsx', sheet_name='summary')
    lsh_wtk = np.array(res.iloc[:,1])
    wtk = np.array(res.iloc[:,2])
    diff = pd.Series(100*(lsh_wtk - wtk))
        
    # Summary statistics on the accuracy differences
    summary = diff.describe()
    print(summary)
    mode = diff.mode()    
    
    # Histogram of accuracy differences
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    plt.hist(diff, bins=25, facecolor='g')
    plt.xlabel('Accuracy difference (%)')
    plt.ylabel('Counts')
    plt.grid(True)
    plt.tight_layout(pad=0.3)
    plt.savefig(args.fdir + 'diff_hist.pdf')
        
    # Wilcoxon signed rank test
    w, p = wilcoxon(diff)
    print(f'p-value of the Wilcoxon signed rank test: {p}')

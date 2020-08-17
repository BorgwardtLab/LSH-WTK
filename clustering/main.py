# -*- coding: utf-8 -*-
import matplotlib
import argparse
import sys
sys.path.append('..')

from src.lsh_cluster_classes import LSH, Match, Thresholds, Cluster, TS, Signature
from src.utilities import heatmap
from data.synthetic import dat
from src.preprocessing import preprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fdir',
        type=str, 
        default='figures/', 
        help='Directory to save the figures'
        )
    parser.add_argument(
        '--w', 
        type=int, 
        default=15, 
        help='Subsequence length'
        )
    parser.add_argument(
        '--hash_size', 
        type=int, 
        default=15
        )
    parser.add_argument(
        '--num_tables',
        type=int,
        default=30,
        help='Number of hash tables'
        )
    
    args = parser.parse_args()

    # Font style consistent with latex  
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams.update({'font.size': 11.5})
          
    ts = TS(dat)
    fname = args.fdir + 'ts.pdf'
    ylim = ts.plot_ts(fname, figsize=(10,3.5), linewidth=3)
    
    # Extracts subsequences based on a sliding window approach
    subs = ts.subsequences(w=args.w)
    fname = args.fdir + 'subs.pdf'
    ts.plot_subs(subs, ylim, 5, fname, figsize=(5,3.5), linewidth=1.4)
        
    
    # Mean-centers subsequences and shifts them to the positive orthant
    subs_proc = preprocess(subs) 
    
    
    # LSH using the random projection method
    lsh = LSH(num_tables=args.num_tables, hash_size=args.hash_size, inp_mtx=subs_proc)
    lsh.gen_rand_proj_lsh()
    hash_tables = lsh.get_hash_tables()
    
    
    # Computes the match matrix
    mtch = Match(hash_tables)
    match_mtx = mtch.get_match_mtx()
    match_counts = mtch.get_match_counts()
    # Plots the match matrix
    fname = args.fdir + 'match_mtx.pdf'
    heatmap(data=match_mtx, fname=fname, show_label=False, show_text=True)
    
    # Threshold for the binary match matrix is the average match count
    # in the upper triangular matrix of the match matrix without the diagonal
    thr = Thresholds(match_counts)
    threshold = thr.get_avg()
    
    
    cl = Cluster(match_mtx, threshold)
    # Computes the binary match matrix
    binary_match_mtx = cl.get_binary_match_mtx()
    # Plots the binary match matrix
    fname = args.fdir + 'binary_match_mtx.pdf'
    heatmap(data=binary_match_mtx, fname=fname, show_label=False, show_text=False)
    
    
    # Finds maximal cliques based on the binary match matrix
    cluster_mtx = cl.get_cliques(binary_match_mtx)
    # Plots the cluster matrix
    fname = args.fdir + 'cluster_mtx.pdf'
    heatmap(data=cluster_mtx, fname=fname, show_label=False, show_text=False, 
            figsize=(8.5,2.5))
    
    
    # Visualizes subsequences in clusters
    fname = args.fdir + 'clusters.pdf'
    ts.visualize_clusters(subs, cluster_mtx, ylim, 5, fname, figsize=(5,3.5), 
                          linewidth=1.4)
    
    
    # Signature representation of the time series
    sgn = Signature(subs, cluster_mtx)
    hist = sgn.get_hist()
    mean_subs = sgn.get_mean_subs()
    fname = args.fdir + 'mean_subs.pdf'
    sgn.plot_representative_subs(mean_subs, hist, ylim, fname, figsize=(2.5,2.8), 
                                 linewidth=2)

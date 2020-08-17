# -*- coding: utf-8 -*-
from lsh_cluster_classes import LSH, Match, Thresholds, Cluster, TS, Signature
from preprocessing import preprocess


def LSHcluster(tseries, w, hash_size=15, num_tables=30, pctl=0):
    '''
    LSH based clustering algorithm for time series.
    
    Parameters
    ----------
    tseries : ndarray
        1-dim array containing the time series
    w : int
        Subsequence length.
    hash_size : int, optional
        Length of the binary hash vector.
    num_tables : int, optional
        Number of hash tables.
    pctl : float, optional
        Given percentile of the match counts to use as a threshold.
    
    Returns
    -------
    mean_subs : ndarray
        2-dim array where each row corresponds to the mean subsequence of a 
        cluster.
    weigths : ndarray
        1-dim array of weigths corresponding to the fractions of subsequences
        present in the clusters.
    '''
    ts = TS(tseries)
    subs = ts.subsequences(w)
    subs_proc = preprocess(subs)
    
    # LSH using the random projection method
    lsh = LSH(num_tables, hash_size, subs_proc)
    lsh.gen_rand_proj_lsh()
    hash_tables = lsh.get_hash_tables()
    
    # computing the match matrix
    mtch = Match(hash_tables)
    match_mtx = mtch.get_match_mtx()
    match_counts = mtch.get_match_counts()
    
    # computing the threshold for the similarity matrix
    thr = Thresholds(match_counts)
    threshold = thr.get_percentile(pctl, num_tables)
    
    cl = Cluster(match_mtx, threshold)
    # computing the binary match matrix
    binary_match_mtx = cl.get_binary_match_mtx()
    
    # computing the cliques based on the binary match matrix
    cluster_mtx = cl.get_cliques(binary_match_mtx)
    
    # create a signature from the clusters
    sgn = Signature(subs, cluster_mtx)
    weights = sgn.get_weights()
    mean_subs = sgn.get_mean_subs()
    
    return mean_subs, weights


def LSHclusterOverlap(tseries, w, hash_size=15, num_tables=30, pctl=0, overlap_ratio=0.2):
    '''
    LSH based clustering algorithm for time series. It uses overlap restrictions
    between subsequences when forming clusters.
    
    Parameters
    ----------
    tseries : ndarray
        1-dim array containing the time series
    w : int
        Subsequence length.
    hash_size : int, optional
        Length of the binary hash vector.
    num_tables : int, optional
        Number of hash tables.
    pctl : float, optional
        Given percentile of the match counts to use as a threshold.
    overlap_ratio : float
        Upper bound on the overlap ratios between subsequences. It must be
        between 0 and 1.
    
    Returns
    -------
    mean_subs : ndarray
        2-dim array where each row corresponds to the mean subsequence of a 
        cluster.
    weigths : ndarray
        1-dim array of weigths corresponding to the fractions of subsequences
        present in the clusters.
    '''
    
    ts = TS(tseries)
    subs = ts.subsequences(w)
    subs_proc = preprocess(subs)
    
    # LSH using the random projection method
    lsh = LSH(num_tables, hash_size, subs_proc)
    lsh.gen_rand_proj_lsh()
    hash_tables = lsh.get_hash_tables()
    
    # computing the match matrix
    mtch = Match(hash_tables)
    match_mtx = mtch.get_match_mtx()
    match_counts = mtch.get_match_counts()
    
    # computing the threshold for the similarity matrix
    thr = Thresholds(match_counts)
    threshold = thr.get_percentile(pctl, num_tables)
    
    cl = Cluster(match_mtx, threshold)
    # computing the binary match matrix
    binary_match_mtx = cl.get_binary_match_mtx()
    
    # computing the cliques based on the binary match matrix
    cluster_mtx = cl.get_cliques_overlap(binary_match_mtx, w, overlap_ratio)
    
    # create a signature from the clusters
    sgn = Signature(subs, cluster_mtx)
    weights = sgn.get_weights()
    mean_subs = sgn.get_mean_subs()
    
    return mean_subs, weights




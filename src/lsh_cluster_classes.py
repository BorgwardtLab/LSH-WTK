# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class LSH:
    '''
    Class containing methods for Locality-Sensitive Hashing.
    
    Attributes
    ----------
    hash_size : int
        Length of the hash value.
    num_tables : int
        Number of hash tables to use.
    inp_mtx  : ndarray
        2-dim array where each row corresponds to a subsequence.
    num_subs : int
        Number of subsequences in the dataset.
    subs_length: int
        Length of the subsequences.
    hash_tables : ndarray
        3-dim array where the first dimension corresponds to 2-dim hash tables.
    
    Methods
    -------
    gen_rand_proj_lsh()
        Generates a 3-dim array containing the hash tables
        by applying the random projection method of LSH.
    gen_spherical_lsh()
        Generates a 3-dim array containing the hash tables
        by applying spherical (cross-polytope) LSH.
    
    '''
    def __init__(self, num_tables, hash_size, inp_mtx):
        self.hash_size = hash_size
        self.num_tables = num_tables
        self.num_subs = inp_mtx.shape[0]
        self.subs_length = inp_mtx.shape[1]
        self.inp_mtx = inp_mtx
        self.hash_tables = np.empty((num_tables, self.num_subs, hash_size))
        np.random.seed(35)
           
    def gen_rand_proj_lsh(self):
        for idx in range(self.num_tables):
            R = np.random.randn(self.subs_length, self.hash_size)
            self.hash_tables[idx] = (np.dot(self.inp_mtx, R) > 0).astype('int')

    def single_spherical_lsh(self):
        # sphere projection
        l2 = np.linalg.norm(self.inp_mtx, axis=-1, ord=2)
        unit_mtx = self.inp_mtx / l2.reshape(self.num_subs, 1)
        # initialize hash table
        hash_table = np.empty((self.num_subs, self.hash_size))
        for idx in range(self.hash_size):
            R = np.random.randn(self.subs_length, self.subs_length)
            rot = np.dot(unit_mtx, R)    
            # sphere projection of the rotated vectors
            l2_rot = np.linalg.norm(rot, axis=-1, ord=2)
            unit_rot = rot / l2_rot.reshape(self.num_subs, 1)    
            concat = np.concatenate((unit_rot, -unit_rot), axis=1)            
            # hash value is the coordinate with the maximum value
            hash_table[:,idx] = np.argmax(concat, axis=1)
        return hash_table
        
    def gen_spherical_lsh(self):
        for idx in range(self.num_tables):
            self.hash_tables[idx] = self.single_spherical_lsh()
            
    def get_hash_tables(self):
        return self.hash_tables
     
               
class Match:
    '''
    Creates match matrices for each hash table. Given a hash table,
    the (i,j)-th element of the match matrix indicates whether subsequences
    s_i and s_j have the exact same hash value in the table.
    
    Attributes
    ----------
    hash_tables : ndarray
        3-dim array where the first dimension corresponds to 2-dim hash tables.
    num_tables : int
        Number of hash tables.
    num_subs : int
        Number of subsequences in the dataset.
        
    Methods
    -------
    get_match_matrices()
        Returns a 3-dim array where the first dimension corresponds to
        2-dim match matrices. Each match matrix corresponds to a hash table.
    get_match_matrix()
        Returns a 2-dim array match matrix where the (i,j)-th element
        is the sum of the (i,j)-th elements of all the match matrices. 
    get_match_counts()
        Returns an array containing the upper triangular matrix entries 
        (excluding the diagonal) of the match matrix.
    '''
    
    def __init__(self, hash_tables):
        self.num_tables = hash_tables.shape[0]
        self.num_subs = hash_tables.shape[1]
        self.hash_tables = hash_tables
        
    def get_match_matrices(self):
        match_matrices = np.zeros((self.num_tables, self.num_subs, self.num_subs))
        for idx in range(self.num_tables):
            for i, h_i in enumerate(self.hash_tables[idx]):
                for j, h_j in enumerate(self.hash_tables[idx]):
                    match_matrices[idx, i, j] = 1*np.array_equal(h_i, h_j)
        return match_matrices
    
    def get_match_mtx(self):
        match_matrices = self.get_match_matrices()
        return np.sum(match_matrices, axis=0)
    
    def get_match_counts(self):
        match_mtx = self.get_match_mtx()
        match_counts = match_mtx[np.triu_indices(len(match_mtx), 1)]
        return match_counts
        

class Thresholds:
    '''
    Collection of methods to calculate the threshold for the binary match matrix.
    
    Attributes
    ----------
    match_counts : ndarray
        Array containing the upper triangular matrix entries 
        (excluding the diagonal) of the match matrix.
        
    '''
    
    def __init__(self, match_counts):
        self.match_counts = match_counts
        
    def get_avg(self):
        '''
        Returns the average of the match counts.
        '''
        return np.mean(self.match_counts)
    
    def get_percentile(self, q, num_tables=None):
        '''
        Returns the q-th percentile of the match counts. If `q` is larger than
        100, a value larger than num_tables is returned.
        
        Parameters
        ----------
        q : float
            Percentile to compute. It must be between 0 and 100 inclusive. If
            it is larger than 100, a value bigger than num_tables is returned.
        num_tables: int
            Number of hash tables used in the clustering algorithm.
        '''
        if q > 100:
            return num_tables + 1
        else:
            return np.percentile(self.match_counts, float(q))


class Cluster:
    '''
    Collection of methods to form clusters of subsequences.
    
    Attributes
    ----------
    match_mtx : ndarray
        2-dim array containing the match counts for each pair of 
        subsequences.
    threshold : float
        Threshold to use for computing the binary match matrix.
    num_sub : int
        Number of subsequences of the time series.
    '''
    
    def __init__(self, match_mtx, threshold):
        self.match_mtx = match_mtx
        self.threshold = threshold
        self.num_sub = len(match_mtx)
    
    def get_binary_match_mtx(self):
        return self.match_mtx >= self.threshold        
                
    def get_cliques(self, binary_match_mtx):
        '''
        Finds maximal cliques using the adjacency matrix created from the 
        binary match matrix.
        
        Parameters
        ----------
        binary_match_mtx : ndarray
            2-dim binary array where the (i,j)-th element indicates whether the 
            corresponding match count is not smaller than a threshold.
        
        Returns
        -------
        clique_mtx : ndarray
            2-dim binary array where each row corresponds to a maximal clique.
        '''
        # adjacency matrix of the undirected graph
        adj_mtx = binary_match_mtx.astype(int)
        np.fill_diagonal(adj_mtx, 0)
        nodes = np.arange(self.num_sub)
        # initialize clique matrix
        clique_mtx = list()
        
        while len(nodes) > 0:
            # adjacency matrix of the subgraph
            sub_adj_mtx = adj_mtx[np.ix_(nodes, nodes)]
            # degrees of the nodes
            degrees = np.sum(sub_adj_mtx, axis=0)
            # index of the node with the highest degree
            v_ind = np.argmax(degrees)
            # find a maximal clique for vertex v
            clique = np.zeros(len(nodes), dtype=int)
            clique[v_ind] = 1
            clique_size = 1
            # indices of other nodes to test for inclusion in the clique
            others_ind = np.delete(np.arange(len(nodes)), v_ind)
            
            # find a maximal clique (not necessarily the maximum clique)
            for j in others_ind:
                if np.dot(clique,sub_adj_mtx[j]) == clique_size:
                    clique[j] = 1
                    clique_size += 1
            clique_row = np.zeros(self.num_sub)
            clique_row[nodes[clique.astype(bool)]] = 1
            clique_mtx.append(clique_row)
            nodes = nodes[~clique.astype(bool)]
            
        clique_mtx = np.array(clique_mtx)
        return clique_mtx

    def get_cliques_overlap(self, binary_match_mtx, w, overlap_ratio):
        '''
        Finds maximal cliques using the adjacency matrix created from the 
        binary match matrix. It applies overlap restrictions between
        subsequences when forming cliques.
        
        Parameters
        ----------
        binary_match_mtx : ndarray
            2-dim binary array where the (i,j)-th element indicates whether the 
            corresponding match count is not smaller than a threshold.
        w : int
            Subsequence length.
        overlap_ratio : float
            Upper bound on the overlap ratios between subsequences. It must be
            between 0 and 1.
        
        Returns
        -------
        clique_mtx : ndarray
            2-dim binary array where each row corresponds to a maximal clique.
        '''
        window = np.ceil(0.5 * w * (1 - overlap_ratio))
        # adjacency matrix of the undirected graph
        adj_mtx = binary_match_mtx.astype(int)
        np.fill_diagonal(adj_mtx, 0)
        if np.sum(adj_mtx) == 0:
            return np.diag(np.ones(len(adj_mtx)))
        nodes = np.arange(self.num_sub)
        # initialize clique matrix
        clique_mtx = list()
        
        while len(nodes) > 0:
            # adjacency matrix of the subgraph
            sub_adj_mtx = adj_mtx[np.ix_(nodes, nodes)]
            # degrees of the nodes
            degrees = np.sum(sub_adj_mtx, axis=0)
            # index of the node with the highest degree
            v_ind = np.argmax(degrees)
            # find a maximal clique for vertex v
            clique = np.zeros(len(nodes), dtype=int)
            clique[v_ind] = 1
            clique_size = 1
            # indices of other nodes to test for inclusion in the clique
            others_ind = np.ones(len(nodes), dtype=int)
            # start node of the overlap window
            wstart = int(max(0, v_ind-window))
            # end node of the overlap window
            wend = int(min(len(nodes), v_ind+window))
            # do not test nodes for inclusion in the clique within the overlap window
            others_ind[wstart:wend+1] = 0
            
            while sum(others_ind) > 0:
                other = np.argmax(others_ind)
                if np.dot(clique,sub_adj_mtx[other]) == clique_size:
                    clique[other] = 1
                    clique_size += 1
                    wstart = int(max(0, other - window))
                    wend = int(min(len(nodes), other + window))
                    others_ind[wstart:wend+1] = 0
                else:
                    others_ind[other] = 0
                
            clique_row = np.zeros(self.num_sub)
            clique_row[nodes[clique.astype(bool)]] = 1
            clique_mtx.append(clique_row)
            nodes = nodes[~clique.astype(bool)]
            
        clique_mtx = np.array(clique_mtx)
        return clique_mtx
        
class TS:
    '''
    Class for extracting and visualizing subsequences of a time series. 
    
    Attributes
    ----------
    ts : ndarray
        1-dim array containing the time series.
    m : int
        Time series length.
    '''
    
    def __init__(self, ts):
        self.ts = np.asarray(ts)
        self.m = ts.size
        
    def subsequences(self, w):
        '''
        Extracts subsequences based on a sliding window approach.
        Bock et al. (2019)
        
        Parameters
        ----------
        w : int
            Subsequence length.
        
        Returns
        -------
        ndarray
            2-dim array where each row corresponds to a w-length subsequence.
        '''
        shape = (self.m - w + 1, w)
        strides = self.ts.strides * 2
        
        return np.lib.stride_tricks.as_strided(self.ts, shape=shape, 
                                                     strides=strides)  
    
    def subsequences_stride(self, w, stride_per_w):
        '''
        Extract subsequences based on a sliding window approach with stride.
        
        Parameters
        ----------
        w : int
            Subsequence length.
        stride_per_w : float
            Ratio of the stride over subsequences length. It must be between
            0 and 1.
        
        Returns
        -------
         ndarray
            2-dim array where each row corresponds to a w-length subsequence.
        '''
        stride = int(np.round(w * stride_per_w))
        num_subs = int(np.floor((self.m - w + 1)/stride))
        
        shape = (num_subs, w)
        strides = (stride * self.ts.strides[0], self.ts.strides[0])
        
        return np.lib.stride_tricks.as_strided(self.ts, shape=shape, 
                                                     strides=strides)

    def plot_ts(self, fname, figsize, **kwargs):
        '''
        Plots the time series such that the figure follows a specific 
        formatting.
        
        Parameters
        ----------
        fname : str
            Name of the figure including its path.
        figsize : tuple
            Size of the figure
            
        Returns
        -------
        ylim : tuple
            Limits of the y axis.
        '''
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.ts, **kwargs)
        ax.set_xlim(0,self.m-1)
        
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        fig.tight_layout(pad=0)
        fig.savefig(fname)
    
        return ax.get_ylim()
    
        
    def plot_subs(self, subs, ylim, num_rows, fname, figsize, text_labels=[],
                  **kwargs):
        '''
        Visualizes the extracted subsequences in a grid of subplots 
        
        Parameters
        ----------
        subs : ndarray
            2-dim array where each row corresponds to a w-length subsequence.
        ylim : tuple
            Limits of the y axis.
        num_rows : int
            Number of rows in the grid.
        fname : str
            Name of the figure including its path.
        figsize : tuple
            Size of the figure.
        text_labels : list, optional
            Labels (indices) to show on specific subplots.
            
        '''
        num_subs = subs.shape[0]
        w = subs.shape[1]
        num_cols = int(np.ceil(num_subs / num_rows))
        
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, 
                                sharex='col', sharey='row', 
                                gridspec_kw={'hspace': 0, 'wspace': 0},
                                figsize=figsize)
        
        count = 0
        for ax_i in axes:
            for ax_j in ax_i:
                if count < num_subs:
                    ax_j.plot(subs[count,:], **kwargs)
                    ax_j.set_ylim(ylim)
                    ax_j.set_xlim((0,w-1))
                    if count in text_labels:
                        ax_j.text(x=0.85, y=0.85, s=count, color='tab:blue', 
                                  ha='center', va='center', transform=ax_j.transAxes)
                    ax_j.xaxis.set_visible(False)
                    ax_j.yaxis.set_visible(False)
                else:
                    plt.delaxes(ax_j)
                count = count + 1
        
        fig.tight_layout(pad=0)
        fig.savefig(fname)
        
    def plot_single_subs(self, subs, ind, ylim, fname, figsize, **kwargs):
        '''
        Plots a single w-length subsequence.
        
        Parameters
        ----------
        subs : ndarray
            2-dim array where each row corresponds to a w-length subsequence.
        ind : int
            Index of the subsequence to be plotted.
        ylim : tuple
            Limits of the y axis.
        fname : str
            Name of the figure including its path.
        figsize : tuple
            Size of the figure.
        '''
        
        w = subs.shape[1]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(subs[ind,:], **kwargs)
        ax.set_ylim(ylim)
        ax.set_xlim((0,w-1))
        ax.text(x=0.85, y=0.85, s=ind, color='tab:blue', ha='center',
                  va='center', transform=ax.transAxes)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        fig.tight_layout(pad=0)
        fig.savefig(fname)
             
    def visualize_clusters(self, subs, cluster_mtx, ylim, num_rows, fname,
                           figsize, text_labels=[], **kwargs):
        '''
        Visualizes subsequences in clusters.
        
        Parameters
        ----------
        subs : ndarray
            2-dim array where each row corresponds to a w-length subsequence.
        cluster_mtx : ndarray
            2-dim binary array where each row corresponds to a cluster.
        ylim : tuple
            Limits of the y axis.
        fname : str
            Name of the figure including its path.
        figsize : tuple
            Size of the figure.
        text_labels : list, optional
            Labels (indices) to show on specific subplots.
        '''
        
        colors = cm.nipy_spectral(np.linspace(0, 1, len(cluster_mtx)))
        num_subs = subs.shape[0]
        w = subs.shape[1]
        num_cols = int(np.ceil(num_subs / num_rows))
        
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, 
                                sharex='col', sharey='row', 
                                gridspec_kw={'hspace': 0, 'wspace': 0},
                                figsize=figsize)
        
        count = 0
        for ax_i in axes:
            for ax_j in ax_i:
                if count < num_subs:
                    cluster_ind = int(np.where(cluster_mtx[:,count] == 1)[0])
                    ax_j.plot(subs[count,:], color=colors[cluster_ind], 
                              **kwargs)
                    ax_j.set_ylim(ylim)
                    ax_j.set_xlim((0,w-1))
                    if count in text_labels:
                        ax_j.text(x=0.85, y=0.85, s=count, color=colors[cluster_ind],
                                  ha='center', va='center', transform=ax_j.transAxes)
                    ax_j.xaxis.set_visible(False)
                    ax_j.yaxis.set_visible(False)
                else:
                    plt.delaxes(ax_j)
                count = count + 1
        
        fig.tight_layout(pad=0)
        fig.savefig(fname)
        
    def plot_single_subs_in_cluster(self, subs_mtx, subs_ind, cluster_ind, 
                                    cluster_mtx, ylim, fname, figsize, **kwargs):
        '''
        Plots a single w-length subsequence using the color of its cluster.
        
        Parameters
        ----------
        subs_mtx : ndarray
            2-dim array where each row corresponds to a w-length subsequence.
        subs_ind : int
            Index of the subsequence to be plotted.
        cluster_ind : int
            Index of its cluster.
        cluster_mtx : ndarray
            2-dim binary array where each row corresponds to a cluster.
        ylim : tuple
            Limits of the y axis.
        fname : str
            Name of the figure including its path.
        figsize : tuple
            Size of the figure.
        '''
        colors = cm.nipy_spectral(np.linspace(0, 1, len(cluster_mtx)))
        w = subs_mtx.shape[1]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(subs_mtx[subs_ind,:], color=colors[cluster_ind], **kwargs)
        ax.set_ylim(ylim)
        ax.set_xlim((0,w-1))
        ax.text(x=0.85, y=0.85, s=subs_ind, color=colors[cluster_ind],
                ha='center', va='center', transform=ax.transAxes)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        fig.tight_layout(pad=0)
        fig.savefig(fname)
        
        
class Signature:
    '''
    Collection of methods that create signature represenation of a time series
    based on its subsequences.
    
    Methods
    -------
    get_weights()            
        Assigns a weight to each cluster based on the fraction of subsequences
        present in the cluster.
    get_mean_subs()
        Returns the mean subsequences for the clusters.
    '''
    
    def __init__(self, subs, cluster_mtx):
        self.subs = subs
        self.cluster_mtx = cluster_mtx
        self.w = subs.shape[1]
        self.num_subs = subs.shape[0]
        self.num_clusters = cluster_mtx.shape[0]
        # Time series length
        self.m = self.num_subs + self.w - 1
    
    def get_hist(self):
        hist = np.sum(self.cluster_mtx, axis=1)
        return hist.astype(int)
    
    def get_weights(self):
        hist = self.get_hist()
        weights = hist/self.num_subs
        return weights
    
    def get_mean_subs(self):
        mean_subs = np.empty((self.num_clusters, self.w))
        for i, clust_i in enumerate(self.cluster_mtx):
            ind = np.where(clust_i)[0]
            clust_subs = self.subs[ind]
            mean_subs[i] = np.mean(clust_subs, axis=0)
        return mean_subs
    
    def plot_representative_subs(self, repr_subs, hist, ylim, fname, figsize, 
                                 modify_text=[], **kwargs):
        '''
        For each cluster, it plots the mean subsequence and indicates the 
        number of cluster members.
        
        Parameters
        ----------
        repr_subs : ndarray
            2-dim array where each row corresponds to the mean subsequence of a 
            cluster.
        hist : ndarray
            1-dim array containing the number of subsequences in each cluster.
        '''
        colors = cm.nipy_spectral(np.linspace(0, 1, self.num_clusters))
        num_cols = int(np.floor(np.sqrt(self.num_clusters)))
        num_rows = int(np.ceil(self.num_clusters / num_cols))
        
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, 
                                sharex='col', sharey='row', 
                                gridspec_kw={'hspace': 0, 'wspace': 0},
                                figsize=figsize)
        
        count = 0
        for ax_i in axes:
            for ax_j in ax_i:
                if count < self.num_clusters:
                    ax_j.plot(repr_subs[count,:], color=colors[count], **kwargs)
                    ax_j.set_ylim(ylim)
                    ax_j.set_xlim((0,self.w-1))
                    if count in modify_text:
                        ax_j.text(x=0.5, y=0.85, s=hist[count], color='k', ha='center',
                                  va='center', transform=ax_j.transAxes)
                    else:
                        ax_j.text(x=0.85, y=0.85, s=hist[count], color='k', ha='center',
                                  va='center', transform=ax_j.transAxes)
                    ax_j.xaxis.set_visible(False)
                    ax_j.yaxis.set_visible(False)
                else:
                    plt.delaxes(ax_j)
                count = count + 1
        
        fig.tight_layout(pad=0)
        fig.savefig(fname)
        
        
            
            
            
            


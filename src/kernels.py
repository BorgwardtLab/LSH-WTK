import numpy as np
import ot
from lsh_cluster import LSHclusterOverlap


def pairwise_subsequence_kernel(
    time_series_train,
    time_series_test,
    w,
    pctl,
    functor,
    **kwargs
    ):
    '''
    Applies a calculation functor to all pairs of a data set. As
    a result, two matrices will be calculated:

    1. The square matrix between all pairs of training samples
    2. The rectangular matrix between test samples (rows), and
       training samples (columns).

    These matrices can be fed directly to a classifier.

    Notice that this function will apply the kernel to *each* of
    the subsequences in both time series.
    '''

    n = len(time_series_train)
    m = len(time_series_test)


    K_train = np.zeros((n, n))  # Need to initialize with zeros for symmetry
    K_test = np.empty((m, n))   # Since this is rectangular, no need for zeros


    subsequences_train = dict()
    subsequences_test = dict()


    for i, ts_i in enumerate(time_series_train):
        mean_subs, weights = LSHclusterOverlap(ts_i, w, pctl=pctl)
        subsequences_train[i] = mean_subs
        subsequences_train['weights_'+str(i)] = weights

    for i, ts_i in enumerate(time_series_test):
        mean_subs, weights = LSHclusterOverlap(ts_i, w, pctl=pctl)
        subsequences_test[i] = mean_subs
        subsequences_test['weights_'+str(i)] = weights
    

    # Evaluate the functor for *all* relevant pairs, while filling up
    # the initially empty kernel matrices.
    for i, ts_i in enumerate(time_series_train):
        for j, ts_j in enumerate(time_series_train[i:]):
            s_i = subsequences_train[i]     # first shapelet
            w_i = subsequences_train['weights_' + str(i)]
            
            s_j = subsequences_train[i + j] # second shapelet
            w_j = subsequences_train['weights_' + str(i + j)]
            
            K_train[i, i + j] = functor(s_i, s_j, w_i, w_j, **kwargs)

        for j, ts_j in enumerate(time_series_test):
            s_i = subsequences_train[i] # first shapelet
            w_i = subsequences_train['weights_' + str(i)]

            # Second shapelet; notice that there is no index shift in
            # comparison to the code above.
            s_j = subsequences_test[j]
            w_j = subsequences_test['weights_' + str(j)]

           
            # Fill the test matrix; since it has different dimensions
            # than the training matrix, the indices are swapped here.
            K_test[j, i] = functor(s_i, s_j, w_i, w_j, **kwargs)


    # Makes the matrix symmetric since we only fill the upper diagonal
    # in the code above.
    K_train = K_train + K_train.T

    return K_train, K_test


def wasserstein_kernel(subsequences_1, subsequences_2, weights_1=[], weights_2=[], 
                       metric='euclidean'):
    '''
    Calculates the distance between two time series using their
    corresponding set of subsequences. The metric used to align
    them may be optionally changed.
    '''
        
    C = ot.dist(subsequences_1, subsequences_2, metric=metric)        
    return ot.emd2(weights_1, weights_2, C)


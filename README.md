# Generalizing the Wasserstein Time Series Kernel Using Locality-Sensitive Hashing
This is the source code for my master thesis submitted on August 17, 2020. The thesis proposes a kernel method (LSH-WTK) that generalizes the Wasserstein Time Series Kernel (WTK). It builds upon WTK by utilizing the distribution of time series shapelets (subsequences). 'Histograms' of subsequences are created by clustering subsequences of similar shape applying a novel clustering algorithm based on Locality-Sensitive Hashing. The files are organized in the following way.

## clustering
The folder contains the code for the illustration of the LSH based clustering algorithm on a synthetic time series. To generate the figures used in the thesis, simply execute the `main.py` script. You can also use different subsequence lengths `w`, hash sizes and number of hash rounds `num_tables` in the clustering algorithm.
```
$ python main.py --w 15 --hash_size 15 --num_tables 30
```

## data
The folder contains the synthetic time series and a small dataset called Chinatown from the [UEA & UCR archive](http://www.timeseriesclassification.com/) which can be used to run our kernelized SVM classifier.

## lsh-wtk
This folder contains the source code of our classifier algorithm. To run LSH-WTK on the example dataset, simply execute the `main.py` script and provide the name of the dataset.
```
$ python main.py Chinatown
```
As LSH-WTK is a generalization of WTK, it is possible to run the latter method by specifying the percentile threshold to be larger than 100.
```
$ python main.py Chinatown --pctl 101
```

## results
The folder contains all the accuracy results of our method reported in the thesis. `analisys.py` produces the summary statistics and the histogram of the accuracy differences between LSH-WTK and WTK. It also performs the Wilcoxon signed-rank test on the differences.

## src
Contains the source codes of the LSH based clustering algorithm and LSH-WTK.

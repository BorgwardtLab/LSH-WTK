# -*- coding: utf-8 -*-
'''
Generates a synthetic time series with two peaks. 
'''

import numpy as np
import scipy.stats as stats

np.random.seed(1)

mu = 0
variance = 0.4
sigma = np.sqrt(variance)
length = 35
x = np.linspace(mu - 3 * sigma, mu + 10 * sigma, length)
dat = stats.norm.pdf(x, mu, sigma)

add = dat[:17]
dat = np.concatenate((dat,add), axis=None)

# add some noise 
noise = np.random.normal(loc=0, scale=0.005, size=(len(dat)))
dat = dat + noise



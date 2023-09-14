#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(datestr)s

@author: %(username)s
"""

#%%
# %pylab
# %load_ext autoreload
# %autoreload 2
#%reload_ext autoreload

import numpy as np

#%%
""" 1000x30 pixel mock image """
n_row = 128
n_col = 30

mock_spec = (np.cos(np.linspace(0,20,n_row))+10)*0+1
#plot(mock_spec)

data = np.zeros((n_row, n_col))
for i in range(n_row):
    prof = np.histogram(np.random.randn(10000),density=True, bins=np.linspace(-4,4,n_col+1))[0]
    prof /= np.sum(prof)
    data[i] = prof*mock_spec[i]

#%%

def extract_sum(im_chunk):
    """ dispersion axis is 0 """
    return np.sum(im_chunk, axis=1)

figure();
plot(extract_sum(data))
plot(mock_spec)
plot(spec_extr, 'b')
plot(spec_extr_w, 'r')

figure()
hist(spec_extr-mock_spec, alpha=0.5)
hist(spec_extr_w-mock_spec, alpha=0.5)

figure()
hist(spec_extr/mock_spec, alpha=0.5)
hist(spec_extr_w/mock_spec, alpha=0.5)
#%%

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


x_test = np.linspace(-4,4,n_col+1)
x_test = 0.5*np.diff(x_test)+x_test[:-1]
pgaussian = gaussian(x_test, 0,1)
#%%
G = 1.
sRON = 0.

Wxl = 1/(data/G+sRON**2.)
Wxl = np.where(np.isfinite(Wxl)&(Wxl>0), Wxl, 0.)

prof_init = np.repeat(np.median(data,axis=0).reshape(1,-1), n_row, axis=0)
# perfect gaussian
prof_init = np.repeat(pgaussian.reshape(1,-1), n_row, axis=0)
prof_norm = prof_init/np.sum(prof_init, axis=1).reshape(-1,1)

prof_norm_rmax = np.max(prof_norm, axis=1)
prof_norm *= (prof_norm>prof_norm_rmax.reshape(-1,1)*0.05)

spec_extr_w = np.sum(Wxl*data*prof_norm, axis=1)/np.sum(Wxl*prof_norm**2., axis=1)
spec_extr = np.sum(data*prof_norm, axis=1)/np.sum(prof_norm**2., axis=1)


#%%
i = 65
figure();
plot(data[i])
plot(prof_norm[i]*spec_extr[i])
#%%
figure()
plot(data.T)
prof_init = np.median(data,axis=0)
plot(prof_init, 'r', lw=9)

#%%

figure()
imshow(data, aspect="auto")




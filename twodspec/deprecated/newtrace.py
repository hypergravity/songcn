#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
#%%
import numpy as np
from astropy.io import fits
from laspec.normalization import normalize_spectrum_iter
from scipy import signal

from twodspec import extract
from twodspec.aperture import Aperture
from matplotlib.pyplot import figure, plot, imshow

#%%
def gaussian_kernel(sigma=2., n_sigma=5):
    """ generate Gaussian kernel

    Parameters
    ----------
    sigma : float, optional
        The sigma of the Gaussian. The default is 2..
    n_sigma : flaot, optional
        The width of the Gaussian in terms of sigma. The default is 5.

    Returns
    -------
    y : ndarray
        The Gaussian kernel array.

    """
    npix = np.int(n_sigma*sigma)+1
    x = np.arange(-npix, npix+1, 1)
    y = np.exp(-0.5*(x/sigma)**2)
    y /= np.sum(y)
    return y


def box_kernel(fullwidth=10):
    npix = np.int(fullwidth)
    kernel = np.ones(npix)
    kernel /= np.sum(kernel)
    return kernel

#%%
def trace_naive_max(flat, sigma=7., maxdev=7.):
    """ trace aperture using naive method -- along local max

    Parameters
    ----------
    flat : ndarray
        FLAT.
    sigma : float, optional
        The sigma of Gaussian kernel. The default is 7..
    maxdev : float, optional
        The max deviation of local max. The default is 7.

    Returns
    -------
    ap_center : ndarray (n_ap, n_pix)
        The array for center of aperture.

    """
    flat = np.asarray(flat, float)
    n_row, n_col = flat.shape
    i_row_center = np.int(n_row/2)
    
    # smooth flat along spacial axis
    flat_smooth = np.zeros_like(flat)
    gkernel = gaussian_kernel(sigma)
    for i_row in range(n_row):
        flat_smooth[i_row] = signal.fftconvolve(flat[i_row], gkernel, mode="same")
    
    # find local max
    flat_smooth_diff = np.diff(flat_smooth, axis=1)
    flat_localmax = np.zeros_like(flat, bool)
    flat_localmax[:,1:-1] = (flat_smooth_diff[:,1:]<0)&(flat_smooth_diff[:,:-1]>0)
    
    # start from center row
    ind_col_localmax, = np.where(flat_localmax[i_row_center])
    n_ap_try = len(ind_col_localmax)
    ap_center_try = np.ones((n_ap_try, n_row)) *np.nan
    ap_center_try[:, i_row_center] = ind_col_localmax
    
    # find local max col each row 
    each_row_localmax = [np.where(flat_localmax[i_row])[0] for i_row in range(n_row)]
    
    # loop for trials
    for i_ap in range(n_ap_try):
        for i_row in range(i_row_center-1, 0-1, -1):
            this_d = np.abs(each_row_localmax[i_row]-ap_center_try[i_ap, i_row+1])
            if np.min(this_d) <= sigma:
                ap_center_try[i_ap, i_row] = each_row_localmax[i_row][np.argmin(this_d)]
            else:
                break
        for i_row in range(i_row_center+1, n_row, 1):
            this_d = np.abs(each_row_localmax[i_row]-ap_center_try[i_ap, i_row-1])
            if np.min(this_d) <= sigma:
                ap_center_try[i_ap, i_row] = each_row_localmax[i_row][np.argmin(this_d)]
            else:
                break
    ind_good_ap = np.sum(np.isfinite(ap_center_try), axis=1) == n_row
    n_ap = np.sum(ind_good_ap)
    ap_center = ap_center_try[ind_good_ap]
    
    return ap_center

flat = fits.getdata("/Users/cham/projects/song/star_spec/20191105/night/ext/masterflat_20191105_slit5.fits")
ap_center = trace_naive_max(flat, sigma=7, maxdev=10)
plot(ap_center_try.T, np.arange(2048))
imshow(np.log10(flat))
print(ap_center.shape)
#%%
def trace_naive_min(flat, sigma=7., maxdev=5):
    flat = np.asarray(flat, float)
    n_row, n_col = flat.shape
    i_row_center = np.int(n_row/2)
    
    # smooth flat
    flat_smooth = np.zeros_like(flat)
    gkernel = gaussian_kernel(sigma)
    for i_row in range(n_row):
        flat_smooth[i_row] = signal.fftconvolve(flat[i_row], gkernel, mode="same")
    
    # find local max
    flat_smooth_diff = np.diff(flat_smooth, axis=1)
    flat_localmax = np.zeros_like(flat, bool)
    flat_localmax[:,1:-1] = (flat_smooth_diff[:,1:]>0)&(flat_smooth_diff[:,:-1]<0)
    
    # start from center row
    #maxdev = np.int(sigma)
    ind_col_localmax, = np.where(flat_localmax[i_row_center])
    n_ap_try = len(ind_col_localmax)
    ap_center_try = np.ones((n_ap_try, n_row)) *np.nan
    ap_center_try[:, i_row_center] = ind_col_localmax
    
    # each row local max col
    each_row_localmax = [np.where(flat_localmax[i_row])[0] for i_row in range(n_row)]
    
    # loop for trials
    for i_ap in range(n_ap_try):
        for i_row in range(i_row_center-1, 0-1, -1):
            this_d = np.abs(each_row_localmax[i_row]-ap_center_try[i_ap, i_row+1])
            if np.min(this_d) <= sigma:
                ap_center_try[i_ap, i_row] = each_row_localmax[i_row][np.argmin(this_d)]
            else:
                break
        for i_row in range(i_row_center+1, n_row, 1):
            this_d = np.abs(each_row_localmax[i_row]-ap_center_try[i_ap, i_row-1])
            if np.min(this_d) <= sigma:
                ap_center_try[i_ap, i_row] = each_row_localmax[i_row][np.argmin(this_d)]
            else:
                break
    ind_good_ap = np.sum(np.isfinite(ap_center_try), axis=1) == n_row
    n_ap = np.sum(ind_good_ap)
    ap_center = ap_center_try[ind_good_ap]
    
    return ap_center


#%%
figure()
plot()

# %%
figure();
plot(flat_smooth[1000]); 
#plot(np.diff(flat_smooth[1000]))
plot(flat_smooth[1000][1:]-flat_smooth[1000][:-1])

flat = fits.getdata("/hydrogen/projects/others/yujingcheng/20191212/quartz.fits.gz")[:2048,:1024]
star = fits.getdata("/hydrogen/projects/others/yujingcheng/20191212/hr4468.fits")[:2048,:1024]
star = fits.getdata("/media/cham/Ubuntu 17.1/b0111.fits")[:2048,:1024]

ap_center = trace_naive_max(flat, sigma=5, disperse_axis="col", )
print(ap_center.shape)
#imshow(np.log10(flat))
figure(); imshow(flat); plot(ap_center.T,np.arange(2048),"w")


ap = Aperture(ap_center.shape[0], ap_center, ap_width=10)
ap.get_image_info(flat)
ap.polyfit(4)


#%%

_blz, _norm = extract.make_normflat(flat, ap, ap_width=12)
figure(); imshow(_norm)
rextr = extract.extract_all(flat, ap, ap_width=12)


#%%

starbg = ap.background(star, q=(5,5))
rextr = extract.extract_all(star-starbg, ap, ap_width=10)

figure(); plot(star[1200]); plot(starbg[1200])
#%%
figure();
plot(np.arange(2048),rextr["spec_sum"][10].T / _blz[10].T)
plot(np.arange(2048)+2048,rextr["spec_sum"][11].T/ _blz[11].T)

#figure();plot(_blz.T)
#%%
star0  = fits.getdata("/media/cham/Ubuntu 17.1/obj4blue_multi.fits")
#%%

flux0 = star0[1,5,:].T
flux1 = rextr["spec_extr"][-5].T
flux2 = rextr["spec_extr2"][-5].T
wave = np.arange(2048)


flux0n,_ = normalize_spectrum_iter(wave, flux0,niter=3)
flux1n,_ = normalize_spectrum_iter(wave, flux1,niter=3)
flux2n,_ = normalize_spectrum_iter(wave, flux2,niter=3)

figure();
plot(flux0n);
plot(flux1n+1);
plot(flux2n+1);

import joblib
joblib.dump(rextr, "/media/cham/Ubuntu 17.1/rextr_songcn.dump")
#import joblib
#rextr = joblib.load("~")
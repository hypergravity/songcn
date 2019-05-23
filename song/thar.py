# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:27:58 2017

@author: cham

@SONG: RMS =  0.00270124312246
delta_rv = 299792458/5500*0.00270124312246 = 147.23860278870518 m/s

LAMOST: R~1800,  299792.458/6000*3A = 150km/s WCALIB: 10km/s delta_rv = 5km/s
MMT: R~2500,  299792.458/R = 119.9km/s  rms=0.07A, delta_rv = 5km/s
SONG: 1800,  299792.458/6000*3A = 150km/s delta_rv = 5km/s
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import binned_statistic

from twodspec import calibration as calib


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def nanrms(x):
    return np.sqrt(np.nanmean(np.square(x)))


def calibrate(thar1d_simple, thar_solution_temp, thar_list,
              poly_order=(5, 10), slit=0):
    wave_temp, thar_temp, order_temp = thar_solution_temp

    # 1.initial solution
    print("@SONG: [ThAr] 2D cross-correlation ...")
    shift, corr2d = calib.thar1d_corr2d(
        thar1d_simple, thar_temp, y_shiftmax=3, x_shiftmax=20)
    print("@SONG: [ThAr] the shift is ", shift)
    wave_init = calib.interpolate_wavelength(
        wave_temp, shift, thar_temp, thar1d_simple)
    order_init = calib.interpolate_order(
        order_temp, shift, thar1d_simple) + 80

    print("@SONG: [ThAr] refine wavelength ...""")
    # 2.fit Gaussians to Thar lines
    lc_coord, lc_order, lc_thar, popt, pcov = calib.refine_thar_positions(
        wave_init, order_init, thar1d_simple, thar_list,
        fit_width=.3, lc_tol=.1, k=3, n_jobs=-1, verbose=10)

    # select using center deviation & line SNR
    ind_good0 = np.logical_and(
        np.abs(popt[:, 2] - lc_thar) < 5,  # (popt[:,3]*3),
        (popt[:, 1] / np.sqrt(2. * np.pi) / popt[:, 3] / np.abs(
            popt[:, 0])) > .1)
    print(np.sum(ind_good0))

    # 3.rejections of outliers
    ind_good1 = calib.clean_thar_polyfit1d_reject(
        lc_coord, lc_order, lc_thar, popt, ind_good0=ind_good0, deg=1, w=None,
        epsilon=0.002, n_reserve=8)
    print(np.sum(ind_good1))

    # 4. fit final solution
    print("@SONG: [ThAr] fit wavelength solution ")
    x_mini_lsq, ind_good_thar, scaler_coord, scaler_order, scaler_ml = calib.fit_grating_equation(
        lc_coord, lc_order, lc_thar, popt, pcov, ind_good_thar0=ind_good1,
        poly_order=poly_order, max_dev_threshold=.003, n_iter=1, lar=False,
        nl_eachorder=5)

    # construct grids for coordinates & order
    grid_coord, grid_order = np.meshgrid(np.arange(thar1d_simple.shape[1]),
                                         np.arange(
                                             thar1d_simple.shape[0]) + 80)

    # 4'.fit grating function
    sgrid_fitted_wave = calib.grating_equation_predict(
        grid_coord, grid_order, x_mini_lsq, poly_order,
        scaler_coord, scaler_order, scaler_ml)

    # 5.get the fitted wavelength
    lc_thar_fitted = calib.grating_equation_predict(
        lc_coord, lc_order, x_mini_lsq, poly_order,
        scaler_coord, scaler_order, scaler_ml)

    results = [sgrid_fitted_wave, lc_thar_fitted]

    # 6.figures for diagnostics
    bins = np.arange(4500, 7500, 500)
    bins_med, _, _ = binned_statistic(lc_thar[ind_good_thar],
                                      lc_thar_fitted[ind_good_thar] - lc_thar[
                                          ind_good_thar], statistic=np.median,
                                      bins=bins)
    bins_rms, _, _ = binned_statistic(lc_thar[ind_good_thar],
                                      lc_thar_fitted[ind_good_thar] - lc_thar[
                                          ind_good_thar], statistic=nanrms,
                                      bins=bins)

    """ [Figure]: calibration diagnostics """
    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(111)
    plt.plot(lc_thar, lc_thar_fitted - lc_thar, '.')
    plt.plot(lc_thar[ind_good_thar],
             lc_thar_fitted[ind_good_thar] - lc_thar[ind_good_thar], 'r.')
    plt.errorbar(bins[:-1] + np.diff(bins) * .5, bins_med, bins_rms, color='k',
                 ecolor='k')
    plt.xlim(4300, 7200)
    plt.ylim(-.008, .008)
    plt.xlabel("Wavelength (A)")
    plt.ylabel("$\lambda(solution)-\lambda(true)$")
    plt.title("RMS = {:05f} A for SLIT {:d} [{} lines]".format(
        rms(lc_thar_fitted[ind_good_thar] - lc_thar[ind_good_thar]), slit,
        len(ind_good_thar)))
    plt.legend(["deviation of all lines", "deviation of used lines",
                "mean RMS in bins"])
    fig.tight_layout()
    # fig.savefig(dir_work+"thar{}_{:s}".format(slit, thar_fn.replace(".fits", "_diagnostics.svg")))
    # plt.close(fig)
    results.append(fig)

    fig = plt.figure(figsize=(24, 8))
    plt.imshow(np.log10(thar1d_simple), aspect='auto', cmap=cm.viridis,
               vmin=np.nanpercentile(np.log10(thar1d_simple), 5),
               vmax=np.nanpercentile(np.log10(thar1d_simple), 95))
    plt.plot(lc_coord, lc_order - 80, ls='', marker='s', mfc='None', mec='b')
    plt.plot(lc_coord[ind_good_thar], lc_order[ind_good_thar] - 80, ls='',
             marker='s', mfc='None', mec='r')
    plt.xlabel("CCD Coordinate")
    plt.ylabel("Order")
    plt.colorbar()
    fig.tight_layout()
    # fig.savefig(dir_work+"thar{}_{:s}".format(slit, thar_fn.replace(".fits", "_used_lines.svg")))
    # plt.close(fig)
    results.append(fig)

    return results

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
from astropy import table
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import minimize

from twodspec.polynomial import Poly2DFitter


def corr_thar(wave_temp, thar_temp, thar_obs, maxshift=100):
    """
    Calculate the shift between thar_obs and thar_temp using correlation.
    Returns the shifted interpolated wavelength array

    Parameters
    ----------
    wave_temp : ndarray
        wavelength template.
    thar_temp : ndarray
        thar template.
    thar_obs : ndarray
        thar observation.
    maxshift : int, optional
        max shift. The default is 100.

    Returns
    -------
    wave_corr : ndarray
        shifted interpolated wavelength array.

    """
    # assert number of orders are the same
    assert thar_obs.shape == thar_temp.shape
    nrow, ncol = thar_temp.shape
    icol0 = np.int(0.25 * ncol)
    icol1 = np.int(0.75 * ncol)
    assert icol0 > maxshift

    nshift_grid = np.arange(-maxshift, maxshift + 1)
    nshift_dotmax = np.zeros(nrow)
    for irow in np.arange(nrow):
        ind_dotmax = np.argmax(
            [np.dot(thar_temp[irow][icol0 + ishift:icol1 + ishift], thar_obs[irow][icol0:icol1]) for ishift in
             nshift_grid])
        nshift_dotmax[irow] = nshift_grid[ind_dotmax]
    bulkshift = np.median(nshift_dotmax)
    npm1 = np.sum(np.abs(nshift_dotmax - bulkshift) <= 1)
    npm2 = np.sum(np.abs(nshift_dotmax - bulkshift) <= 2)
    assert npm2 > 0.5 * nrow
    print("@corr_thar: bulkshift={} (±1 {}/{}) (±2 {}/{})".format(bulkshift, npm1, nrow, npm1, nrow))

    xcoord = np.arange(ncol)
    wave_corr = interp1d(xcoord - bulkshift, wave_temp, kind="linear", fill_value="extrapolate")(xcoord)
    return wave_corr


def ccfmax_gauss(x, b=0, c=1):
    """
    Generate a Gaussian array, centering at b, with width of c.

    Parameters
    ----------
    x : ndarray
        x array.
    b : float, optional
        The center of Gaussian. The default is 0.
    c : float, optional
        The width of Gaussian. The default is 1.

    Returns
    -------
    y
        The Gaussian array.

    """
    y = np.exp(-0.5 * ((x - b) / c) ** 2.)
    return y / np.sum(y)


def ccfmax_cost(x0, x, y, width=2):
    """
    The cost function for ccfmax

    Parameters
    ----------
    x0 : float
        initial guess of max position.
    x : array
        x.
    y : array
        y.
    width : float, optional
        the width of Gaussian. The default is 2.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    y_model = ccfmax_gauss(x, x0, width)
    return - np.dot(y, y_model)


def ccfmax(x0, x, y, width=2, method="Nelder-Mead"):
    """
    The ccf max method

    Parameters
    ----------
    x0 : float
        initial guess of max position.
    x : array
        x.
    y : array
        y.
    width : float, optional
        the width of Gaussian. The default is 2.
    method : str, optional
        minimization method. The default is "Nelder-Mead".

    Returns
    -------
    pccf : TYPE
        DESCRIPTION.

    """
    pccf = minimize(ccfmax_cost, x0=x0, args=(x, y, width), method=method)
    return pccf


def gauss(x, a, b, c):
    """
    generate a Gaussian function

    Parameters
    ----------
    x : array like
        x coordinates.
    a : float
        amplitude (centeral height).
    b : float
        center.
    c : float
        sigma.

    Returns
    -------
    y
        y = Gaussian(x | a, b, c).

    """
    c = np.abs(c)
    return a / np.sqrt(2. * np.pi) / c * np.exp(-0.5 * ((x - b) / c) ** 2.)


def find_lines(wave_init, thar_obs, thar_line_list, npix_chunk=20):
    """
    Find emission lines in ThAr spectrum.

    Parameters
    ----------
    wave_init : array
        initial wavelength solution.
    thar_obs : array
        observed ThAr spectrum.
    thar_line_list : array
        the ThAr line list (a list of wavelengths).
    npix_chunk : int, optional
        the chunk length (half). The default is 20.

    Returns
    -------
    tlines : astropy.table.Table
        A table of identified lines.

    """
    # get a chunk [5*5] ±5sigma
    # shift: 1-2 pixel
    # typical oversampling: 1/3R --> 3pixel=FWHM
    # LAMOST MRS overestimate: 0.7 / 0.1 --> 7pixel=FWHM
    norder, npix = wave_init.shape
    xcoord = np.arange(npix)

    tlines = []
    # for each order
    for iorder in range(norder):
        # this order
        this_wave_init = wave_init[iorder]
        this_wave_min = np.min(this_wave_init)
        this_wave_max = np.max(this_wave_init)
        this_thar_obs = thar_obs[iorder]
        this_line_list = thar_line_list[np.logical_and(thar_line_list > this_wave_min, thar_line_list < this_wave_max)]

        # for each line
        for this_line in this_line_list:
            # init x position
            this_line_x_init = np.interp(this_line, this_wave_init, xcoord)
            this_line_x_init_int = np.int(this_line_x_init)  # np.argmin(np.abs((this_wave_init-this_line)))

            # get a chunk
            if npix_chunk < this_line_x_init_int < npix - npix_chunk:
                this_line_slc = slice(this_line_x_init_int - npix_chunk, this_line_x_init_int + npix_chunk)
                this_line_xcoord = xcoord[this_line_slc]
                this_line_thar = this_thar_obs[this_line_slc]
                this_line_base = np.percentile(this_line_thar, q=20)  # 25th percentile as baseline
                # if this_line_base < 0:
                #     continue

                # 1. Gaussian fit
                try:
                    popt, pcov = curve_fit(gauss, this_line_xcoord, this_line_thar - this_line_base,
                                           p0=[100, this_line_x_init, 1.5], )
                    # bounds=(np.array([0,-np.inf,1]), np.array([np.inf,np.inf,np.inf])))
                    this_line_a_gf = popt[0]
                    this_line_c_gf = popt[2]
                    this_line_x_gf = popt[1]
                    this_line_wave_init_gf = np.interp(popt[1], xcoord, this_wave_init)
                except:
                    this_line_a_gf = np.nan
                    this_line_c_gf = np.nan
                    this_line_x_gf = np.nan
                    this_line_wave_init_gf = np.nan
                # 2. CCF method
                try:
                    pccf = ccfmax(this_line_x_init, this_line_xcoord, this_line_thar, width=1.2, method="Nelder-Mead")
                    this_line_x_ccf = np.float(pccf.x)
                    this_line_wave_init_ccf = np.interp(this_line_x_ccf, xcoord, this_wave_init)
                except:
                    this_line_x_ccf = np.nan
                    this_line_wave_init_ccf = np.nan

                # gather results
                this_result = dict(
                    order=iorder,
                    line=this_line,
                    line_x_init=this_line_x_init,
                    # gf
                    line_x_gf=this_line_x_gf,
                    line_a_gf=this_line_a_gf,
                    line_c_gf=this_line_c_gf,
                    line_wave_init_gf=this_line_wave_init_gf,
                    # ccf
                    line_x_ccf=this_line_x_ccf,
                    line_wave_init_ccf=this_line_wave_init_ccf,
                    line_base=this_line_base
                )
                # np.array([iorder, this_line_xcenter, this_line, line_wave_gf, line_wave_ccf,
                #           this_line_base, popt[0]/np.sqrt(2.*np.pi)/popt[2],
                #           *popt, np.float(pccf.x)]))
                tlines.append(this_result)
    tlines = table.Table(tlines)
    print("@find_lines: {}/{} lines using GF / CCF!".format(
        np.sum(np.isfinite(tlines["line_x_gf"])),
        np.sum(np.isfinite(tlines["line_x_ccf"]))
    ))

    return tlines


def grating_equation(x, y, z, deg=(4, 10), nsigma=3, min_select=None):
    """
    Fit a grating equation (2D polynomial function) to data

    Parameters
    ----------
    x : array
        x coordinates of emission lines
    y : array
        order number.
    z : array
        The true wavelengths of lines.
    deg : tuple, optional
        The degree of the 2D polynomial. The default is (4, 10).
    nsigma : float, optional
        The data outside of the nsigma*sigma radius is rejected iteratively. The default is 3.
    min_select : int or None, optional
        The minimal number of selected lines. The default is None.

    Returns
    -------
    pf1, pf2, indselect

    """
    indselect = np.ones_like(x, dtype=bool)
    iiter = 0
    # pf1
    while True:
        pf1 = Poly2DFitter(x[indselect], y[indselect], z[indselect], deg=deg, pw=1, robust=False)
        z_pred = pf1.predict(x, y)
        z_res = z_pred - z
        sigma = np.std(z_res[indselect])
        indreject = np.abs(z_res[indselect]) > nsigma * sigma
        n_reject = np.sum(indreject)
        if n_reject == 0:
            # no lines to kick
            break
        elif isinstance(min_select, int) and min_select >= 0 and np.sum(indselect) <= min_select:
            # selected lines reach the threshold
            break
        else:
            # continue to reject lines
            indselect &= np.abs(z_res) < nsigma * sigma
            iiter += 1
        print("@grating_equation: iter-{} \t{} lines kicked, {} lines left, rms={:.5f} A".format(
            iiter, n_reject, np.sum(indselect), sigma))
    pf1.rms = sigma

    # pf2
    pf2 = Poly2DFitter(x[indselect], y[indselect], z[indselect], deg=deg, pw=2, robust=False)
    pf2.rms = np.std(pf2.predict(x[indselect], y[indselect]) - z[indselect])

    print("@grating_equation: {} iterations, rms = {:.5f}/{:.5f} A".format(iiter, pf1.rms, pf2.rms))
    return pf1, pf2, indselect


def test_corr_thar():
    import joblib
    wave_temp = joblib.load("/figs/wave_temp.dump")
    thar_temp = joblib.load("/figs/thar_temp.dump")
    thar_obs = joblib.load("/figs/thar_obs.dump")
    # load thar line list
    thar_line_list = joblib.load("/figs/thar_line_list.dump")
    # correlation for initial wavelength guess 
    wave_init = corr_thar(wave_temp, thar_temp, thar_obs, maxshift=100)
    # find thar lines
    tlines = find_lines(wave_init, thar_obs, thar_line_list, npix_chunk=20)

    # # initialize
    x = tlines["line_x_ccf"]
    y = tlines["order"]
    z = tlines["line"]

    # indfnt = np.isfinite(x)
    # x = x[indfnt]
    # y = y[indfnt]
    # z = z[indfnt]

    # fit grating equation
    pf1, pf2, indselect = grating_equation(x, y, z, deg=(4, 10), nsigma=3)
    tlines.add_column(table.Column(indselect, "indselect"))
    # predict wavelength solution
    nx, norder = thar_obs.shape
    mx, morder = np.meshgrid(np.arange(norder), np.arange(nx))
    wave_solu = pf2.predict(mx, morder)

    plt.figure()
    plt.plot(wave_temp.T, thar_temp.T, "b")
    plt.plot(wave_init.T, thar_obs.T, "r")
    plt.plot(wave_solu.T, thar_obs.T, "cyan")
    plt.vlines(thar_line_list, 0, 1e6)
    return

# %%
# deprecated!
# def calibrate(thar1d_simple, thar_solution_temp, thar_list,
#               poly_order=(5, 10), slit=5):
#     wave_temp, thar_temp, order_temp = thar_solution_temp
#     thar_temp += 400
#     thar_temp[thar_temp<0] = 0

#     wave_temp = np.flipud(wave_temp)
#     thar_temp = np.flipud(thar_temp)
#     # # 1.initial solution
#     # print("@SONG: [ThAr] 2D cross-correlation ...")
#     # shift, corr2d = calib.thar1d_corr2d(
#     #     thar1d_simple, thar_temp, y_shiftmax=3, x_shiftmax=40)
#     # figure();imshow(corr2d)

#     # print("@SONG: [ThAr] the shift is ", shift)
#     # wave_init = calib.interpolate_wavelength(
#     #     wave_temp, shift, thar_temp, thar1d_simple)
#     # order_init = calib.interpolate_order(
#     #     order_temp, shift, thar1d_simple) + 80

#     print("@SONG: [ThAr] refine wavelength ...""")
#     # 2.fit Gaussians to Thar lines
#     lc_coord, lc_order, lc_thar, popt, pcov = calib.refine_thar_positions(
#         wave_init, order_init, thar1d_simple, thar_list,
#         fit_width=.3, lc_tol=.1, k=3, n_jobs=-1, verbose=10)

#     # select using center deviation & line SNR
#     ind_good0 = np.logical_and(
#         np.abs(popt[:, 2] - lc_thar) < 5,  # (popt[:,3]*3),
#         (popt[:, 1] / np.sqrt(2. * np.pi) / popt[:, 3] / np.abs(
#             popt[:, 0])) > .1)
#     print(np.sum(ind_good0))

#     # 3.rejections of outliers
#     ind_good1 = calib.clean_thar_polyfit1d_reject(
#         lc_coord, lc_order, lc_thar, popt, ind_good0=ind_good0, deg=1, w=None,
#         epsilon=0.002, n_reserve=8)
#     print(np.sum(ind_good1))

#     # 4. fit final solution
#     print("@SONG: [ThAr] fit wavelength solution ")
#     x_mini_lsq, ind_good_thar, scaler_coord, scaler_order, scaler_ml = calib.fit_grating_equation(
#         lc_coord, lc_order, lc_thar, popt, pcov, ind_good_thar0=ind_good1,
#         poly_order=poly_order, max_dev_threshold=.003, n_iter=1, lar=False,
#         nl_eachorder=5)

#     # construct grids for coordinates & order
#     grid_coord, grid_order = np.meshgrid(np.arange(thar1d_simple.shape[1]),
#                                          np.arange(
#                                              thar1d_simple.shape[0]) + 80)

#     # 4'.fit grating function
#     sgrid_fitted_wave = calib.grating_equation_predict(
#         grid_coord, grid_order, x_mini_lsq, poly_order,
#         scaler_coord, scaler_order, scaler_ml)

#     # 5.get the fitted wavelength
#     lc_thar_fitted = calib.grating_equation_predict(
#         lc_coord, lc_order, x_mini_lsq, poly_order,
#         scaler_coord, scaler_order, scaler_ml)

#     results = [sgrid_fitted_wave, lc_thar_fitted]

#     # 6.figures for diagnostics
#     bins = np.arange(4500, 7500, 500)
#     bins_med, _, _ = binned_statistic(lc_thar[ind_good_thar],
#                                       lc_thar_fitted[ind_good_thar] - lc_thar[
#                                           ind_good_thar], statistic=np.median,
#                                       bins=bins)
#     bins_rms, _, _ = binned_statistic(lc_thar[ind_good_thar],
#                                       lc_thar_fitted[ind_good_thar] - lc_thar[
#                                           ind_good_thar], statistic=nanrms,
#                                       bins=bins)

#     """ [Figure]: calibration diagnostics """
#     fig = plt.figure(figsize=(12, 8))
#     fig.add_subplot(111)
#     plt.plot(lc_thar, lc_thar_fitted - lc_thar, '.')
#     plt.plot(lc_thar[ind_good_thar],
#              lc_thar_fitted[ind_good_thar] - lc_thar[ind_good_thar], 'r.')
#     plt.errorbar(bins[:-1] + np.diff(bins) * .5, bins_med, bins_rms, color='k',
#                  ecolor='k')
#     plt.xlim(4300, 7200)
#     plt.ylim(-.008, .008)
#     plt.xlabel("Wavelength (A)")
#     plt.ylabel("$\lambda(solution)-\lambda(true)$")
#     plt.title("RMS = {:05f} A for SLIT {:d} [{} lines]".format(
#         rms(lc_thar_fitted[ind_good_thar] - lc_thar[ind_good_thar]), slit,
#         len(ind_good_thar)))
#     plt.legend(["deviation of all lines", "deviation of used lines",
#                 "mean RMS in bins"])
#     fig.tight_layout()
#     # fig.savefig(dir_work+"thar{}_{:s}".format(slit, thar_fn.replace(".fits", "_diagnostics.svg")))
#     # plt.close(fig)
#     results.append(fig)

#     fig = plt.figure(figsize=(24, 8))
#     plt.imshow(np.log10(thar1d_simple), aspect='auto', cmap=cm.viridis,
#                vmin=np.nanpercentile(np.log10(thar1d_simple), 5),
#                vmax=np.nanpercentile(np.log10(thar1d_simple), 95))
#     plt.plot(lc_coord, lc_order - 80, ls='', marker='s', mfc='None', mec='b')
#     plt.plot(lc_coord[ind_good_thar], lc_order[ind_good_thar] - 80, ls='',
#              marker='s', mfc='None', mec='r')
#     plt.xlabel("CCD Coordinate")
#     plt.ylabel("Order")
#     plt.colorbar()
#     fig.tight_layout()
#     # fig.savefig(dir_work+"thar{}_{:s}".format(slit, thar_fn.replace(".fits", "_used_lines.svg")))
#     # plt.close(fig)
#     results.append(fig)

#     return results

# -*- coding: utf-8 -*-
"""

Author
------
Bo Zhang

Email
-----
bozhang@nao.cas.cn

Created on
----------
- Sun May 28 13:00:00 2017

Modifications
-------------
-

Aims
----
- routines to substract scattered light

"""


import numpy as np
from skimage.filters import gaussian
from scipy.signal import medfilt2d


def straight_line(x, y):
    """ return coefs for a straight line """
    a = np.float64(np.diff(y) / np.diff(x))
    b = np.float64(y[0] - a * x[0])
    return a, b


def apbackground(img, ap_center, q=(50, 5), npix_inter=7, sigma=(10, 10),
                 kernel_size=(11, 11)):
    """ determine background/scattered light using inter-aperture pixels
    
    Parameters
    ----------
    img: ndarray
        the image whose background is to be determined
    ap_center: ndarray
        aperture center array, (n_aperture, n_pixel)
    q: tuple of float
        the starting and ending percentile
    npix_inter:
        the number of pixel that will be used to determine the background
    sigma: tuple
        gaussian smoothing parameter
    kernel_size: tuple
        median smoothing parameter

    Returns
    -------
    bg0
    
    """
    n_ap = ap_center.shape[0]
    nrow, ncol = img.shape
    x = np.arange(ncol, dtype=float)
    npix_inter_hf = int(npix_inter / 2)
    if isinstance(q, tuple):
        q = np.linspace(q[0], q[1], n_ap)

    bg0 = np.zeros_like(img, float)
    for i_row in range(nrow):
        # each row
        for i_ap in range(n_ap):
            # each aperture
            if i_ap == 0:
                # the first aperture
                i_med_r = (ap_center[i_ap][i_row] + ap_center[i_ap + 1][
                    i_row]) / 2
                i_med = ap_center[i_ap][i_row]
                i_med_l = 2 * i_med - i_med_r

                i_med_r = int(i_med_r)
                # i_med = int(i_med)
                i_med_l = int(i_med_l)

                y_med_r = np.percentile(
                    img[i_row, np.max((0, i_med_r - npix_inter_hf)):np.max((1, i_med_r + npix_inter_hf + 1))], q[i_ap])
                y_med_l = np.percentile(
                    img[i_row, np.max((0, i_med_l - npix_inter_hf)):np.max((1, i_med_l + npix_inter_hf + 1))], q[i_ap])

                a, b = straight_line([x[i_med_l], x[i_med_r]], [y_med_l, y_med_r])
                bg0[i_row, :i_med_r] = a * x[:i_med_r] + b

            if i_ap == n_ap - 1:
                # the last aperture
                i_med = ap_center[i_ap][i_row]
                i_med_l = (ap_center[i_ap - 1][i_row] + ap_center[i_ap][
                    i_row]) / 2
                i_med_r = 2 * i_med - i_med_l

                # i_med = int(i_med)
                i_med_r = int(i_med_r)
                i_med_l = int(i_med_l)

                y_med_r = np.percentile(
                    img[i_row, i_med_r - npix_inter_hf:i_med_r + npix_inter_hf + 1], q[i_ap])
                y_med_l = np.percentile(
                    img[i_row, i_med_l - npix_inter_hf:i_med_l + npix_inter_hf + 1], q[i_ap])

                a, b = straight_line([x[i_med_l], x[i_med_r]], [y_med_l, y_med_r])
                bg0[i_row, i_med_l:] = a * x[i_med_l:] + b

            else:
                # the middle aperture
                # i_med = ap_center[i_ap][i_row]
                i_med_l = (ap_center[i_ap - 1][i_row] + ap_center[i_ap][i_row]) / 2
                i_med_r = (ap_center[i_ap][i_row] + ap_center[i_ap + 1][i_row]) / 2

                # i_med = int(i_med)
                i_med_r = int(i_med_r)
                i_med_l = int(i_med_l)

                y_med_r = np.percentile(
                    img[i_row, np.min((ncol-1, i_med_r - npix_inter_hf)):np.min((ncol, i_med_r + npix_inter_hf + 1))], q[i_ap])
                y_med_l = np.percentile(
                    img[i_row, np.min((ncol-1, i_med_l - npix_inter_hf)):np.min((ncol, i_med_l + npix_inter_hf + 1))], q[i_ap])

                a, b = straight_line([x[i_med_l], x[i_med_r]], [y_med_l, y_med_r])
                bg0[i_row, i_med_l:i_med_r] = a * x[i_med_l:i_med_r] + b

    # do a smooth
    # bgg = gaussian(bg0, sigma=sigma)
    #
    # bgm = medfilt2d(bg0, kernel_size=kernel_size)
    # bgmg = gaussian(bgm, sigma=sigma)

    if kernel_size is not None:
        bg0 = medfilt2d(bg0, kernel_size=kernel_size)
    if sigma is not None:
        bg0 = gaussian(bg0, sigma=sigma)
    return bg0


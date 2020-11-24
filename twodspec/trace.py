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
- Fri Apr 28 14:00:00 2017

Modifications
-------------
-

Aims
----
- ways to trace apertures

"""

import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy import signal
# from skimage import feature


# not used ??
# def trace_canny_row(flat, details=False, *args, **kwargs):
#     """ trace apertures using canny edge method in skimage for row apertures
#
#     Parameters
#     ----------
#     flat: numpy.ndarray
#         the image used to trace apertures, usually FLAT is used
#     details: bool
#         if True, return details
#     args, kwargs:
#         args & kwargs passed to skimage.feature.canny()
#
#     """
#
#     # 0. conservatively, transform to numpy.array
#     flatdata = np.array(flat)
#     n_row, n_col = flatdata.shape
#
#     # 1. find the edges using skimage.feature.canny()
#     # edge = feature.canny(flat.data, sigma=3.0, low_threshold=.05,
#     #                      use_quantiles=True)
#     edge = feature.canny(flatdata, *args, **kwargs)
#
#     # 2. pick out reasonable edges as aperture edges
#     # criterion: from column 1 to column -1
#     edge_good = []
#     for i_row in range(n_row):
#         i_col = 1
#         if edge[i_row, i_col]:
#             ap_list = []
#             ofst_row = 0
#             ap_list.append((i_row + ofst_row))
#             for i_col in range(2, n_col - 1):
#                 if edge[i_row + ofst_row, i_col]:
#                     ap_list.append((i_row + ofst_row))
#                 elif i_row + ofst_row + 1 < n_row - 1 and edge[
#                                     i_row + ofst_row + 1, i_col]:
#                     ofst_row += 1
#                     ap_list.append((i_row + ofst_row))
#                 elif i_row + ofst_row - 1 > 0 and edge[
#                                     i_row + ofst_row - 1, i_col]:
#                     ofst_row -= 1
#                     ap_list.append((i_row + ofst_row))
#                 else:
#                     # ap_list = []
#                     # ofst_row = 0
#                     break
#         if i_col >= flat.data.shape[1] - 2:
#             edge_good.append(ap_list)
#
#     ap_col = np.arange(1, n_col - 1, dtype=int)
#     ap_row = np.array(edge_good, dtype=int)
#
#     # 3. classify edges to {upper, lower} category
#     isupper = np.array(
#         [np.percentile(flat[ap_row_ - 3, ap_col], 50) < np.percentile(
#             flat[ap_row_ + 3, ap_col], 50) for ap_row_ in ap_row])
#
#     # 4. adopt good edges as true apertures
#     isadopted = np.zeros_like(isupper, dtype=bool)
#
#     # find an approporiate start for aperture
#     iap_start = 0
#     ap_upper = []
#     ap_lower = []
#     while True:
#         # if out of range, break
#         if iap_start >= len(isupper) - 1:
#             break
#
#         # grab apertures from the left
#         if isupper[iap_start] and not isupper[iap_start + 1]:
#             # and 10<np.median(ap_row[iap_start+1]-ap_row[iap_start])<30
#             # then, this is a good aperture
#             ap_upper.append(ap_row[iap_start])
#             ap_lower.append(ap_row[iap_start + 1])
#             isadopted[iap_start:iap_start + 2] = True
#             iap_start += 2
#         else:
#             # then, this is not a good aperture
#             print("@trace_canny: bad aperture for iap_start = ", iap_start)
#             iap_start += 1
#
#     # transform to numpy.array format
#     ap_upper = np.array(ap_upper)
#     ap_lower = np.array(ap_lower)
#     # fill left & right ends
#     ap_lower = np.hstack((ap_lower[:, 0].reshape(-1, 1), ap_lower,
#                           ap_lower[:, -1].reshape(-1, 1)))
#     ap_upper = np.hstack((ap_upper[:, 0].reshape(-1, 1), ap_upper,
#                           ap_upper[:, -1].reshape(-1, 1)))
#     # number of apertures
#     n_ap = ap_upper.shape[0]
#
#     # 5. assign results to Aperture object
#     results = dict(
#         # number of apertures found
#         n_ap=n_ap,
#
#         # edge int arrays, upper & lower seperated
#         ap_lower=ap_lower,
#         ap_upper=ap_upper,
#         ap_center=(ap_lower + ap_upper)/2
#     )
#
#     if not details:
#         return results
#     else:
#         details = dict(
#             # raw edges
#             edge=edge,
#
#             # edge int arrays, upper & lower together
#             ap_col=ap_col,
#             ap_row=ap_row,
#
#             # adopted edges & info
#             edge_good=edge_good,
#             isupper=isupper,
#             isadopted=isadopted,
#         )
#         return results, details
#
#
# def trace_canny_col(flat, details=False, verbose=True, *args, **kwargs):
#     """ trace apertures using canny method in skimage for column apertures
#
#     Parameters
#     ----------
#     flat: numpy.ndarray
#         the image used to trace apertures, usually FLAT is used
#     details: bool
#         if True, return details
#     args, kwargs:
#         args & kwargs passed to skimage.feature.canny()
#
#     """
#
#     # 0. conservatively, transform to numpy.array
#     flatdata = np.array(flat)
#     n_row, n_col = flatdata.shape
#
#     # 1. find the edges using skimage.feature.canny()
#     # edge = feature.canny(flat.data, sigma=3.0, low_threshold=.05,
#     #                      use_quantiles=True)
#     edge = feature.canny(flatdata, *args, **kwargs)
#
#     # 2. pick out reasonable edges as aperture edges
#     # criterion: from column 1 to column -1
#     n_row, n_col = np.rot90(flat.data).shape
#     edge_good = []
#     for i_col in range(n_row):
#         i_row = 1
#         if edge[i_row, i_col]:
#             ap_list = []
#             ofst_col = 0
#             ap_list.append(i_col + ofst_col)
#             for i_row in range(2, n_col - 1):
#                 # starts from the first row
#                 if edge[i_row, i_col + ofst_col]:
#                     # if the same column in the next row is True
#                     ap_list.append(i_col + ofst_col)
#                 elif i_col + ofst_col + 1 <= n_col - 1 \
#                         and edge[i_row, i_col + ofst_col + 1]:
#                     # offset = +1
#                     ofst_col += 1
#                     ap_list.append(i_col + ofst_col)
#                 elif i_col + ofst_col - 1 >= 0 \
#                         and edge[i_row, i_col + ofst_col - 1]:
#                     # offset = -1
#                     ofst_col -= 1
#                     ap_list.append(i_col + ofst_col)
#                 else:
#                     break
#         if i_row >= flat.data.shape[1] - 2:
#             edge_good.append(ap_list)
#
#     ap_row = np.arange(1, n_row - 1, dtype=int)
#     ap_col = np.array(edge_good, dtype=int)
#
#     # 3. classify edges to {upper, lower} category
#     isupper = np.array(
#         [np.percentile(flat[ap_row, ap_col_ - 3], 50) < np.percentile(
#             flat[ap_row, ap_col_ + 3], 50) for ap_col_ in ap_col])
#
#     # 4. adopt good edges as true apertures
#     isadopted = np.zeros_like(isupper, dtype=bool)
#
#     # find an approporiate start for aperture
#     iap_start = 0
#     ap_upper = []
#     ap_lower = []
#     while True:
#         # if out of range, break
#         if iap_start >= len(isupper) - 1:
#             break
#
#         # grab apertures from the left
#         if isupper[iap_start] and not isupper[iap_start + 1]:
#             # and 10<np.median(ap_row[iap_start+1]-ap_row[iap_start])<30
#             # then, this is a good aperture
#             ap_upper.append(ap_col[iap_start])
#             ap_lower.append(ap_col[iap_start + 1])
#             isadopted[iap_start:iap_start + 2] = True
#             iap_start += 2
#         else:
#             # then, this is not a good aperture
#             if verbose:
#                 print("@trace_canny: bad aperture for iap_start = ", iap_start)
#             iap_start += 1
#
#     # transform to numpy.array format
#     ap_upper = np.array(ap_upper)
#     ap_lower = np.array(ap_lower)
#     # fill left & right ends
#     ap_lower = np.hstack((ap_lower[:, 0].reshape(-1, 1), ap_lower,
#                           ap_lower[:, -1].reshape(-1, 1)))
#     ap_upper = np.hstack((ap_upper[:, 0].reshape(-1, 1), ap_upper,
#                           ap_upper[:, -1].reshape(-1, 1)))
#     # number of apertures
#     n_ap = ap_upper.shape[0]
#
#     # 5. assign results to Aperture object
#     results = dict(
#         # number of apertures found
#         n_ap=n_ap,
#
#         # edge int arrays, upper & lower seperated
#         ap_lower=ap_lower,
#         ap_upper=ap_upper,
#         ap_center=(ap_lower + ap_upper) / 2
#     )
#
#     if not details:
#         return results
#     else:
#         details = dict(
#             # raw edges
#             edge=edge,
#
#             # edge int arrays, upper & lower together
#             ap_col=ap_col,
#             ap_row=ap_row,
#
#             # adopted edges & info
#             edge_good=np.array(edge_good),
#             isupper=isupper,
#             isadopted=isadopted,
#         )
#         return results, details


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
    npix = np.int(n_sigma * sigma) + 1
    x = np.arange(-npix, npix + 1, 1)
    y = np.exp(-0.5 * (x / sigma) ** 2)
    y /= np.sum(y)
    return y


# not used
def box_kernel(fullwidth=10):
    npix = np.int(fullwidth)
    kernel = np.ones(npix)
    kernel /= np.sum(kernel)
    return kernel


def trace_naive_max(flat, sigma=7., maxdev=0, irow_start=None):
    """ trace aperture using naive method -- along local max

    Parameters
    ----------
    flat : ndarray
        FLAT.
    sigma : float, optional
        The sigma of Gaussian kernel. The default is 7.
    maxdev : float, optional
        The max deviation of local max while tracing. The default is 0.
    irow_start :
        The number of the starting row

    Returns
    -------
    ap_center : ndarray (n_ap, n_pix)
        The array for center of aperture.

    """
    flat = np.asarray(flat, float)
    n_row, n_col = flat.shape
    if irow_start is None:
        i_row_center = np.int(n_row / 2)
    else:
        i_row_center = irow_start

    # smooth flat along spacial axis
    flat_smooth = np.zeros_like(flat)
    gkernel = gaussian_kernel(sigma)
    for i_row in range(n_row):
        flat_smooth[i_row] = signal.fftconvolve(flat[i_row], gkernel, mode="same")

    # find local max
    flat_smooth_diff = np.diff(flat_smooth, axis=1)
    flat_localmax = np.zeros_like(flat, bool)
    flat_localmax[:, 1:-1] = (flat_smooth_diff[:, 1:] < 0) & (flat_smooth_diff[:, :-1] > 0)

    # start from center row
    ind_col_localmax, = np.where(flat_localmax[i_row_center])
    n_ap_try = len(ind_col_localmax)
    ap_center_try = np.ones((n_ap_try, n_row)) * np.nan
    ap_center_try[:, i_row_center] = ind_col_localmax

    # find local max col each row
    each_row_localmax = [np.where(flat_localmax[i_row])[0] for i_row in range(n_row)]

    # loop for trials
    if maxdev <= 0 or maxdev is None:
        maxdev = sigma
    for i_ap in range(n_ap_try):
        for i_row in range(i_row_center - 1, 0 - 1, -1):
            this_d = np.abs(each_row_localmax[i_row] - ap_center_try[i_ap, i_row + 1])
            if np.min(this_d) <= maxdev:
                ap_center_try[i_ap, i_row] = each_row_localmax[i_row][np.argmin(this_d)]
            else:
                break
        for i_row in range(i_row_center + 1, n_row, 1):
            this_d = np.abs(each_row_localmax[i_row] - ap_center_try[i_ap, i_row - 1])
            if np.min(this_d) <= maxdev:
                ap_center_try[i_ap, i_row] = each_row_localmax[i_row][np.argmin(this_d)]
            else:
                break
    ind_good_ap = np.sum(np.isfinite(ap_center_try), axis=1) == n_row
    n_ap = np.sum(ind_good_ap)
    ap_center = ap_center_try[ind_good_ap]

    return ap_center


def trace_naive_min(flat, sigma=7., maxdev=0):
    """ trace aperture using naive method -- along local min

    Parameters
    ----------
    flat : ndarray
        FLAT.
    sigma : float, optional
        The sigma of Gaussian kernel. The default is 7.
    maxdev : float, optional
        The max deviation of local max while tracing. The default is 0.

    Returns
    -------
    ap_center : ndarray (n_ap, n_pix)
        The array for center of aperture..

    """
    flat = np.asarray(flat, float)
    n_row, n_col = flat.shape
    i_row_center = np.int(n_row / 2)

    # smooth flat
    flat_smooth = np.zeros_like(flat)
    gkernel = gaussian_kernel(sigma)
    for i_row in range(n_row):
        flat_smooth[i_row] = signal.fftconvolve(flat[i_row], gkernel, mode="same")

    # find local max
    flat_smooth_diff = np.diff(flat_smooth, axis=1)
    flat_localmax = np.zeros_like(flat, bool)
    flat_localmax[:, 1:-1] = (flat_smooth_diff[:, 1:] > 0) & (flat_smooth_diff[:, :-1] < 0)

    # start from center row
    ind_col_localmax, = np.where(flat_localmax[i_row_center])
    n_ap_try = len(ind_col_localmax)
    ap_center_try = np.ones((n_ap_try, n_row)) * np.nan
    ap_center_try[:, i_row_center] = ind_col_localmax

    # each row local max col
    each_row_localmax = [np.where(flat_localmax[i_row])[0] for i_row in range(n_row)]

    # loop for trials
    if maxdev <= 0 or maxdev is None:
        maxdev = sigma
    for i_ap in range(n_ap_try):
        for i_row in range(i_row_center - 1, 0 - 1, -1):
            this_d = np.abs(each_row_localmax[i_row] - ap_center_try[i_ap, i_row + 1])
            if np.min(this_d) <= maxdev:
                ap_center_try[i_ap, i_row] = each_row_localmax[i_row][np.argmin(this_d)]
            else:
                break
        for i_row in range(i_row_center + 1, n_row, 1):
            this_d = np.abs(each_row_localmax[i_row] - ap_center_try[i_ap, i_row - 1])
            if np.min(this_d) <= maxdev:
                ap_center_try[i_ap, i_row] = each_row_localmax[i_row][np.argmin(this_d)]
            else:
                break
    ind_good_ap = np.sum(np.isfinite(ap_center_try), axis=1) == n_row
    # n_ap = np.sum(ind_good_ap)
    ap_center = ap_center_try[ind_good_ap]

    return ap_center


def test_trace_naive_max():
    plt.rcParams.update({"font.size": 15})
    flat = fits.getdata("/Users/cham/projects/song/star_spec/20191105/night/ext/masterflat_20191105_slit5.fits")
    ap_center = trace_naive_max(flat, sigma=7, maxdev=10)
    fig = plt.figure(figsize=(8,6))
    plt.plot(ap_center.T, np.arange(2048), "-", c="w",label="apertures",lw=1)
    plt.imshow(np.log10(flat),cmap=plt.cm.jet)
    # plt.set_xticks(np.linspace(0,2047,))
    plt.colorbar()
    plt.xlabel("CCD X coordinate")
    plt.ylabel("CCD Y coordinate")
    fig.tight_layout()
    # fig.savefig(os.getenv("HOME")+"/PycharmProjects/songcn/figs/test_trace_naive_max.pdf")
    print(ap_center.shape)
    return


if __name__ == "__main__":
    test_trace_naive_max()
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

import numpy as np
from skimage import feature


def trace_canny(flat, details=False, *args, **kwargs):
    """

    Parameters
    ----------
    flat: numpy.ndarray
        the image used to trace apertures, usually FLAT is used
    details: bool
        if True, return details
    args, kwargs:
        args & kwargs passed to skimage.feature.canny()

    """

    # 0. conservatively, transform to numpy.array
    flatdata = np.array(flat)
    n_row, n_col = flatdata.shape

    # 1. find the edges using skimage.feature.canny()
    # edge = feature.canny(flat.data, sigma=3.0, low_threshold=.05,
    #                      use_quantiles=True)
    edge = feature.canny(flatdata, *args, **kwargs)

    # 2. pick out reasonable edges as aperture edges
    # criterion: from column 1 to column -1
    edge_good = []
    for i_row in range(n_row):
        i_col = 1
        if edge[i_row, i_col]:
            ap_list = []
            ofst_row = 0
            ap_list.append((i_row + ofst_row))
            for i_col in range(2, n_col - 1):
                if edge[i_row + ofst_row, i_col]:
                    ap_list.append((i_row + ofst_row))
                elif i_row + ofst_row + 1 < n_row - 1 and edge[
                                    i_row + ofst_row + 1, i_col]:
                    ofst_row += 1
                    ap_list.append((i_row + ofst_row))
                elif i_row + ofst_row - 1 > 0 and edge[
                                    i_row + ofst_row - 1, i_col]:
                    ofst_row -= 1
                    ap_list.append((i_row + ofst_row))
                else:
                    # ap_list = []
                    # ofst_row = 0
                    break
        if i_col >= flat.data.shape[1] - 2:
            edge_good.append(ap_list)

    ap_col = np.arange(1, n_col - 1, dtype=int)
    ap_row = np.array(edge_good, dtype=int)

    # 3. classify edges to {upper, lower} category
    isupper = np.array(
        [np.percentile(flat[ap_row_ - 3, ap_col], 50) < np.percentile(
            flat[ap_row_ + 3, ap_col], 50) for ap_row_ in ap_row])

    # 4. adopt good edges as true apertures
    isadopted = np.zeros_like(isupper, dtype=bool)

    # find an approporiate start for aperture
    iap_start = 0
    ap_upper = []
    ap_lower = []
    while True:
        # if out of range, break
        if iap_start >= len(isupper) - 1:
            break

        # grab apertures from the left
        if isupper[iap_start] and not isupper[iap_start + 1]:
            # and 10<np.median(ap_row[iap_start+1]-ap_row[iap_start])<30
            # then, this is a good aperture
            ap_upper.append(ap_row[iap_start])
            ap_lower.append(ap_row[iap_start + 1])
            isadopted[iap_start:iap_start + 2] = True
            iap_start += 2
        else:
            # then, this is not a good aperture
            print("@trace_canny: bad aperture for iap_start = ", iap_start)
            iap_start += 1

    # transform to numpy.array format
    ap_upper = np.array(ap_upper)
    ap_lower = np.array(ap_lower)
    # fill left & right ends
    ap_lower = np.hstack((ap_lower[:, 0].reshape(-1, 1), ap_lower,
                          ap_lower[:, -1].reshape(-1, 1)))
    ap_upper = np.hstack((ap_upper[:, 0].reshape(-1, 1), ap_upper,
                          ap_upper[:, -1].reshape(-1, 1)))
    # number of apertures
    n_ap = ap_upper.shape[0]

    # 5. assign results to Aperture object
    results = dict(
        # number of apertures found
        n_ap=n_ap,

        # edge int arrays, upper & lower seperated
        ap_lower=ap_lower,
        ap_upper=ap_upper,
        ap_center=(ap_lower + ap_upper)/2
    )

    if not details:
        return results
    else:
        details = dict(
            # raw edges
            edge=edge,

            # edge int arrays, upper & lower together
            ap_col=ap_col,
            ap_row=ap_row,

            # adopted edges & info
            edge_good=edge_good,
            isupper=isupper,
            isadopted=isadopted,
        )
        return results, details


def trace_canny_col(flat, details=False, verbose=True, *args, **kwargs):
    """

    Parameters
    ----------
    flat: numpy.ndarray
        the image used to trace apertures, usually FLAT is used
    details: bool
        if True, return details
    args, kwargs:
        args & kwargs passed to skimage.feature.canny()

    """

    # 0. conservatively, transform to numpy.array
    flatdata = np.array(flat)
    n_row, n_col = flatdata.shape

    # 1. find the edges using skimage.feature.canny()
    # edge = feature.canny(flat.data, sigma=3.0, low_threshold=.05,
    #                      use_quantiles=True)
    edge = feature.canny(flatdata, *args, **kwargs)

    # 2. pick out reasonable edges as aperture edges
    # criterion: from column 1 to column -1
    n_row, n_col = np.rot90(flat.data).shape
    edge_good = []
    for i_col in range(n_row):
        i_row = 1
        if edge[i_row, i_col]:
            ap_list = []
            ofst_col = 0
            ap_list.append(i_col + ofst_col)
            for i_row in range(2, n_col - 1):
                # starts from the first row
                if edge[i_row, i_col + ofst_col]:
                    # if the same column in the next row is True
                    ap_list.append(i_col + ofst_col)
                elif i_col + ofst_col + 1 <= n_col - 1 \
                        and edge[i_row, i_col + ofst_col + 1]:
                    # offset = +1
                    ofst_col += 1
                    ap_list.append(i_col + ofst_col)
                elif i_col + ofst_col - 1 >= 0 \
                        and edge[i_row, i_col + ofst_col - 1]:
                    # offset = -1
                    ofst_col -= 1
                    ap_list.append(i_col + ofst_col)
                else:
                    break
        if i_row >= flat.data.shape[1] - 2:
            edge_good.append(ap_list)

    ap_row = np.arange(1, n_row - 1, dtype=int)
    ap_col = np.array(edge_good, dtype=int)

    # 3. classify edges to {upper, lower} category
    isupper = np.array(
        [np.percentile(flat[ap_row, ap_col_ - 3], 50) < np.percentile(
            flat[ap_row, ap_col_ + 3], 50) for ap_col_ in ap_col])

    # 4. adopt good edges as true apertures
    isadopted = np.zeros_like(isupper, dtype=bool)

    # find an approporiate start for aperture
    iap_start = 0
    ap_upper = []
    ap_lower = []
    while True:
        # if out of range, break
        if iap_start >= len(isupper) - 1:
            break

        # grab apertures from the left
        if isupper[iap_start] and not isupper[iap_start + 1]:
            # and 10<np.median(ap_row[iap_start+1]-ap_row[iap_start])<30
            # then, this is a good aperture
            ap_upper.append(ap_col[iap_start])
            ap_lower.append(ap_col[iap_start + 1])
            isadopted[iap_start:iap_start + 2] = True
            iap_start += 2
        else:
            # then, this is not a good aperture
            if verbose:
                print("@trace_canny: bad aperture for iap_start = ", iap_start)
            iap_start += 1

    # transform to numpy.array format
    ap_upper = np.array(ap_upper)
    ap_lower = np.array(ap_lower)
    # fill left & right ends
    ap_lower = np.hstack((ap_lower[:, 0].reshape(-1, 1), ap_lower,
                          ap_lower[:, -1].reshape(-1, 1)))
    ap_upper = np.hstack((ap_upper[:, 0].reshape(-1, 1), ap_upper,
                          ap_upper[:, -1].reshape(-1, 1)))
    # number of apertures
    n_ap = ap_upper.shape[0]

    # 5. assign results to Aperture object
    results = dict(
        # number of apertures found
        n_ap=n_ap,

        # edge int arrays, upper & lower seperated
        ap_lower=ap_lower,
        ap_upper=ap_upper,
        ap_center=(ap_lower + ap_upper) / 2
    )

    if not details:
        return results
    else:
        details = dict(
            # raw edges
            edge=edge,

            # edge int arrays, upper & lower together
            ap_col=ap_col,
            ap_row=ap_row,

            # adopted edges & info
            edge_good=np.array(edge_good),
            isupper=isupper,
            isadopted=isadopted,
        )
        return results, details

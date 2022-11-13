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

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, morphology


def straight_line_2pts(x, y):
    """ return coefs for a straight line: y = a*x+b, x and y are 2-element array """
    a = (y[1] - y[0]) / (x[1] - x[0])
    b = y[0] - a * x[0]
    return a, b


def background(img, al, q=(40, 5), median_sigma=15, gaussian_sigma=15):
    """
    Estimate background / scattered light using inter-aperture pixels.

    Parameters
    ----------
    img : ndarray
        the image whose background is to be determined
    al : ApertureList
        aperture center array, (n_aperture, n_pixel)
    q : tuple of float
        the starting and ending percentile
    median_sigma : int
        median smoothing parameter
    gaussian_sigma : int or tuple
        gaussian smoothing parameter

    Returns
    -------
    numpy.ndarray
        The estimated background / scattered light.
    """
    n_row, n_col = img.shape
    mask = al.get_mask(ap_width=10)  # interpolate for True values
    bg = np.where(mask, 0., img)
    qs = np.linspace(*q, num=len(al))

    # img_coord = np.arange(n_row)

    for i_ap in range(len(al)):
        print(f"processing {i_ap}")
        if i_ap == 0:
            ap_center_left = (2 * al[i_ap].ap_center_floor - al[i_ap + 1].ap_center_floor).astype(int)
            ap_center_center = al[i_ap].ap_center_floor.astype(int)
            ap_center_right = al[i_ap + 1].ap_center_floor.astype(int)
        elif i_ap == len(al) - 1:
            ap_center_left = al[i_ap - 1].ap_center_floor.astype(int)
            ap_center_center = al[i_ap].ap_center_floor.astype(int)
            ap_center_right = (2 * al[i_ap].ap_center_floor - al[i_ap - 1].ap_center_floor).astype(int)
        else:
            ap_center_left = al[i_ap - 1].ap_center_floor.astype(int)
            ap_center_center = al[i_ap].ap_center_floor.astype(int)
            ap_center_right = al[i_ap + 1].ap_center_floor.astype(int)

        pos_left = .5 * (ap_center_left + ap_center_center)
        pos_right = .5 * (ap_center_center + ap_center_right)

        for i_row in range(n_row):
            line_slope, line_intersect = straight_line_2pts(
                [
                    pos_left[i_row],
                    pos_right[i_row]
                ],
                [
                    np.percentile(img[i_row, max(0, ap_center_left[i_row]):max(1, ap_center_center[i_row])], qs[i_ap]),
                    np.percentile(img[i_row, max(0, ap_center_center[i_row]):max(1, ap_center_right[i_row])], qs[i_ap]),
                ]
            )
            interp_start = 0 if i_ap == 0 else max(0, ap_center_left[i_row])
            interp_stop = n_col if i_ap == len(al) - 1 else min(n_col, ap_center_right[i_row])
            if i_ap == 0:
                mask[i_row, :interp_stop] = True
            if i_ap == len(al) - 1:
                mask[i_row, interp_start:] = True
            bg[i_row, interp_start:interp_stop] = np.where(
                mask[i_row, interp_start:interp_stop],
                line_slope * np.arange(interp_start, interp_stop) + line_intersect,
                bg[i_row, interp_start:interp_stop],
            )

    print("Doing median filtering ...")
    bg1 = filters.median(bg, footprint=morphology.disk(median_sigma), mode="reflect")
    print("Doing gaussian filtering ...")
    bg2 = filters.gaussian(bg1, sigma=gaussian_sigma, mode="reflect")
    return bg2


# if __name__ == "__main__":
#     fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
#     axs[0].imshow(np.log10(img))
#     axs[1].imshow(np.log10(bg1))
#     axs[0].plot(al.ap_center.T, np.arange(2048), "w-")
#     axs[1].plot(al.ap_center.T, np.arange(2048), "w-")
#
#     figure(figsize=(12, 4))
#     # plot(img[1000])
#     # plot(bg2[1000])
#     plot(img[0])
#     plot(bg[0])
#     plot(bg1[0])
#     plot(bg2[0])
#
#     figure()
#     plot(img[i_row])
#     plot(img[i_row] * mask[i_row])

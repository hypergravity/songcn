import logging
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import argrelextrema

LocalMax = namedtuple(
    typename="LocalMax",
    field_names=[
        "num",  # number of maxima
        "max_ind",  # index of maxima
        "max_val",  # value of maxima
        "max_val_interp",  # value interpolated from neighboring minima
        "max_snr",  # SNR of maxima (max/max_val_interp)
        "min_ind",  # index of minima
        "side_ind",  # index of two sides
        "side_mask",  # True if out of bounds
    ]
)


def generate_pulse_kernel(kernel_width=15):
    """ Generate a pulse kernel. """
    kernel = np.zeros((kernel_width + 2), dtype=float)
    kernel[1:-1] = 1
    return kernel


def find_local_max(arr, kernel_width=15):
    """ Find local max in 1D array.

    Parameters
    ----------
    arr : np.array
        The 1D array.
    kernel_width : int
        The kernel width in number of pixels.

    """
    arr_convolved = np.convolve(arr, generate_pulse_kernel(kernel_width=kernel_width), mode="same")
    # find local max using scipy.signal.argrelextrema
    ind_local_max = argrelextrema(arr_convolved, np.greater_equal, order=kernel_width, mode='clip')[0]
    logging.info(f"{len(ind_local_max)} maxima found")
    # interpolate for local min
    ind_local_max_delta = np.diff(ind_local_max) / 2
    ind_local_min_derived = np.hstack(
        (
            ind_local_max[:-1] - ind_local_max_delta,
            ind_local_max[-1] - ind_local_max_delta[-1],
            ind_local_max[-1] + ind_local_max_delta[-1],
        )
    ).astype(int)
    # calculate SNR for local max
    ind_two_sides = np.array([
        ind_local_min_derived[:-1],
        ind_local_min_derived[1:]
    ])
    ind_two_sides_mask = np.logical_or(ind_two_sides < 0, ind_two_sides > len(arr) - 1)
    ind_two_sides_valid = np.where(ind_two_sides_mask, 0, ind_two_sides)  # do not go out of bounds
    # estimate SNR of local max, clip out-of-bounds values
    interp_val_local_max = np.ma.MaskedArray(
        data=arr_convolved[ind_two_sides_valid],
        mask=ind_two_sides_mask,
    ).mean(axis=0)
    assert interp_val_local_max.mask.sum() == 0
    interp_val_local_max = interp_val_local_max.data
    val_local_max = arr_convolved[ind_local_max]
    snr_local_max = val_local_max / interp_val_local_max
    return LocalMax(
        num=len(ind_local_max),
        max_ind=ind_local_max,
        max_val=val_local_max,
        max_val_interp=interp_val_local_max,
        max_snr=snr_local_max,
        min_ind=ind_local_min_derived,
        side_ind=ind_two_sides,
        side_mask=ind_two_sides_mask,
    )


def find_local_max_2d(img, pos_init=(1, 1), extra_bin=1, kernel_width=10):
    """ slice by slice """
    is_local_max = np.zeros_like(img, dtype=bool)
    for i_row in range(img.shape[0]):
        starting_row = i_row - extra_bin
        stop_row = i_row + 1 + extra_bin
        this_slice = img[starting_row:stop_row].sum(axis=0)
        localmax = find_local_max(this_slice, kernel_width=kernel_width)
        is_local_max[i_row, localmax.max_ind] = True
        print(f"irow = {i_row}, {localmax.num} max found")
    return is_local_max


def trace_one_aperture(img, init_pos=(1000, 993), extra_bin=1, kernel_width=10, maxdev=5):
    """
    Given an initial position, search for the aperture.

    The image will be convolved with a pulse kernel slice by slice.
    Then the search starts from initial position row by row.

    Parameters
    ----------
    img : np.ndarray
        The 2D image.
    init_pos : tuple
        A tuple of (row, col) of the initial position.
    extra_bin : int
        The number of extra binning rows to enhance SNR.
    kernel_width : int
        The width of the pulse kernel.
    maxdev : int
        The max deviation of ap_col allowed in consecutive two rows.

    Returns
    -------
    tuple
        (ap_col, ap_mask) where ap_col is the aperture center index and ap_mask the validity.
        If the ap_col is bad, ap_mask is set True.
    """
    n_row, n_col = img.shape
    init_row, init_col = init_pos
    if kernel_width is not None:
        # convolve image slice by slice (row by row)
        img = np.array(
            [
                np.convolve(
                    img[i_row - extra_bin - extra_bin:i_row + 1 + extra_bin].sum(axis=0),
                    generate_pulse_kernel(kernel_width=kernel_width),
                    mode="same"
                ) for i_row in np.arange(n_row)
            ]
        )

    ap_col = np.zeros(n_row, dtype=int)
    ap_mask = np.zeros(n_row, dtype=bool)
    ap_col[init_row] = init_col
    # search in current row
    i_row = init_row  # this is special
    chunk_data = img[i_row, ap_col[i_row] - 2 * kernel_width:ap_col[i_row] + 1 + 2 * kernel_width]
    argmax = argrelextrema(chunk_data, np.greater_equal, order=kernel_width, mode='clip')[0]
    ind_mindev = np.argmin(np.abs(argmax - 2 * kernel_width))
    if np.min(np.abs(argmax[ind_mindev] - 2 * kernel_width)) <= maxdev:
        ap_col[i_row] = ap_col[i_row] - 2 * kernel_width + argmax[ind_mindev]
    else:
        logging.warning(f"Warning: deviation({np.min(np.abs(argmax[ind_mindev] - 2 * kernel_width)):d})"
                        f" larger than maxdev({maxdev}) in Row {i_row}")
        ap_col[i_row] = ap_col[i_row]
        ap_mask[i_row] = True

    # search the above
    for i_row in np.arange(init_row)[::-1]:
        # try:
        chunk_start = max(0, ap_col[i_row + 1] - 2 * kernel_width)
        chunk_stop = min(n_col - 1, ap_col[i_row + 1] + 1 + 2 * kernel_width)
        chunk_data = img[i_row, chunk_start:chunk_stop]
        argmax = argrelextrema(chunk_data, np.greater_equal, order=kernel_width, mode='clip')[0]
        ind_mindev = np.argmin(np.abs(argmax + chunk_start - ap_col[i_row + 1]))
        if np.min(np.abs(argmax[ind_mindev] - 2 * kernel_width)) <= maxdev:
            ap_col[i_row] = ap_col[i_row + 1] - 2 * kernel_width + argmax[ind_mindev]
        else:
            logging.warning(f"Warning: deviation({np.min(np.abs(argmax[ind_mindev] - 2 * kernel_width)):d})"
                            f" larger than maxdev({maxdev}) in Row {i_row}")
            ap_col[i_row] = ap_col[i_row + 1]
            ap_mask[i_row] = True
        # except:
        #     ap_col[i_row] = ap_col[i_row + 1]
        #     ap_mask[i_row] = True

    # search the below
    for i_row in np.arange(init_row + 1, n_row):
        # try:
        chunk_start = max(0, ap_col[i_row - 1] - 2 * kernel_width)
        chunk_stop = min(n_col - 1, ap_col[i_row - 1] + 1 + 2 * kernel_width)
        chunk_data = img[i_row, chunk_start:chunk_stop]
        argmax = argrelextrema(chunk_data, np.greater_equal, order=kernel_width, mode='clip')[0]
        ind_mindev = np.argmin(np.abs(argmax + chunk_start - ap_col[i_row - 1]))
        if np.min(np.abs(argmax[ind_mindev] - 2 * kernel_width)) <= maxdev:
            ap_col[i_row] = ap_col[i_row - 1] - 2 * kernel_width + argmax[ind_mindev]
        else:
            logging.warning(f"Warning: deviation({np.min(np.abs(argmax[ind_mindev] - 2 * kernel_width)):d})"
                            f" larger than maxdev({maxdev}) in Row {i_row}")
            ap_col[i_row] = ap_col[i_row - 1]
            ap_mask[i_row] = True
        # except:
        #     ap_col[i_row] = ap_col[i_row - 1]
        #     ap_mask[i_row] = True

    return ap_col, ap_mask


if __name__ == "__main__":
    fp = "/Users/cham/projects/song/star_spec/20191105/night/ext/masterflat_20191105_slit5.fits"
    img = fits.getdata(fp)
    central_ind = int(img.shape[0] / 2)
    central_slice = img[central_ind:central_ind + 10].sum(axis=0)
    for kernel_width in np.arange(5, 30):
        localmax = find_local_max(arr=central_slice, kernel_width=kernel_width)
        print(
            f"kernel_width: {kernel_width}, number of max: {localmax.num}, "
            f"median SNR: {np.median(localmax.max_snr):.2f}, mean SNR: {np.mean(localmax.max_snr):.2f}"
        )

    localmax = find_local_max(arr=central_slice, kernel_width=15)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(np.log10(img), extent=(-.5, 2047.5, -.5, 2047.5,), origin="lower")
    for i_ap in np.arange(localmax.num):
        ap_col, ap_mask = trace_one_aperture(
            img, init_pos=(central_ind, localmax.max_ind[i_ap]), extra_bin=1, kernel_width=15, maxdev=3)
        ax.plot(ap_col, np.arange(img.shape[0]))
        print(f"i_ap = {i_ap}, masksum = {sum(ap_mask)}")
    plt.show()

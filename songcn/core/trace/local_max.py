import logging
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

# LocalMax
LocalMax = namedtuple(
    typename="LocalMax",
    field_names=[
        "num",  # number of maxima
        "max_ind",  # index of maxima -> most important, the start point of `trace_one_aperture`
        "max_val",  # value of maxima
        "max_val_interp",  # value interpolated from neighboring minima
        "max_snr",  # SNR of maxima (max/max_val_interp)
        "min_ind",  # index of minima
        "side_ind",  # index of two sides
        "side_mask",  # True if out of bounds
    ],
)


def find_local_max(x, kernel_width=15) -> LocalMax:
    """Convolve 1d array `x` with a pulse kernel with width `kernel_width`, and find local maxima.

    Parameters
    ----------
    x : np.ndarray
        The 1D array.
    kernel_width : int
        The kernel width in number of pixels.
    """
    # convolve with a gaussian for each row
    arr_convolved = np.convolve(
        x, generate_pulse_kernel(kernel_width=kernel_width), mode="same"
    )
    # find local max using scipy.signal.argrelextrema
    ind_local_max = argrelextrema(
        arr_convolved, np.greater_equal, order=kernel_width, mode="clip"
    )[0]
    logging.info(f"{len(ind_local_max)} maximums found")
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
    ind_two_sides = np.array([ind_local_min_derived[:-1], ind_local_min_derived[1:]])
    ind_two_sides_mask = np.logical_or(ind_two_sides < 0, ind_two_sides > len(x) - 1)
    ind_two_sides_valid = np.where(
        ind_two_sides_mask, 0, ind_two_sides
    )  # do not go out of bounds
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


def generate_pulse_kernel(kernel_width: int = 15) -> np.ndarray:
    """Generate a pulse kernel."""
    kernel = np.zeros((kernel_width + 2), dtype=float)
    kernel[1:-1] = 1
    return kernel


def trace_one_aperture(
    image: np.ndarray,
    init_pos: tuple = (1000, 993),
    extra_bin: int = 1,
    kernel_width: int = 10,
    max_dev: int = 5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an initial position, search for the aperture.

    Assuming the dispersion direction is vertical and spatial direction is horizontal.
    The image will be convolved with a pulse kernel slice by slice.
    Then the search starts from initial position in two directions (above and below) row by row.
    At the end of the story, `ap_col` and `ap_mask` is return, both of them are (n_row,) `np.ndarray`.

    Parameters
    ----------
    image : np.ndarray
        The 2D image.
    init_pos : tuple
        A tuple of (row, col) of the initial position.
    extra_bin : int
        The number of extra binning rows to enhance SNR.
    kernel_width : int
        The width of the pulse kernel.
    max_dev : int
        The max deviation of ap_col allowed in consecutive two rows.
    verbose : bool
        If True, print verbose info.

    Returns
    -------
    tuple
        (ap_col, ap_mask) where ap_col is the aperture center index and ap_mask the validity.
        If the ap_col is bad, ap_mask is set True.
    """
    n_row, n_col = image.shape
    init_row, init_col = init_pos
    if kernel_width is not None:
        # convolve image slice by slice (row by row)
        image = np.array(
            [
                np.convolve(
                    image[i_row - extra_bin - extra_bin : i_row + 1 + extra_bin].sum(
                        axis=0
                    ),
                    generate_pulse_kernel(kernel_width=kernel_width),
                    mode="same",
                )
                for i_row in np.arange(n_row)
            ]
        )

    ap_col = np.zeros(n_row, dtype=int)  # (n_row,) the column indices of the aperture
    ap_mask = np.zeros(n_row, dtype=bool)  # (n_row,) the mask of the aperture

    ap_col[init_row] = init_col
    # search in current row
    i_row = init_row  # this is special
    chunk_data = image[
        i_row, ap_col[i_row] - 2 * kernel_width : ap_col[i_row] + 1 + 2 * kernel_width
    ]
    argmax = argrelextrema(
        chunk_data, np.greater_equal, order=kernel_width, mode="clip"
    )[0]
    ind_min_dev = np.argmin(np.abs(argmax - 2 * kernel_width))
    if np.min(np.abs(argmax[ind_min_dev] - 2 * kernel_width)) <= max_dev:
        ap_col[i_row] = ap_col[i_row] - 2 * kernel_width + argmax[ind_min_dev]
    else:
        if verbose:
            logging.warning(
                f"Warning: deviation({np.min(np.abs(argmax[ind_min_dev] - 2 * kernel_width)):d})"
                f" larger than max_dev({max_dev}) in Row {i_row}"
            )
        ap_col[i_row] = ap_col[i_row]
        ap_mask[i_row] = True

    # search the above
    for i_row in np.arange(init_row)[::-1]:
        # try:
        chunk_start = max(0, ap_col[i_row + 1] - 2 * kernel_width)
        chunk_stop = min(n_col - 1, ap_col[i_row + 1] + 1 + 2 * kernel_width)
        chunk_data = image[i_row, chunk_start:chunk_stop]
        argmax = argrelextrema(
            chunk_data, np.greater_equal, order=kernel_width, mode="clip"
        )[0]
        ind_min_dev = np.argmin(np.abs(argmax + chunk_start - ap_col[i_row + 1]))
        if np.min(np.abs(argmax[ind_min_dev] - 2 * kernel_width)) <= max_dev:
            ap_col[i_row] = ap_col[i_row + 1] - 2 * kernel_width + argmax[ind_min_dev]
        else:
            if verbose:
                logging.warning(
                    f"Warning: deviation({np.min(np.abs(argmax[ind_min_dev] - 2 * kernel_width)):d})"
                    f" larger than max_dev({max_dev}) in Row {i_row}"
                )
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
        chunk_data = image[i_row, chunk_start:chunk_stop]
        argmax = argrelextrema(
            chunk_data, np.greater_equal, order=kernel_width, mode="clip"
        )[0]
        ind_min_dev = np.argmin(np.abs(argmax + chunk_start - ap_col[i_row - 1]))
        if np.min(np.abs(argmax[ind_min_dev] - 2 * kernel_width)) <= max_dev:
            ap_col[i_row] = ap_col[i_row - 1] - 2 * kernel_width + argmax[ind_min_dev]
        else:
            logging.warning(
                f"Warning: deviation({np.min(np.abs(argmax[ind_min_dev] - 2 * kernel_width)):d})"
                f" larger than maxdev({max_dev}) in Row {i_row}"
            )
            ap_col[i_row] = ap_col[i_row - 1]
            ap_mask[i_row] = True
        # except:
        #     ap_col[i_row] = ap_col[i_row - 1]
        #     ap_mask[i_row] = True

    return ap_col, ap_mask


if __name__ == "__main__":
    import os
    from astropy.io import fits
    import numpy as np

    print("Current workding directory:", os.getcwd())
    fp = "/Users/cham/VSCProjects/songcn/data/20191031/flat-bias.fits"

    print(f"Read file: {fp}")
    image = fits.getdata(fp)

    print("Get central slice")
    central_ind = int(image.shape[0] / 2)
    central_slice = image[central_ind : central_ind + 10].sum(axis=0)

    print("Search for local maxima with different kernel widths")
    for kernel_width in np.arange(5, 30):
        localmax = find_local_max(x=central_slice, kernel_width=kernel_width)
        print(
            f"kernel_width: {kernel_width}, number of max: {localmax.num}, "
            f"median SNR: {np.median(localmax.max_snr):.2f}, mean SNR: {np.mean(localmax.max_snr):.2f}"
        )

    print("Approximately set kernel_width to 15")
    localmax = find_local_max(x=central_slice, kernel_width=15)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(
        np.log10(image),
        extent=(
            -0.5,
            2047.5,
            -0.5,
            2047.5,
        ),
        origin="lower",
    )
    for i_ap in np.arange(localmax.num):
        ap_col, ap_mask = trace_one_aperture(
            image,
            init_pos=(central_ind, localmax.max_ind[i_ap]),  # (1024, 2026)
            extra_bin=1,
            kernel_width=15,
            max_dev=3,
        )
        ax.plot(ap_col, np.arange(image.shape[0]), c="k")
        print(f"i_ap = {i_ap}, masksum = {sum(ap_mask)}")
    plt.show()

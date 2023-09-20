from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from skimage import filters, morphology


def smooth_row(
    bg_row: npt.NDArray, q_array: npt.NDArray, regularize=True
) -> npt.NDArray:
    n_element = len(bg_row)
    coord = np.arange(n_element, dtype=int)
    # bg_row = np.array([1, 2, np.nan, np.nan, 1, 2, 4, np.nan, 2])
    # (n_element + 2,)
    bg_row_ext = np.hstack((np.nan, bg_row, np.nan))
    mask = np.isfinite(bg_row_ext).astype(np.int8)
    # (n_element + 1,)
    mask_diff = np.diff(mask)
    # (n_chunk, 2) int
    chunk_indices = np.array([np.where(mask_diff > 0)[0], np.where(mask_diff < 0)[0]]).T

    if regularize:
        # assume the apertures are regular
        # limit the left/right edge aperture, extrapolate to the outer region.
        # so that the outer part of the image (there is light but not identified as apertures)
        # will not affect the identified ones.
        chunk_indices[0, 0] = max(
            0, chunk_indices[0, 1] - chunk_indices[1, 1] + chunk_indices[1, 0]
        )
        chunk_indices[-1, 1] = min(
            chunk_indices[-1, 1],
            chunk_indices[-1, 0] + chunk_indices[-2, 1] - chunk_indices[-2, 0],
        )

    # for i in range(chunk_indices.shape[0]):
    #     print(chunk_indices[i])

    n_chunks = chunk_indices.shape[0]
    coord_chunks = [
        np.mean(
            coord[chunk_indices[i_chunk, 0] : chunk_indices[i_chunk, 1]],
        )
        for i_chunk in range(n_chunks)
    ]
    data_chunks = [
        np.percentile(
            bg_row[chunk_indices[i_chunk, 0] : chunk_indices[i_chunk, 1]],
            q=np.mean(q_array[chunk_indices[i_chunk, 0] : chunk_indices[i_chunk, 1]]),
        )
        for i_chunk in range(n_chunks)
    ]
    # for i_chunk in range(n_chunks):
    #     print(
    #         f"q={np.mean(q_array[chunk_indices[i_chunk, 0] : chunk_indices[i_chunk, 1]])}"
    #     )

    bg_row_interp = np.interp(
        coord,
        coord_chunks,
        data_chunks,
    )
    smoothed_row = np.where(np.isfinite(bg_row), bg_row, bg_row_interp)
    smoothed_row[: int(np.floor(coord_chunks[0]))] = (
        1.5 * data_chunks[0] - 0.5 * data_chunks[1]
    )
    smoothed_row[int(np.ceil(coord_chunks[-1])) :] = (
        1.5 * data_chunks[-1] - 0.5 * data_chunks[-2]
    )
    return smoothed_row


def smooth_background(
    bg: npt.NDArray,
    q: Union[float, tuple[float, float]] = (40, 5),
    sigma_median: Optional[int] = 11,
    sigma_gaussian: Optional[int] = 11,
    regularize=True,
) -> npt.NDArray:
    """Smooth background with median and gaussian filter."""
    # determine image shape
    n_row, n_col = bg.shape
    # generate q array
    if type(q) is float:
        # constant q
        # print(f"Const. q: {q}")
        q_array = q * np.ones(n_col, dtype=float)
    else:
        # linear q
        # print(f"Linear q: {q}")
        q_array = np.linspace(start=q[0], stop=q[1], num=n_col, dtype=float)

    # step 1: interpolate bg row by row
    bg_smoothed = np.array(
        [
            smooth_row(bg[i_row], q_array, regularize=regularize)
            for i_row in range(n_row)
        ]
    )

    # step 2: smooth bg
    if sigma_median is not None:
        print("Apply median filter ...")
        bg_smoothed = filters.median(
            bg_smoothed, footprint=morphology.disk(sigma_median), mode="reflect"
        )
    if sigma_gaussian is not None:
        print("Apply gaussian filter ...")
        bg_smoothed = filters.gaussian(
            bg_smoothed, sigma=sigma_gaussian, mode="reflect"
        )

    return bg_smoothed

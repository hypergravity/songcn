import numpy as np


def trace_local_max(img, starting_row, starting_col, maxdev=10, fov=20, ntol=5):
    """ trace aperture from a starting point (row, col) """
    nrow, ncol = img.shape
    assert ntol < starting_col < ncol - ntol
    # find the local max for this column
    this_ind_row = -np.ones(nrow, dtype=int)
    # this_ind_col = np.arange(ncol, dtype=int)
    # search this column
    this_ind_row[starting_col] = find_local_max(
        img[:, starting_col], starting_row, maxdev=maxdev, fov=fov)
    # search left
    for icol in range(starting_col - 1, -1, -1):
        # find the closest finite value
        icol_right = np.min((icol + 1 + ntol, starting_col + 1))
        icol_closest_valid = np.where(this_ind_row[icol + 1:icol_right] > 0)[0]
        if len(icol_closest_valid) > 0:
            this_ind_row[icol] = find_local_max(
                img[:, icol], this_ind_row[icol + 1:icol_right][icol_closest_valid[0]], maxdev=maxdev, fov=fov)
        else:
            break
    # search right
    for icol in range(starting_col + 1, ncol, 1):
        # find the closest finite value
        icol_left = np.max((icol - ntol, starting_col))
        icol_closest_valid = np.where(this_ind_row[icol_left:icol] > 0)[0]
        if len(icol_closest_valid) > 0:
            this_ind_row[icol] = find_local_max(
                img[:, icol], this_ind_row[icol_left:icol][icol_closest_valid[-1]], maxdev=maxdev, fov=fov)
        else:
            break
    return this_ind_row


def find_local_max(coldata, irow, maxdev=10, fov=20):
    if irow < 0:
        return -1
    indmax = np.argmax(coldata[irow - fov:irow + fov + 1])
    if np.abs(indmax - fov) > maxdev:
        return -1
    else:
        return indmax + irow - fov

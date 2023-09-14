def find_local_max_2d(img, pos_init=(1, 1), extra_bin=1, kernel_width=10) -> np.ndarray:
    """slice by slice"""
    is_local_max = np.zeros_like(img, dtype=bool)
    for i_row in range(img.shape[0]):
        starting_row = i_row - extra_bin
        stop_row = i_row + 1 + extra_bin
        this_slice = img[starting_row:stop_row].sum(axis=0)
        localmax = find_local_max(this_slice, kernel_width=kernel_width)
        is_local_max[i_row, localmax.max_ind] = True
        print(f"irow = {i_row}, {localmax.num} max found")
    return is_local_max

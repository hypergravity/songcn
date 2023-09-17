from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .background import smooth_background
from .polynomial import PolyFit1D
from .trace import trace_one_aperture, find_local_max_1d, LocalMax


class ApertureList(list):
    """A list of `Aperture` instances."""

    def __init__(self, *args, **kwargs):
        super(ApertureList, self).__init__(*args, **kwargs)

    def __repr__(self):
        return f"<ApertureList: N={len(self)}>"

    @property
    def center(self) -> np.ndarray:
        # gather the centers of all apertures
        return np.array([_.center for _ in self])

    def get_mask(self, width: Optional[int] = 11) -> Optional[np.ndarray]:
        if len(self) == 0:
            return None
        else:
            mask = self[0].get_mask(width=width)
            for _ in self[1:]:
                mask |= _.get_mask(width=width)
            return mask

    def view_profile(self, img, row=0, fov_width=20, norm=True):
        plt.figure()
        for ap in self:
            center = ap.center[row]
            ap_center_floor = ap.ind_col_center_floor[row]
            ap_coord = np.arange(-fov_width, fov_width + 1)
            plt.plot(
                ap_coord,
                img[row, ap_center_floor - fov_width : ap_center_floor + fov_width + 1]
                / img[row, ap_center_floor],
            )

    def clip(self):
        """Reserve the apertures is_good and in_bounds."""
        for idx in range(len(self))[::-1]:
            if not (self[idx].is_good and self[idx].in_bounds):
                self.pop(idx)

    def get_background(
        self,
        image: npt.NDArray,
        width: Optional[int] = None,
        fill_value: float = np.nan,
    ) -> npt.NDArray:
        """Get background (inter-order pixels)."""
        return np.where(self.get_mask(width=width), fill_value, image)

    def smooth_background(
        self,
        image: npt.NDArray,
        width: Optional[int] = None,
        q: Union[float, tuple[float, float]] = (40.0, 5.0),
        sigma_median: Optional[int] = 15,
        sigma_gaussian: Optional[int] = 15.0,
    ) -> npt.NDArray:
        """Smooth background with inter-aperture pixels for ``image``."""
        bg = self.get_background(image, width=width)
        return smooth_background(
            bg,
            q=q,
            sigma_median=sigma_median,
            sigma_gaussian=sigma_gaussian,
        )


class Aperture:
    """Aperture class.

    Attributes
    ----------

    """

    def __init__(
        self,
        ind_col: np.ndarray,
        mask: Union[None, np.ndarray] = None,
        width: int = 15,
        deg: int = 4,
        image_shape: tuple = (2048, 2048),
        image: Union[None, np.ndarray] = None,
    ):
        """ """
        # image shape
        if image is not None:
            self.n_row, self.n_col = image.shape
        else:
            self.n_row, self.n_col = image_shape
        assert self.n_row == len(ind_col)

        # construct aperture coordinates: ind_row and ind_col
        self.ind_row = np.arange(len(ind_col), dtype=int)
        self.ind_col = ind_col
        # other parameters
        self.width = width
        self.mask = np.zeros_like(ind_col, dtype=bool) if mask is None else mask
        # aperture is good if all pixels are good
        self.is_good = sum(self.mask) == 0
        # PolyFit1D result
        self.pf1 = PolyFit1D(
            self.ind_row[~self.mask],
            self.ind_col[~self.mask],
            deg=deg,
            pw=1,
            robust=True,
        )
        self.ind_col_center = self.pf1(self.ind_row)
        self.ind_col_center_floor = np.floor(self.ind_col_center).astype(int)
        self.ind_col_center_remainder = self.ind_col_center - self.ind_col_center_floor
        self.ind_col_left = self.ind_col_center - self.width
        self.ind_col_right = self.ind_col_center + self.width

        self.in_bounds = (
            all(self.ind_col_left > 0)
            and all(self.ind_col_left < self.n_row - 1)
            and all(self.ind_col_right > 0)
            and all(self.ind_col_right < self.n_row - 1)
        )

    @property
    def mesh_col(self) -> np.ndarray:
        """Get index meshgrid for column."""
        return (
            self.ind_col_center_floor.reshape(-1, 1)
            - self.width
            + np.arange(2 * self.width + 1)
        )

    @property
    def mesh_row(self) -> np.ndarray:
        """Get index meshgrid for row."""
        return np.repeat(self.ind_row.reshape(-1, 1), 2 * self.width + 1, axis=1)

    def get_cutout(
        self,
        image: npt.NDArray,
        sub_row: Union[None, tuple] = (0, 256),
        correction: bool = False,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Get cutout image for this aperture.

        Parameters
        ----------
        correction
        image
        sub_row

        Returns
        -------

        """
        if sub_row is None:
            slc = slice(self.n_row)
        else:
            slc = slice(*sub_row)

        mesh_row = self.mesh_row[slc]
        mesh_col = (
            self.mesh_col[slc]
            if not correction
            else self.mesh_col[slc] - self.ind_col_center[slc].reshape(-1, 1)
        )
        mesh_image = image[self.mesh_row, self.mesh_col][slc]
        return mesh_row, mesh_col, mesh_image

    def get_mask(self, width: Optional[int] = None) -> npt.NDArray[np.bool_]:
        """Get the mask of this aperture in the whole image."""
        mask = np.zeros((self.n_row, self.n_col), dtype=bool)
        if width is None:
            # use self.width
            mask[self.mesh_row, self.mesh_col] = True
        else:
            mesh_row = np.repeat(self.ind_row.reshape(-1, 1), 2 * width + 1, axis=1)
            mesh_col = (
                self.ind_col_center_floor.reshape(-1, 1)
                - width
                + np.arange(2 * width + 1)
            )
            mask[mesh_row, mesh_col] = True
        return mask

    def __repr__(self):
        return f"<Aperture: ({self.n_row}, {self.n_col}): is good: {self.is_good}: in bounds:{self.in_bounds}>"

    @staticmethod
    def get_central_row(image: np.ndarray, nrow_stack: int = 1) -> np.ndarray:
        """Stacked central 2*``nrow_stack`` rows into 1d array."""
        assert nrow_stack > 0
        ind_central_row = int(image.shape[0] / 2)
        # stack the central 2*nrow_stack rows
        central_slice = image[
            ind_central_row - nrow_stack : ind_central_row + nrow_stack
        ].sum(axis=0)
        return central_slice

    @staticmethod
    def find_local_max_1d(x: np.ndarray, kernel_width: int = 15) -> LocalMax:
        """Convolve 1d array `x` with a pulse kernel with width `kernel_width`, and find local maxima.

        Parameters
        ----------
        x : np.ndarray
            The 1D array.
        kernel_width : int
            The kernel width in number of pixels.
        """
        return find_local_max_1d(x=x, kernel_width=kernel_width)

    @staticmethod
    def trace_one_aperture(
        image: npt.NDArray,
        init_pos: tuple[int, int] = (1000, 993),
        extra_bin: int = 1,
        kernel_width: int = 10,
        max_dev: int = 5,
        verbose: bool = False,
    ) -> tuple[npt.NDArray, npt.NDArray]:
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
            (ind_col, mask) where ind_col is the aperture center index and mask the validity.
            If the ind_col is bad, mask is set True.
        """
        return trace_one_aperture(
            image,
            init_pos=init_pos,
            extra_bin=extra_bin,
            kernel_width=kernel_width,
            max_dev=max_dev,
            verbose=verbose,
        )

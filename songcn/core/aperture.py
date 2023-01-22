import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from songcn.core.polynomial import PolyFit1D
from songcn.core.trace import trace_one_aperture, find_local_max


class ApertureList(list):
    def __init__(self, *args, **kwargs):
        super(ApertureList, self).__init__(*args, **kwargs)

    def __repr__(self):
        return f"<ApertureList: N={len(self)}>"

    @property
    def ap_center(self):
        return np.array([_.ap_center for _ in self])

    def get_mask(self, ap_width=None):
        if len(self) == 0:
            return False
        else:
            mask = self[0].get_mask(ap_width=ap_width)
            for _ in self[1:]:
                mask |= _.get_mask(ap_width=ap_width)
            return mask

    def view_profile(self, img, row=0, fov_width=20, norm=True):
        plt.figure()
        for ap in self:
            ap_center = ap.ap_center[row]
            ap_center_floor = ap.ap_center_floor[row]
            ap_coord = np.arange(-fov_width, fov_width + 1)
            plt.plot(
                ap_coord,
                img[row, ap_center_floor - fov_width:ap_center_floor + fov_width + 1] / img[row, ap_center_floor]
            )

    def clip(self):
        """ Reserve the apertures is_good and in_bounds. """
        for idx in range(len(self))[::-1]:
            if not (self[idx].is_good and self[idx].in_bounds):
                self.pop(idx)


class Aperture:
    def __init__(self, imgshape, ap_col, ap_mask=None, ap_width=15, deg=4):
        # the image shape is used
        self.n_row, self.n_col = imgshape
        assert self.n_row == len(ap_col)
        self.ap_coord = np.arange(len(ap_col), dtype=int)
        self.ap_col = ap_col
        self.ap_width = ap_width
        self.ap_mask = np.zeros_like(ap_col, dtype=bool) if ap_mask is None else ap_mask
        self.is_good = sum(self.ap_mask) == 0
        self.pf1 = PolyFit1D(
            self.ap_coord[~self.ap_mask],
            self.ap_col[~self.ap_mask],
            deg=deg, pw=1, robust=True
        )
        self.ap_center = self.pf1(self.ap_coord)
        self.ap_center_floor = np.floor(self.ap_center).astype(int)
        self.ap_center_remainder = self.ap_center - self.ap_center_floor
        self.ap_lower = self.ap_center - self.ap_width
        self.ap_upper = self.ap_center + self.ap_width

        self.in_bounds = (
                all(self.ap_lower > 0) and
                all(self.ap_lower < self.n_row - 1) and
                all(self.ap_upper > 0) and
                all(self.ap_upper < self.n_row - 1)
        )

    @property
    def ap_region_col(self):
        return self.ap_center_floor.reshape(-1, 1) - self.ap_width + np.arange(2 * self.ap_width + 1)

    @property
    def ap_region_row(self):
        return np.repeat(self.ap_coord.reshape(-1, 1), 2 * self.ap_width + 1, axis=1)

    def get_cutout(self, img, ind_row=(0, 256)):
        slc = slice(*ind_row)
        return self.ap_region_row[slc], \
               self.ap_region_col[slc], \
               img[self.ap_region_row, self.ap_region_col][slc]

    def get_cutout_corrected(self, img, ind_row=(0, 256)):
        slc = slice(*ind_row)
        return self.ap_region_row[slc], \
            self.ap_region_col[slc] - self.ap_center[slc].reshape(-1, 1), \
            img[self.ap_region_row, self.ap_region_col][slc]

    def get_mask(self, ap_width=None):
        mask = np.zeros((self.n_row, self.n_col), dtype=bool)
        if ap_width is None:
            mask[self.ap_region_row, self.ap_region_col] = True
        else:
            ap_region_row = np.repeat(self.ap_coord.reshape(-1, 1), 2 * ap_width + 1, axis=1)
            ap_region_col = self.ap_center_floor.reshape(-1, 1) - ap_width + np.arange(2 * ap_width + 1)
            mask[ap_region_row, ap_region_col] = True
        return mask

    def __repr__(self):
        return f"<Aperture: ({self.n_row}, {self.n_col}): is good: {self.is_good}: in bounds:{self.in_bounds}>"

    @property
    def region(self):
        return


def test():
    fp = "/Users/cham/projects/song/star_spec/20191105/night/ext/masterflat_20191105_slit5.fits"
    img = fits.getdata(fp)
    central_ind = int(img.shape[0] / 2)
    central_slice = img[central_ind:central_ind + 10].sum(axis=0)
    localmax = find_local_max(arr=central_slice, kernel_width=15)

    import logging
    logging.info("Start tracing apertures ...")
    al = ApertureList()
    for i_ap in tqdm(range(localmax.num), unit="Aperture"):
        # print(f"Tracing Aperture {i_ap}")
        ap_col, ap_mask = trace_one_aperture(
            img, init_pos=(central_ind, localmax.max_ind[i_ap]), extra_bin=1, kernel_width=15, maxdev=4)
        ap = Aperture(img.shape, ap_col, ap_mask, ap_width=15, deg=2)
        al.append(ap)
    al.clip()
    print(al, al.get_mask().sum())
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(np.log10(img), origin="lower")
    axs[1].imshow(al.get_mask(10), origin="lower")

    # al.view_profile(img, 1)  # 11 is good for estimating background

    # remove background
    from songcn.core.background import background

    bg = background(img, al, q=(30, 3), median_sigma=11, gaussian_sigma=11)
    img_bg = img - bg

    # extract spectrum
    al.view_profile(img_bg, 1000)  # 11 is good for estimating background

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    axs[0].imshow(np.log10(img))
    axs[1].imshow(np.log10(bg))
    axs[0].plot(al.ap_center.T, np.arange(2048), "w-")
    axs[1].plot(al.ap_center.T, np.arange(2048), "w-")


# ap = al[30]
#
# ap_row, ap_col, ap_img = ap.get_cutout(img_bg)
#
# figure()
# imshow(ap_row, ap_col+ap.ap_center_remainder[:256][:, None],  ap_img)
#
# #%%
# ap_row, ap_col, ap_img = ap.get_cutout(img_bg, (0, 100))
# from matplotlib.colors import LightSource
# import matplotlib.pyplot as plt
# # Set up plot
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#
# ls = LightSource(270, 45)
# # To use a custom hillshading mode, override the built-in shading and pass
# # in the rgb colors of the shaded surface calculated from "shade".
# rgb = ls.shade(ap_img, cmap=cm.jet, vert_exag=0.1, blend_mode='soft')
# surf = ax.plot_surface(ap_row, ap_col, ap_img, rstride=1, cstride=1, facecolors=rgb,
#                        linewidth=0, antialiased=False, shade=False)
#

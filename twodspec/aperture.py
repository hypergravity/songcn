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
- Fri Nov 25 12:53:24 2016

Modifications
-------------
- re-organized on Tue May 23 21:00:00 2017

Aims
----
- utils for apertures

"""

import warnings
from copy import copy

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.signal import medfilt2d, medfilt
from scipy.stats import binned_statistic

from .background import apbackground
from .trace import trace_canny_col
from .deprecated.aprec import AprecList, Aprec


# ###################################################### #
# Here goes the code that will be used after 2017.04.27
# ###################################################### #

class Aperture(object):
    """ trace apertures from FLAT or SCI images
    
    The Aperture class defines the framework of Aperture instances.
    However, multiple choices of tracing methods are provided, such as
    1> canny edge detector (for SONG FLAT, implemented)
    2> find local maximum (for HRS, implemented but not integrated)
    
    Other possible methods that could be used to detect apertures:
    1> (Generalized) Hough transformation
    2> Sobel operator?
    3> generalized find local maximum (along an axis but using multiple pixels)
    4> peak local max, from skimage
    5> clustering? sounds infeasible
    6> xcorr. cross-correlation between columns
    
    """
    # ######################### #
    # image info
    # ######################### #
    imshape = None
    x = None
    y = None
    mx = None
    my = None

    # ######################### #
    # trace apertures
    # ######################### #
    status_traced = False
    # [aperture] method used to trace apertures
    method = ""
    # [aperture] raw data maintained during tracing
    trace_details = None
    # [aperture] number of apertures found
    n_ap = None

    # [aperture] aperture edges & center
    ap_lower = None
    ap_upper = None
    ap_center = None

    ap_width = None

    # ######################### #
    # fit apertures & interpolate
    # ######################### #
    status_polyfit = False
    # fitted & interpolated apertures
    # fittype = None
    polydeg = None

    # --------------------------------------- #
    # fitted polynomial coefs
    ap_upper_chebcoef = None
    ap_lower_chebcoef = None
    ap_center_chebcoef = None

    # interpolated edge & center
    ap_upper_interp = None
    ap_lower_interp = None
    ap_center_interp = None

    # ######################### #
    # re-fine aperture using xcorr
    # ######################### #
    status_xcorr = False
    # ultra-fine aperture edge & center using xcorr
    ap_upper_xcorr = None
    ap_lower_xcorr = None
    ap_center_xcorr = None

    # ####################################
    # the properties below are deprecated
    # they will go to "details"
    # ####################################

    # region & weight will be geneated instantateously
    # ap_region = None
    # ap_weight = None

    # width will be calculated instantaneously
    # ap_width_original = None
    # ap_width_narrowed = None

    # # raw edges
    # edge = None
    # # adopted edges & info
    # edge_good = None
    # isupper = None
    # isadopted = None
    #
    # # edge int arrays, upper & lower together
    # ap_col = None
    # ap_row = None
    #
    # ap_col_interp = None

    def __init__(self, n_ap=None, ap_center=None, ap_width=15):
        """ initialize with traces

        Parameters
        ----------
        n_ap:
            number of apertures
        ap_center:
            aperture center (n_ap x n_pix)
        ap_width:
            aperture width
        """

        self.n_ap = n_ap

        self.ap_center = ap_center
        if ap_center is not None:
            self.ap_lower = ap_center-ap_width
            self.ap_upper = ap_center-ap_width

        return

    @staticmethod
    def trace(flat, method=None, ap_width=15, verbose=True, *args, **kwargs):
        """ initialize Aperture instance given a method and an image """

        ap = Aperture()

        # assert method is valid
        assert method in {None, "", "canny", "naive"}
        ap.method = method

        # return null Aperture instance
        if method is None or method == "":
            ap.ap_width = ap_width
            return ap

        # ##################### #
        # trace apertures
        # ##################### #

        if verbose:
            print("@Aperture: tracing apertures using method [{0}]".format(method))

        # 1. canny edge detector
        if method == "canny":
            results, details = trace_canny_col(flat, details=True,
                                               verbose=verbose,
                                               *args, **kwargs)

            ap = Aperture(results["n_ap"], results["ap_center"], ap_width)
            ap.trace_details = details

        # 2. naive method
        elif method == "naive":
            pass
        else:
            print("@SONG: invalid method {0}".format(method))

        # change status
        ap.status_traced = True

        # ##################### #
        # verbose
        # ##################### #
        if verbose:
            print("@Aperture: {0} apertures found!".format(ap.n_ap))

        # ##################### #
        # get image info
        # ##################### #
        if verbose:
            print("@Aperture: getting image info ...")
        ap.get_image_info(flat)
        return

    def get_image_info(self, image):
        """ get image information """

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        self.imshape = image.shape
        self.x = np.arange(self.imshape[1], dtype=int)
        self.y = np.arange(self.imshape[0], dtype=int)
        self.mx, self.my = np.meshgrid(self.x, self.y)

    def polyfit(self, deg=4):
        """ fit using chebyshev polynomial for adopted apertures """

        n_col = self.imshape[1]
        n_ap = self.n_ap

        ap_col_interp = np.arange(0, n_col, dtype=int)
        ap_upper_interp = []
        ap_lower_interp = []
        ap_upper_chebcoef = []
        ap_lower_chebcoef = []
        for i in range(n_ap):
            # for upper
            this_chebcoef = np.polynomial.chebyshev.chebfit(
                self.x, self.ap_upper[i], deg=deg)
            ap_upper_chebcoef.append(this_chebcoef)
            ap_upper_interp.append(
                np.polynomial.chebyshev.chebval(ap_col_interp, this_chebcoef))
            # for lower
            this_chebcoef = np.polynomial.chebyshev.chebfit(
                self.x, self.ap_lower[i], deg=deg)
            ap_lower_chebcoef.append(this_chebcoef)
            ap_lower_interp.append(
                np.polynomial.chebyshev.chebval(ap_col_interp, this_chebcoef))

        # transform to numpy.array format
        ap_upper_interp = np.array(ap_upper_interp)
        ap_lower_interp = np.array(ap_lower_interp)
        ap_upper_chebcoef = np.array(ap_upper_chebcoef)
        ap_lower_chebcoef = np.array(ap_lower_chebcoef)

        # interpolated edges
        # self.fittype = fittype,
        self.polydeg = deg,

        # self.ap_col_interp = ap_col_interp,

        self.ap_upper_interp = ap_upper_interp  # these are float arrays
        self.ap_lower_interp = ap_lower_interp
        self.ap_upper_chebcoef = ap_upper_chebcoef
        self.ap_lower_chebcoef = ap_lower_chebcoef

        # center trace: center is not fitted but averaged from edges
        self.ap_center_interp = (ap_upper_interp + ap_lower_interp) / 2.

        print("@SONG: {0} apertures found!".format(n_ap))

    def to_apreclist(self, comment=None, image="flat.fits", axis=2,
                     disp_range=(1, 2048), disp_center=1024,
                     function="chebyshev", order=5, width=(12, 12)):
        """ convert to AprecList instance """

        if function == "chebyshev":
            function = 1

        if isinstance(width, int) or isinstance(width, float):
            width = (np.abs(width), np.abs(width))

        if axis == 1:
            raise(NotImplementedError("Not implemented for axis = 1!"))

            al = AprecList()
            x = self.x + 1  # difference between 1-indexed and 0-indexed array

            ar_list = []
            for i in range(self.n_ap):
                y = self.ap_center[i] + 1

                ar = Aprec()
                ar.image = image
                ar.apID = i + 1
                ar.beamID = i + 1
                ar.low = (-width[0], disp_range[0] - disp_center)
                ar.high = (width[1], disp_range[1] - disp_center)
                ar.axis = axis

                scaler = Scaler(x, (-1, 1))
                # the curve range
                ar.cvrec.range = scaler.scale_back([-1, 1])
                # fit scaled x & y
                x_scaled = scaler.scale(x)
                this_chebcoef = np.polynomial.chebyshev.chebfit(
                    x_scaled, y, deg=order - 1)
                this_y_center = np.polynomial.chebyshev.chebval(
                    scaler.scale(disp_center), this_chebcoef)

                ar.center = this_y_center, disp_center
                this_chebcoef[0] -= this_y_center
                ar.cvrec.coefs = this_chebcoef
                ar.cvrec.function = function
                ar.cvrec.order = order

                ar_list.append(ar)

            al.data = ar_list
            return al
        else:

            al = AprecList()
            x = self.y   # difference between 1-indexed and 0-indexed array

            ar_list = []
            for i in range(self.n_ap):
                # since the rot90 operation:
                y = self.ap_center[i]

                ar = Aprec()
                ar.image = image
                ar.apID = i + 1
                ar.beamID = i + 1
                ar.low = (-width[0], disp_range[0] - disp_center)
                ar.high = (width[1], disp_range[1] - disp_center)
                ar.axis = axis

                scaler = Scaler(x, (-1, 1))
                # the curve range
                ar.cvrec.range = scaler.scale_back([-1, 1])
                # fit scaled x & y
                x_scaled = scaler.scale(x)
                this_chebcoef = np.polynomial.chebyshev.chebfit(
                    x_scaled, y, deg=order - 1)
                this_y_center = np.polynomial.chebyshev.chebval(
                    scaler.scale(disp_center), this_chebcoef)

                ar.center = this_y_center, disp_center
                this_chebcoef[0] -= this_y_center
                ar.cvrec.coefs = this_chebcoef
                ar.cvrec.function = function
                ar.cvrec.order = order

                ar_list.append(ar)

            al.data = ar_list
            return al

    def ap_region(self, i_ap=0, n_extend=0):
        """ return aperture region & weight for a specified aperture """

        this_aperture_weight = np.zeros_like(self.mx, dtype=float)
        this_aperture_condition = np.logical_and(
            self.my > np.floor(self.ap_upper_interp[i_ap] - n_extend),
            self.my < np.ceil(self.ap_lower_interp[i_ap] + n_extend))
        min_disance_to_edge = np.min(np.abs(np.array(
            (self.my - self.ap_upper_interp[i_ap] + n_extend,
             self.my - self.ap_lower_interp[i_ap] - n_extend))), axis=0)
        this_aperture_weight = np.where(
            this_aperture_condition, min_disance_to_edge,
            this_aperture_weight)
        this_aperture_weight = np.where(
            this_aperture_weight > 1, 1.0, this_aperture_weight)

        return this_aperture_condition, this_aperture_weight

    def ap_region_all(self, n_extend=0):
        """ return aperture region & weight for all apertures """

        ap_region_all = np.zeros((self.n_ap, *self.imshape), dtype=bool)
        ap_weight_all = np.zeros((self.n_ap, *self.imshape), dtype=float)

        for i in range(self.n_ap):
            ap_region_all[i], ap_weight_all[i] = \
                self.ap_region(i, n_extend=n_extend)

        return ap_region_all, ap_weight_all

    def ap_region_chunk(self, i_ap=0, chunk_cols=(0, 2048), n_extend=0,
                        n_cushion=3, scheme='global'):
        """  return chunk slice, data and mask
        
        Parameters
        ----------
        i_ap: int
            the aperture serial number
        chunk_cols: tuple
            (start column, stop column)
        n_extend: int
            aperture extension
        n_cushion: int
            cushion for cutting a chunk, could be filled with specified value
        scheme: string
            if local, return a local cut
            if global, upper & lower limit dtermined using the whole aperture

        Returns
        -------

        """

        n_cushion = np.int(n_cushion)

        # column slice
        slice_col = slice(np.int(chunk_cols[0]), np.int(chunk_cols[1]), 1)

        # row slice
        if scheme == 'local':
            lower_limit = np.int(
                np.ceil(self.ap_lower_interp[i_ap, slice_col].max()))
            upper_limit = np.int(
                np.floor(self.ap_upper_interp[i_ap, slice_col].min()))
        elif scheme == "global":
            lower_limit = np.int(np.ceil(self.ap_lower_interp[i_ap, :].max()))
            upper_limit = np.int(np.floor(self.ap_upper_interp[i_ap, :].min()))
        else:
            raise (ValueError("@Aperture: Invalid scheme: {0}".format(scheme)))
        slice_row = slice(np.int(upper_limit - n_cushion - n_extend),
                          np.int(lower_limit + 1 + n_cushion + n_extend), 1)

        # cushion_mask
        this_aperture_condition, this_aperture_weight = \
            self.ap_region(i_ap=i_ap, n_extend=n_extend)
        chunk_mask = this_aperture_condition[slice_row, slice_col]
        chunk_weight = this_aperture_weight[slice_row, slice_col]

        return dict(
            slice_=(slice_row, slice_col),
            slice_row=slice_row,
            slice_col=slice_col,
            mask=chunk_mask,
            weight=chunk_weight
        )

    def ap_region_chunk_rdc(self, i_ap=0, cols=(0, 512), widths=(15, 15)):
        """ this is to get the chunk for pyREDUCE """
        y1, y2 = np.modf(self.ap_center_interp[i_ap][cols[0]:cols[1]])
        nrow = np.int(widths[1] + widths[0])
        ncol = np.int(cols[1] - cols[0])
        slice1 = np.repeat(
            np.arange(cols[0], cols[1], 1, dtype=int).reshape(1, -1),
            nrow, axis=0)
        slice0 = np.repeat(
            np.arange(-widths[0], widths[1], 1, dtype=int).reshape(-1, 1),
            ncol, axis=1) + y2.astype(int)
        return slice0, slice1, y1

    @property
    def ap_width_original(self):
        """ return original aperture widths """
        return np.median(self.ap_lower_interp - self.ap_upper_interp, axis=1)

    def ap_width_extended(self, n_extended=0):
        """ return extended aperture widths """
        return np.median(self.ap_lower_interp - self.ap_upper_interp, axis=1) \
               + 2 * n_extended

    # def calculate_aperture_region(self, n_narrower=0):
    #     assert -9 < n_narrower < 9
    #
    #     x = np.arange(self.imshape[1], dtype=int)
    #     y = np.arange(self.imshape[0], dtype=int)
    #     mx, my = np.meshgrid(x, y)
    #     ap_region = np.zeros((self.n_ap, *self.imshape), dtype=bool)
    #     ap_weight = np.zeros((self.n_ap, *self.imshape), dtype=float)
    #     for i in range(self.n_ap):
    #         this_aperture_weight = np.zeros_like(mx, dtype=float)
    #         aperture_condition = np.logical_and(
    #             my > np.floor(self.ap_upper_interp[i]-n_narrower),
    #             my < np.ceil(self.ap_lower_interp[i]+n_narrower))
    #         min_disance_to_edge = np.min(np.abs(np.array(
    #             (my - self.ap_upper_interp[i] + n_narrower,
    #              self.ap_lower_interp[i] - my - n_narrower))), axis=0)
    #         this_aperture_weight = np.where(
    #             aperture_condition, min_disance_to_edge, this_aperture_weight)
    #         this_aperture_weight = np.where(
    #             this_aperture_weight > 1, 1.0, this_aperture_weight)
    #         ap_region[i] = aperture_condition
    #         ap_weight[i] = this_aperture_weight
    #
    #     return ap_region, ap_weight
    #
    # def update_aperture_region(self, n_narrower=0):
    #     self.ap_region, self.ap_weight = self.calculate_aperture_region(
    #         n_narrower=n_narrower)
    #     self.ap_width_original = np.median(
    #         self.ap_lower_interp - self.ap_upper_interp, axis=1) - 2*n_narrower

    def flatten(self, flat, method='median', **kwargs):
        """  to flatten FLAT

        Parameters
        ----------
        flat:
            the image to be flattened
        method: string
            {"median"}
        kwargs:
            to be passed to specific flattening method

        Returns
        -------
        flattened image

        """
        # copy data
        im = copy(np.array(flat))

        # flatten image
        im_flattened = self._flatten_median(im, **kwargs)
        im_sensitivity = im/im_flattened
        im_sensitivity = np.where(
            np.logical_or(np.logical_not(self.ap_region.sum(axis=0)),
                          np.logical_not(np.isfinite(im_sensitivity))),
            1.0, im_sensitivity)

        return im_flattened, im_sensitivity

    def _flatten_median(self, flat, kernel_size=(11, 3)):
        """ to flatten FLAT using medfilt2d

        Parameters
        ----------
        flat
        kernel_size

        Returns
        -------

        """
        im_flattened = medfilt2d(flat, kernel_size=kernel_size)
        nht = np.fix(kernel_size[0]/2)
        im_flattened[:, :nht] = flat[:, :nht]
        im_flattened[:, -nht:] = flat[:, -nht:]
        print(nht, im_flattened[:, -nht:])
        return im_flattened

    def extract_simple(self, im):
        """ simple extraction """
        return np.nansum((np.array([im])*self.ap_weight), axis=1)

    def extract_median(self, im):
        """ simple extraction """
        data = np.array([im])*self.ap_weight
        data = np.where(self.ap_weight > 0, data, np.nan)
        return np.nanmean(data, axis=1)

    def background(self, im, npix_inter=5, q=(40, 5), sigma=(10, 10), kernel_size=(11, 11)):
        """ newly developed on 2017-05-28, with best performance """
        return apbackground(im, self.ap_center, q=q,
                            npix_inter=npix_inter, sigma=sigma,
                            kernel_size=kernel_size)

    # ####################################################################### #
    # Old decomposition code, decompose an order using an empirical spatial
    # profile and requires the position to be accurately determined, not useful
    # any more.
    # ####################################################################### #
    def decomposition(self, im, im_var, vbinning=4, n_chunks=1,
                      smooth_kernel=3, cushion=4):

        warnings.warn("DEPRECATED use stella instead ", DeprecationWarning)

        im_var = np.abs(im_var)
        im_model = np.zeros_like(im, dtype=float)
        im_profile = np.zeros_like(im, dtype=float)
        blazes = np.zeros((self.n_ap, im.shape[1]), dtype=float)

        # determine chunk start & stop
        chunk_len_typical = np.floor(im.shape[1]/n_chunks).astype(int)
        chunk_len = np.array([chunk_len_typical] * n_chunks).astype(int)
        chunk_len[-1] = im.shape[1] - np.sum(chunk_len[:-1]).astype(int)
        chunk_start = np.hstack((0, np.cumsum(chunk_len)[:-1])).astype(int)
        chunk_stop = np.cumsum(chunk_len).astype(int)
        print("@SONG: break each order to chunks with length: ", chunk_len)

        for i_order in range(self.n_ap):
            # determine the section for this order
            ind_this_order = np.any(self.ap_region[i_order], axis=1)

            # usually a cushion is needed in order to fit blaze function
            if cushion > 0:
                i_head, i_tail = np.where(ind_this_order)[0][[0, -1]]
                for i in range(cushion):
                    ind_this_order[np.max((0, i_head - 1 - i))] = True
                    ind_this_order[np.min((len(ind_this_order) - 1, i_tail + 1 + i))] = True

            # section span in Y direction
            n_yspan = len(np.where(ind_this_order)[0])
            # aperture center offset relative to the center of this section
            this_order_center_ofst = - self.ap_center_interp[i_order] + \
                                     np.where(ind_this_order)[0][0] + 0.5 * n_yspan
            # cut the image to get the section for this order
            this_order = (im * self.ap_region[i_order])[ind_this_order]
            this_order_var = (im_var * self.ap_region[i_order])[ind_this_order]
            # this_order_x = self.mx[ind_this_order]
            # this_order_y = self.my[ind_this_order]

            # break
            for i_chunk in range(n_chunks):
                this_order_chunk = this_order[:, chunk_start[i_chunk]:chunk_stop[i_chunk]]
                this_order_chunk_var = this_order_var[:, chunk_start[i_chunk]:chunk_stop[i_chunk]]
                profile_ofst = this_order_center_ofst[chunk_start[i_chunk]:chunk_stop[i_chunk]]

                # decomposite this chunk
                this_order_chunk_profile_reconstructed, \
                this_order_chunk_reconstructed, blaze = \
                    decomposition_single_order_single_chunk(
                        this_order_chunk, this_order_chunk_var, profile_ofst,
                        vbinning=vbinning, smooth_kernel=smooth_kernel)

                # co-add the chunk model to the final model
                im_model[ind_this_order, chunk_start[i_chunk]:chunk_stop[i_chunk]] \
                    += this_order_chunk_reconstructed
                # co-add the chunk profile to the final profile
                im_profile[ind_this_order, chunk_start[i_chunk]:chunk_stop[i_chunk]] \
                    += this_order_chunk_profile_reconstructed
                # collect blaze function
                blazes[i_order, chunk_start[i_chunk]:chunk_stop[i_chunk]] = blaze
            print("@SONG: modelling order ", i_order)

        return im_model, im_profile, blazes


def decomposition_single_order_single_chunk(
        this_order, this_order_var, profile_ofst,
        vbinning=4, smooth_kernel=None):
    """

    Parameters
    ----------
    this_order: (n_yspan, n_xall)
        the image block
    this_order_var: (n_yspan, n_xall)
        the variance block
    profile_ofst:
        the offset relative to
    vbinning: int
        an oversampling factor
    smooth_blaze: None | int
        if not None, perform a smoothing on the blaze function
        if None, do not smooth

    Returns
    -------

    """
    # make sure *this_order* is an array
    this_order = np.array(this_order)

    # determine profile coordinates, unbinned & binned
    binning_step = 1.0 / vbinning
    profile1d_coord_unbinned = np.arange(
        0, this_order.shape[0], binning_step) - (vbinning-1.)/(2.*vbinning)
    profile1d_coord_unbinned_edge = np.hstack((
        profile1d_coord_unbinned - 0.5 * binning_step,
        profile1d_coord_unbinned[-1] + 0.5 * binning_step))
    profile1d_coord_binned = np.arange(0, this_order.shape[0], 1.0)

    # determine the profile using median
    x = profile1d_coord_binned.reshape(-1, 1) + profile_ofst.reshape(1, -1)
    blaze0 = np.sum(this_order, axis=0)
    values = this_order / blaze0.reshape(1, -1)  # blaze0 only used here
    # do *binned_statistic*
    profile1d_unbinned, bin_edges, binnumber = binned_statistic(
        x.flatten(), values.flatten(),
        statistic=np.nanmedian,
        bins=profile1d_coord_unbinned_edge)
    profile1d_unbinned = np.where(np.isfinite(profile1d_unbinned),
                                  profile1d_unbinned, 0.)
    # force positivity
    profile1d_unbinned = np.where(profile1d_unbinned > 0.,
                                  profile1d_unbinned, 0.)
    # force normalization
    profile1d_unbinned /= np.sum(profile1d_unbinned)

    # determine the blaze function (3 sigma rejection?)
    this_order_profile_reconstructed = binning2dud(
        offset_profile(profile1d_coord_unbinned, profile1d_unbinned,
                       profile_ofst), vbinning)
    this_order_profile_reconstructed /= \
        np.sum(this_order_profile_reconstructed, axis=0).reshape(1, -1)

    # determine the blaze function again
    blaze1 = np.zeros((this_order.shape[1],), dtype=float)
    for i_col in range(this_order.shape[1]):
        blaze1[i_col], ier = leastsq(
            costfun, blaze0[i_col],
            args=(this_order[:, i_col],
                  this_order_profile_reconstructed[:, i_col],
                  this_order_var[:, i_col]),
            xtol=1.0
        )
    this_order_reconstructed = blaze1.reshape(1, -1) * \
                               this_order_profile_reconstructed

    if smooth_kernel is None:
        # no smoothing
        return this_order_profile_reconstructed, this_order_reconstructed, blaze1
    else:
        # smooth blaze function and feed back to reconstructed imaged
        # kernel_size = 3
        lenend = np.int(np.ceil(smooth_kernel/2.))
        blaze2 = medfilt(blaze1, smooth_kernel)
        blaze2[:lenend] = blaze1[:lenend]
        blaze2[-lenend:] = blaze1[-lenend:]
        this_order_reconstructed *= (blaze2/blaze1).reshape(1, -1)
        return this_order_profile_reconstructed, this_order_reconstructed, blaze2


def costfun(x, profile_obs, profile_temp, profile_var):
    weight = np.sqrt(profile_var)
    # istd = np.where(np.isfinite(istd), istd, 0)
    return (x * profile_temp - profile_obs) * weight


def offset_profile(profile1d_coord, profile1d, ofsts):
    f = interp1d(profile1d_coord, profile1d, kind='slinear', bounds_error=False, fill_value=0.)
    profile2d_reconstructed = np.array([f(profile1d_coord+ofst) for ofst in ofsts])
    return profile2d_reconstructed.T


def binning2dlr(data, binning_factor=2):
    data = np.array(data)
    binning_factor = np.int(binning_factor)
    assert binning_factor >= 2
    assert data.ndim == 2
    assert np.mod(data.shape[1], binning_factor) == 0
    binned_data = np.zeros((data.shape[0], np.int(data.shape[1]/binning_factor)), dtype=data.dtype)
    for i in range(binning_factor):
        binned_data += data[:, i::binning_factor]
    return binned_data


def binning2dud(data, binning_factor=2):
    data = np.array(data)
    binning_factor = np.int(binning_factor)
    assert binning_factor >= 2
    assert data.ndim == 2
    assert np.mod(data.shape[0], binning_factor) == 0
    binned_data = np.zeros((np.int(data.shape[0]/binning_factor), data.shape[1]), dtype=data.dtype)
    for i in range(binning_factor):
        binned_data += data[i::binning_factor, :]
    return binned_data


class Scaler(object):

    def __init__(self, arr, range_):
        """ map """
        arr = np.array(arr)
        range_ = np.array(range_)

        scale = np.abs((range_[0] - range_[-1]) / (arr[-1] - arr[0]))
        arr_scale = arr * scale
        ofst = np.min(arr_scale) - np.min(range_)
        arr_scale_ofst = arr_scale - ofst

        self.arr = arr
        self.arr_scaled_ofst = arr_scale_ofst
        self.scale_ = scale
        self.ofst_ = ofst

    def scale(self, data):
        return np.array(data) * self.scale_ - self.ofst_

    def scale_back(self, data):
        return (np.array(data) + self.ofst_) / self.scale_





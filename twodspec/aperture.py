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

import numpy as np

from .background import apbackground
from .extract import (get_aperture_section, extract_profile, extract_aperture, extract_all,
                      make_normflat, extract_profile_simple)
from .trace import trace_naive_max


# ###################################################### #
# Here goes the code that will be used after 2017.04.27  #
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
    # image info
    imshape = None
    x = None
    y = None
    mx = None
    my = None

    # trace apertures
    nap = 0  # number of apertures found
    npix = 0
    istraced = False  # True if traced
    method = ""  # method used to trace apertures
    trace_details = None  # raw data maintained during tracing

    # aperture edges, centers and width
    ap_lower = None
    ap_upper = None
    ap_center = None
    ap_width = 0

    # fit apertures & interpolate
    ispolyfitted = False
    polydeg = 0

    # fitted polynomial coefs
    ap_upper_chebcoef = None
    ap_lower_chebcoef = None
    ap_center_chebcoef = None  # center is not fitted in case of canny method

    # interpolated edge & center
    ap_upper_interp = None
    ap_lower_interp = None
    ap_center_interp = None

    def __init__(self, ap_center=np.array([[]]), ap_width=15):
        """ initialize with traces

        Parameters
        ----------
        ap_center:
            aperture center (n_ap x n_pix)
        ap_width:
            aperture width
        """
        ap_center = np.asarray(ap_center, float)
        self.nap, self.npix = ap_center.shape
        self.ap_center = ap_center
        if ap_center is not None:
            self.ap_lower = ap_center - ap_width
            self.ap_upper = ap_center + ap_width
        self.ap_width = ap_width
        return

    @staticmethod
    def trace(flat, method="naive", ap_width=15, polydeg=4, **kwargs):
        """ trace apertures for FLAT with a specified method

        Example
        -------
        >>> from twodspec.aperture import Aperture
        >>> ap = Aperture.trace(flat, method="naive", polydeg=4, sigma=7, maxdev=7, ap_width=15)
        >>> ap = Aperture.trace(flat, method="canny", polydeg=4, sigma=7, maxdev=7, ap_width=15)

        Parameters
        ----------
        flat : ndarray
            FLAT.
        method : str, optional
            {"naive", "canny", "", None}. The default is None.
        ap_width : float, optional
            the half width of the aperture. The default is 15.
        polydeg : int, optional
            The order of polynomial fitting to apertures. The default is 4.
        **kwargs :
            will be passed to trace method.

        Returns
        -------
        ap : Aperture instance
            The Aperture results.

        """
        # assert method is valid
        assert method in {None, "", "canny", "naive"}

        # trace apertures
        print("@Aperture: tracing apertures using [{0}] method".format(method), end="")

        # return null Aperture instance
        if method is None or method == "":
            # initialization
            ap = Aperture()
            # get image info
            ap.get_image_info(flat)
            return ap
        elif method == "canny":
            # 1. canny edge detector
            results, details = trace_canny_col(flat, details=True, verbose=False, sigma=kwargs["sigma"])
            ap = Aperture(ap_center=results["ap_center"], ap_width=ap_width)
            ap.get_image_info(flat)
            ap.trace_details = details
        elif method == "naive":
            # 2. naive max method
            ap_center = trace_naive_max(flat, sigma=kwargs["sigma"], maxdev=kwargs["maxdev"])
            ap = Aperture(ap_center=ap_center, ap_width=ap_width)
            ap.get_image_info(flat)
        else:
            # otherwise
            print("\n@Aperture: invalid method {0}".format(method))
            return Aperture()
        # change status
        ap.method = method
        ap.istraced = True
        # verbose
        print("  >>>  {0} apertures found!".format(ap.nap))

        # polyfit
        if polydeg is not None:
            ap.polyfit(np.int(polydeg))

        return ap

    def get_image_info(self, image):
        """ get image information """
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        self.imshape = image.shape
        self.x = np.arange(self.imshape[1], dtype=int)
        self.y = np.arange(self.imshape[0], dtype=int)
        self.mx, self.my = np.meshgrid(self.x, self.y)
        return

    def polyfit(self, deg=4):
        """ fit using chebyshev polynomial for adopted apertures """
        # interpolated edges
        self.polydeg = deg

        n_row = self.imshape[0]
        nap = self.nap

        ap_col_interp = np.arange(0, n_row, dtype=int)
        ap_upper_interp = []  # interpolated
        ap_lower_interp = []
        ap_center_interp = []
        ap_upper_chebcoef = []  # chebcoef
        ap_lower_chebcoef = []
        ap_center_chebcoef = []
        for i in range(nap):
            # for upper
            this_chebcoef = np.polynomial.chebyshev.chebfit(
                self.y, self.ap_upper[i], deg=deg)
            ap_upper_chebcoef.append(this_chebcoef)
            ap_upper_interp.append(
                np.polynomial.chebyshev.chebval(ap_col_interp, this_chebcoef))
            # for lower
            this_chebcoef = np.polynomial.chebyshev.chebfit(
                self.y, self.ap_lower[i], deg=deg)
            ap_lower_chebcoef.append(this_chebcoef)
            ap_lower_interp.append(
                np.polynomial.chebyshev.chebval(ap_col_interp, this_chebcoef))
            # for center
            this_chebcoef = np.polynomial.chebyshev.chebfit(
                self.y, self.ap_center[i], deg=deg)
            ap_center_chebcoef.append(this_chebcoef)
            ap_center_interp.append(
                np.polynomial.chebyshev.chebval(ap_col_interp, this_chebcoef))

        # transform to numpy.array format
        self.ap_upper_interp = np.array(ap_upper_interp)
        self.ap_lower_interp = np.array(ap_lower_interp)
        self.ap_center_interp = np.array(ap_center_interp)
        self.ap_upper_chebcoef = np.array(ap_upper_chebcoef)
        self.ap_lower_chebcoef = np.array(ap_lower_chebcoef)
        self.ap_center_chebcoef = np.array(ap_center_chebcoef)
        # center trace: center is not fitted but averaged from edges
        # self.ap_center_interp = (ap_upper_interp + ap_lower_interp) / 2.

        self.ispolyfitted = True
        return

    def background(self, im, npix_inter=5, q=(40, 5), sigma=(10, 10), kernel_size=(11, 11)):
        """ newly developed on 2017-05-28, with best performance """
        return apbackground(im, self.ap_center, q=q, npix_inter=npix_inter, 
                            sigma=sigma, kernel_size=kernel_size)
    
    def get_aperture_section(self, im, iap):
        """ get an aperture section

        Parameters
        ----------
        im : ndarray
            The target image.
        iap : int
            The aperture index.

        Returns
        -------
        ap_im : ndarray
            DESCRIPTION.
        ap_im_xx : ndarray
            x coordinates.
        ap_im_yy : ndarray
            y coordinates.
        ap_im_xx_cor : ndarray
            x offset from center.

        """
        return get_aperture_section(im, self.ap_center_interp[iap], ap_width=self.ap_width)

    def extract_profile_simple(self, ap_im, ap_im_xx_cor, profile_oversample=10, profile_smoothness=1e-1):

        return extract_profile_simple(ap_im, ap_im_xx_cor, self.ap_width,
                                      profile_oversample=profile_oversample,
                                      profile_smoothness=profile_smoothness)

    def extract_profile(self, ap_im, ap_im_xx_cor, profile_smoothness=1e-2, n_chunks=8,
                        ap_width=15., profile_oversample=10., ndev=4):
        return extract_profile(ap_im, ap_im_xx_cor, profile_smoothness=profile_smoothness, n_chunks=n_chunks,
                               ap_width=ap_width, profile_oversample=profile_oversample, ndev=ndev)

    def make_normflat(self, im, max_dqe=0.04, min_snr=20, smooth_blaze=5, n_chunks=8,
                      profile_oversample=10, profile_smoothness=1e-2, num_sigma_clipping=20, gain=1., ron=0, n_jobs=1):
        """ normalize FLAT
        
        Parameters
        ----------
        im : ndarray
            the target image.
        max_dqe : float, optional
            The max deviation of Quantum Efficiency from 1.0. The default is 0.04.
        min_snr : flaot, optional
            Ignore the region with snr<min_snr. The default is 20.
        smooth_blaze : int, optional
            The smooth kernel width / pixel. The default is 5.
        n_chunks : int, optional
            Split each aperture to n_chunks chunks. The default is 8.
        profile_oversample : int, optional
            Oversampling factor of spatial profile. The default is 10.
        profile_smoothness : float, optional
            The smoothness of the profile. The default is 1e-2.
        num_sigma_clipping : float, optional
            The sigma-clipping value / sigma. The default is 20.
        gain : flaot, optional
            The gain of the image. The default is 1..
        ron : float, optional
            The readout noise. The default is 0.
        n_jobs : int, optional
            The number of processes launched. The default is 1.

        Returns
        -------
        blaze, im_norm : ndarray
            The blaze functions and sensitivity image.

        """
        blaze, im_norm = make_normflat(
            im, self, max_dqe=max_dqe, min_snr=min_snr, smooth_blaze=smooth_blaze, n_chunks=n_chunks,
            profile_oversample=profile_oversample, profile_smoothness=profile_smoothness,
            num_sigma_clipping=num_sigma_clipping, gain=gain, ron=ron, n_jobs=n_jobs)
        return blaze, im_norm

    def extract_all(self, im, n_chunks=8, profile_oversample=10, profile_smoothness=1e-2,
                    num_sigma_clipping=10, gain=1., ron=0, n_jobs=-1):
        """ extract all apertures with both simple & profile extraction

        Parameters
        ----------
        im : ndarray
            The target image.
        n_chunks : int, optional
            The number of chunks. The default is 8.
        profile_oversample : int, optional
            The oversampling factor of the profile. The default is 10.
        profile_smoothness : float, optional
            The smoothness of the profile. The default is 1e-2.
        num_sigma_clipping : float, optional
            The sigma clipping threshold. The default is 5..
        gain : float, optional
            The gain of CCD. The default is 1..
        ron : flaot, optional
            The readout noise of CCD. The default is 0.
        n_jobs : int, optional
            The number of processes launched. The default is -1.

        Returns
        -------
        dict
            a dict sconsisting of many results.

        """

        return extract_all(im, self, n_chunks=8, profile_oversample=10, profile_smoothness=1e-2,
                           num_sigma_clipping=10, gain=1., ron=0, n_jobs=n_jobs)

    def extract_aperture(self, iap, im, n_chunks=8, profile_oversample=10, profile_smoothness=1e-2,
                         num_sigma_clipping=5., gain=1., ron=0):
        """ Extract one aperture

        Parameters
        ----------
        iap : int
            Extract the iap th aperture.
        im : ndarray
            The target image.
        n_chunks : int, optional
            The number of chunks. The default is 8.
        profile_oversample : int, optional
            The oversampling factor of the profile. The default is 10.
        profile_smoothness : float, optional
            The smoothness of the profile. The default is 1e-2.
        num_sigma_clipping : float, optional
            The sigma clipping threshold. The default is 5..
        gain : float, optional
            The gain of CCD. The default is 1..
        ron : flaot, optional
            The readout noise of CCD. The default is 0.

        Returns
        -------
        dict
            # ----- combined extraction -----
            spec_extr=spec_extr,
            err_extr=err_extr,
            # ----- first extraction -----
            spec_extr1=spec_extr1,
            err_extr1=err_extr1,
            # ----- second extraction (after sigma-clipping) -----
            spec_extr2=spec_extr2,
            err_extr2=err_extr2,
            # ----- inconsistent pixels between 1&2, True if difference > 3sigma -----
            mask_extr=mask_extr,
            # ----- simple extraction -----
            spec_sum=ap_im.sum(axis=1),
            err_sum=ap_im_err.sum(axis=1),
            # ----- reconstructed profile -----
            prof_recon=prof_recon,
            # ----- reconstructed aperture -----
            ap_im=ap_im,
            ap_im_recon=ap_im_recon,
            ap_im_recon1=ap_im_recon1,
            ap_im_recon2=ap_im_recon2,
            # ----- aperture coordinates -----
            ap_im_xx=ap_im_xx,
            ap_im_xx_cor=ap_im_xx_cor,
            ap_im_yy=ap_im_yy,.

        """
        return extract_aperture(im, self.ap_center_interp[iap], n_chunks=n_chunks, ap_width=self.ap_width,
                                profile_oversample=profile_oversample, profile_smoothness=profile_smoothness,
                                num_sigma_clipping=num_sigma_clipping, gain=gain, ron=ron)


def test_aperture():
    from astropy.io import fits
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15})
    flat = fits.getdata("/Users/cham/projects/song/star_spec/20191105/night/ext/masterflat_20191105_slit5.fits")

    ap = Aperture.trace(flat, method="naive", polydeg=4, sigma=7, maxdev=7, ap_width=15)
    ap = Aperture.trace(flat, method="canny", polydeg=4, sigma=7, maxdev=7, ap_width=15)
    return ap


if __name__ == "__main__":
    test_aperture()

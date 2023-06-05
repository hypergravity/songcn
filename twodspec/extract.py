import numpy as np
from laspec.extern.interpolate import SmoothSpline
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
import joblib


# deprecated
# def rebin(x, y, xx):
#     bins_xx = np.hstack((1.5 * xx[0] - 0.5 * xx[1],
#                          xx[:-1] + 0.5 * np.diff(xx),
#                          1.5 * xx[-1] - 0.5 * xx[-2]))
#     return binned_statistic(x, y, statistic="mean", bins=bins_xx)[0]


# def get_image_coordinates(im):
#     im_xx, im_yy = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
#     return im_xx, im_yy
#
#
# def center_to_edges(centers):
#     return np.hstack((1.5 * centers[0] - 0.5 * centers[1],
#                       centers[:-1] + 0.5 * np.diff(centers),
#                       1.5 * centers[-1] - 0.5 * centers[-2]))


####################################
# aperture processing functions
####################################

def get_aperture_section(im, ap_center_interp, ap_width=15):
    """ get aperture section and coordinates
    
    Parameters
    ----------
    im : ndarray
        target image.
    ap_center_interp : ndarray (npix,)
        ap_center_interp.
    ap_width : float, optional
        ap_width. The default is 15.

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
    n_rows, n_cols = im.shape
    im_xx, im_yy = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

    # cut image to get this aperture region
    ap_center_interp_floor = np.floor(ap_center_interp).astype(int)
    # ap_center_interp_rmdr= ap_center_interp-ap_center_interp_floor

    # ap_im_nrows = n_rows  # not used
    ap_im_ncols = ap_width*2+2

    ap_im_xx = ap_center_interp_floor.reshape(-1, 1)+np.arange(-ap_width, ap_width+2)
    ap_im_yy = im_yy[:, :ap_im_ncols]

    ap_im_xx_flat = ap_im_xx.flatten()
    ap_im_yy_flat = ap_im_yy.flatten()
    ap_im_flat = np.zeros_like(ap_im_xx_flat, dtype=np.float)

    ind_valid = (ap_im_xx_flat >= 0) & (ap_im_xx_flat <= n_cols-1)
    ap_im_flat[ind_valid] = im[ap_im_yy_flat[ind_valid], ap_im_xx_flat[ind_valid]]
    ap_im = ap_im_flat.reshape(ap_im_xx.shape)

    ap_im_xx_cor = ap_im_xx - ap_center_interp.reshape(-1, 1)
    return ap_im, ap_im_xx, ap_im_yy, ap_im_xx_cor


def extract_profile_simple(ap_im, ap_im_xx_cor, ap_width, profile_oversample=10, profile_smoothness=1e-1):

    # 1. normalize image along x
    ap_im_norm = ap_im / np.sum(ap_im, axis=1).reshape(-1, 1)
    # 2. extract profile (quite good so far) for each chunk
    prof_x = np.arange(-ap_width - 1, ap_width + 1, 1 / profile_oversample)
    sp_prof_init = SmoothSpline(
        ap_im_xx_cor.flatten(),
        ap_im_norm.flatten(),
        p=profile_smoothness)
    prof_init = sp_prof_init(prof_x)
    prof_init[prof_init < 0] = 0

    return prof_x, prof_init


def extract_profile(ap_im, ap_im_xx_cor, profile_smoothness=1e-2, n_chunks=8,
                    ap_width=15., profile_oversample=10., ndev=4):
    """ extract profile from aperture image ap_im 
    
    Parameters
    ----------
    ap_im : ndarray
        The target image.
    ap_im_xx_cor : ndarray
        x offset.
    profile_smoothness : float, optional
        The profile smoothness, passed to SmoothSpline. The default is 1e-2.
    n_chunks : int, optional
        The number of chunks. The default is 8.
    ap_width : float, optional
        The aperture width. The default is 15..
    profile_oversample : int, optional
        The oversampling factor of the profile. The default is 10.
    ndev : float, optional
        The number of sigmas when clipping outliers. The default is 4.

    Returns
    -------
    profs_rebin : ndarray
        The normalized rebinned profile.

    """
    # 0. determine chunk length
    n_rows = ap_im.shape[0]
    chunk_len = np.int(n_rows / n_chunks)
    chunk_centers = chunk_len * (np.arange(n_chunks) + 0.5)
    chunk_start = chunk_len * np.arange(n_chunks)

    # 1. normalize image along x
    ap_im_norm = ap_im / np.sum(ap_im, axis=1).reshape(-1, 1)

    # 2. extract profile (quite good so far) for each chunk
    prof_out = np.zeros_like(ap_im_norm, dtype=bool)
    prof_x = np.arange(-ap_width - 1, ap_width + 1 + 1 / profile_oversample, 1 / profile_oversample)
    profs = []
    prof_xnode = np.arange(-ap_width - 1, ap_width + 1 + 1)
    nnode = len(prof_xnode)
    for istart in chunk_start:
        # first smooth and calculate residuals
        this_prof_res = SmoothSpline(
            ap_im_xx_cor[istart:istart + chunk_len].flatten(),
            ap_im_norm[istart:istart + chunk_len].flatten(),
            p=profile_smoothness)(ap_im_xx_cor[istart:istart + chunk_len]) - ap_im_norm[istart:istart + chunk_len]
        # outlier index
        this_prof_out = np.zeros_like(ap_im_norm[istart:istart + chunk_len], dtype=bool)
        # clipping each node
        for inode, this_prof_xnode in enumerate(prof_xnode):
            if inode == 0 or inode == nnode:
                this_prof_xnode_ind = np.abs(ap_im_xx_cor[istart:istart + chunk_len] - this_prof_xnode) < 1
                qs = np.percentile(this_prof_res[this_prof_xnode_ind], q=[25, 50, 75])
                this_prof_med = qs[1]
                this_prof_dev = (qs[2] - qs[0]) / 2
                this_prof_out |= this_prof_xnode_ind & (np.abs(this_prof_res - 0) < ndev * this_prof_dev)
            else:
                this_prof_xnode_ind = np.abs(ap_im_xx_cor[istart:istart + chunk_len] - this_prof_xnode) < 1
                qs = np.percentile(this_prof_res[this_prof_xnode_ind], q=[25, 50, 75])
                this_prof_med = qs[1]
                this_prof_dev = (qs[2] - qs[0]) / 2
                this_prof_out |= this_prof_xnode_ind & (np.abs(this_prof_res - this_prof_med) < ndev * this_prof_dev)
        # second smooth --> as initial profile
        prof_init = SmoothSpline(
            ap_im_xx_cor[istart:istart + chunk_len][this_prof_out].flatten(),
            ap_im_norm[istart:istart + chunk_len][this_prof_out].flatten(),
            p=profile_smoothness)(prof_x)
        # append result
        profs.append(prof_init)
        prof_out[istart:istart + chunk_len] = this_prof_out

    # head and tail
    profs.insert(0, profs[0])
    profs.append(profs[-1])
    profs = np.array(profs)
    profs_y = np.hstack((-1, chunk_centers, n_rows + 1))
    # force positive
    profs = np.where(profs <= 0, 0., profs)

    # 3. interpolate profiles along lambda
    # [assuming profile varies along dispersion axis]
    ap_y = np.arange(n_rows)
    profs_interp = np.array([
        interp1d(profs_y, profs[:, icol], kind="linear")(ap_y) \
        for icol in range(profs.shape[1])]).T

    # 4. interpolate profiles along x
    profs_rebin = np.array([interp1d(prof_x, profs_interp[i], kind="linear", bounds_error=False, fill_value=0)(ap_im_xx_cor[i]) for i in range(n_rows)])
    # normalize
    profs_rebin = profs_rebin / np.sum(profs_rebin, axis=1).reshape(-1, 1)

    return profs_rebin, prof_x, profs_interp, prof_out


def extract_from_profile(ap_im, prof_recon, var=None):
    """ given profile, extract spectrum by lsq fitting

    Parameters
    ----------
    ap_im : ndarray
        The target image.
    prof_recon : ndarray
        The reconstructed profile.
    var : None or ndarray, optional
        The variance of ap_im, (1, npix_section) or like ap_im. The default is None.

    Returns
    -------
    spec_extr1 : ndarray
        The extracted spectrum from fitting profile.

    """
    if var is not None:
        # variance weighted extraction
        spec_extr1 = np.nansum(ap_im * prof_recon / var, axis=1) / np.nansum(prof_recon ** 2. / var, axis=1)
    else:
        # even weighted extraction (prefered)
        spec_extr1 = np.nansum(ap_im * prof_recon, axis=1) / np.nansum(prof_recon ** 2., axis=1)

    return spec_extr1


def extract_aperture(im, ap_center_interp, n_chunks=8, ap_width=15,
                     profile_oversample=10, profile_smoothness=1e-2,
                     num_sigma_clipping=5., gain=1., ron=0):
    """ extract an aperture given the aperture center

    Parameters
    ----------
    im : ndarray
        The target image.
    ap_center_interp : ndarray
        The ap_center_interp.
    n_chunks : int, optional
        The number of chunks. The default is 8.
    ap_width : float, optional
        The width of aperture / pix. The default is 15.
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
    a dict consisting of many results.

    """

    # 1. get aperture section
    ap_im, ap_im_xx, ap_im_yy, ap_im_xx_cor = get_aperture_section(
        im, ap_center_interp, ap_width=ap_width)

    # set negative values to 0
    ap_im = np.where(ap_im > 0, ap_im, 0)

    # error image
    ap_im_errerr = ap_im / gain + ron ** 2. # lijiao
    ap_im_err = np.sqrt(ap_im_errerr) # lijiao

    # 2. extract profile (quite good so far) for each chunk
    prof_recon, prof_xoversample, prof_oversample, prof_out = extract_profile(
        ap_im, ap_im_xx_cor, profile_smoothness=profile_smoothness,
        n_chunks=n_chunks, ap_width=ap_width,
        profile_oversample=profile_oversample)

    # 3. extract using profile
    spec_extr1 = extract_from_profile(ap_im, prof_recon, var=None)
    errerr_extr1 = extract_from_profile(ap_im_errerr, prof_recon, var=None) # lijiao
    err_extr1 = np.sqrt(errerr_extr1) # lijiao

    # 4. 3-sigma clipping
    # reconstruct image
    ap_im_recon1 = prof_recon * spec_extr1.reshape(-1, 1)
    # residual
    ap_im_res = ap_im - ap_im_recon1
    # mask
    ap_im_mask = ap_im_res < ap_im_err * num_sigma_clipping
    prof_recon_masked = prof_recon * ap_im_mask

    # 5. re-extract using profile (robust but not always good)
    spec_extr2 = extract_from_profile(ap_im, prof_recon_masked, var=None)
    errrr_extr2 = extract_from_profile(ap_im_errrr, prof_recon_masked, var=None)
    err_extr2 = np.sqrt(errerr_extr2)

    # 6. combine extraction (robust)
    mask_extr = np.abs((spec_extr2 - spec_extr1) / err_extr2) > 3.
    spec_extr = np.where(mask_extr, spec_extr2, spec_extr1)
    err_extr = np.where(mask_extr, err_extr2, err_extr1)

    # reconstruct image
    ap_im_recon2 = prof_recon * spec_extr2.reshape(-1, 1)
    ap_im_recon = prof_recon * spec_extr.reshape(-1, 1)

    return dict(
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
        ap_im_yy=ap_im_yy,
        ap_im_out=prof_out,
        # ----- oversampled profile ------
        prof_xoversample=prof_xoversample,
        prof_oversample=prof_oversample,
    )


def extract_sum(ap_im, ap_im_xx_cor, ap_width=15):
    """ extract spectrum using simple sum

    Parameters
    ----------
    ap_im : ndarray
        The aperture image.
    ap_im_xx_cor : ndarray
        The x offsets.
    ap_width : float, optional
        The width of aperture. The default is 15.

    Returns
    -------
    ndarray
        simple extraction.

    """
    # determine fraction of pixels
    ap_im_frac = np.ones_like(ap_im)
    # fractional part
    ap_im_frac_l = ap_im_xx_cor + (.5 + ap_width)
    ap_im_frac_r = (.5 + ap_width) - ap_im_xx_cor
    ap_im_frac = np.where(ap_im_frac_l < 1, ap_im_frac_l, ap_im_frac)
    ap_im_frac = np.where(ap_im_frac_r < 1, ap_im_frac_r, ap_im_frac)
    # eliminate negative values
    ap_im_frac = np.where(ap_im_frac < 0, 0, ap_im_frac)

    # sum
    return np.nansum(ap_im_frac * ap_im, axis=1)


####################################
# extract all apertures
####################################

def extract_all(im, ap, n_chunks=8, profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=10, gain=1., ron=0, n_jobs=-1,
                verbose=False, backend="multiprocessing"):
    """ extract all apertures with both simple & profile extraction

    Parameters
    ----------
    im : ndarray
        The target image.
    ap : Aperture
        The Aperture instance.
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
    verbose :
        joblib verbose
    backend :
        joblib backend

    Returns
    -------
    dict
        a dict sconsisting of many results.

    """
    # extract all apertures in parallel
    rs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(joblib.delayed(extract_aperture)(
        im, ap.ap_center_interp[i], n_chunks=n_chunks, profile_oversample=profile_oversample,
        profile_smoothness=profile_smoothness, num_sigma_clipping=num_sigma_clipping, gain=gain, ron=ron)
                                                   for i in range(ap.nap))
    # reconstruct results
    result = dict(
        # combbined
        spec_extr=np.array([rs[i]["spec_extr"] for i in range(ap.nap)]),
        err_extr=np.array([rs[i]["err_extr"] for i in range(ap.nap)]),
        # profile extraction
        spec_extr1=np.array([rs[i]["spec_extr1"] for i in range(ap.nap)]),
        err_extr1=np.array([rs[i]["err_extr1"] for i in range(ap.nap)]),
        # profile extraction sigma-clipping
        spec_extr2=np.array([rs[i]["spec_extr2"] for i in range(ap.nap)]),
        err_extr2=np.array([rs[i]["err_extr2"] for i in range(ap.nap)]),
        # simple extraction
        spec_sum=np.array([rs[i]["spec_sum"] for i in range(ap.nap)]),
        err_sum=np.array([rs[i]["err_sum"] for i in range(ap.nap)]),
        # 1 for difference > 3 sigma
        mask_extr=np.array([rs[i]["mask_extr"] for i in range(ap.nap)]),
    )
    return result


####################################
# make normalized FLAT image
####################################

def local_filter1(x, kw=5, method="mean"):
    """ 1d mean/median filter, used to smooth blaze function """
    if method == "mean":
        f = np.mean
    elif method == "median":
        f = np.median
    else:
        raise(ValueError("bad value for method!"))

    xs = np.copy(x)
    for i in range(kw, np.int(len(x) - kw)):
        xs[i] = f(x[np.int(i - kw):np.int(i + kw + 1)])
    return xs


def make_normflat(im, ap, max_dqe=0.04, min_snr=20, smooth_blaze=5, n_chunks=8,
                  profile_oversample=10, profile_smoothness=1e-2, num_sigma_clipping=20, gain=1., ron=0,
                  n_jobs=-1, verbose=False, backend="multiprocessing"):
    """ normalize FLAT

    Parameters
    ----------
    im : ndarray
        The target image.
    ap : Aperture
        The aperture instance
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
        The number of processes launched. The default is -1.
    verbose:
        defaults to False
    backend:
        defaults to multiprocessing

    Returns
    -------
    blaze, im_norm : ndarray
        The blaze functions and sensitivity image.


    """
    im_recon = np.zeros_like(im)
    blaze = []

    # extract in parallel
    rs = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(joblib.delayed(extract_aperture)(
        im, ap.ap_center_interp[i], n_chunks=n_chunks, ap_width=ap.ap_width,
        profile_oversample=profile_oversample,
        profile_smoothness=profile_smoothness,
        num_sigma_clipping=num_sigma_clipping,
        gain=gain, ron=ron) for i in range(ap.nap))
    # gather results
    for i in range(ap.nap):
        # smooth blaze function (use spec_extr1???)
        this_blaze_smoothed1 = local_filter1(rs[i]["spec_extr1"], kw=np.int(smooth_blaze), method="median")
        this_blaze_smoothed2 = local_filter1(this_blaze_smoothed1, kw=np.int(smooth_blaze), method="mean")
        blaze.append(this_blaze_smoothed2)
        im_recon[rs[i]["ap_im_yy"], rs[i]["ap_im_xx"]] = rs[i]["prof_recon"] * this_blaze_smoothed2.reshape(-1, 1)
    blaze = np.array(blaze)
    # make norm image
    im_norm = im / im_recon

    # max deviation for quantum efficiency
    im_norm = np.where(np.abs(im_norm - 1) < max_dqe, im_norm, 1.)

    # for low SNR pixel, set 1
    im = np.where(im <= 0, 0, im)
    im_err = np.sqrt(im / gain + ron ** 2)
    im_snr = im / im_err
    im_norm = np.where(im_snr < min_snr, 1, im_norm)

    return blaze, im_norm

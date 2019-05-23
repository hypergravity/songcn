import numpy as np
from twodspec.extern.interpolate import SmoothSpline
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic


# deprecated
def rebin(x, y, xx):
    bins_xx = np.hstack((1.5 * xx[0] - 0.5 * xx[1],
                         xx[:-1] + 0.5 * np.diff(xx),
                         1.5 * xx[-1] - 0.5 * xx[-2]))
    return binned_statistic(x, y, statistic="mean", bins=bins_xx)[0]


def get_image_coordinates(im):
    im_xx, im_yy = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    return im_xx, im_yy


def center_to_edges(centers):
    return np.hstack((1.5 * centers[0] - 0.5 * centers[1],
                      centers[:-1] + 0.5 * np.diff(centers),
                      1.5 * centers[-1] - 0.5 * centers[-2]))


####################################
# aperture processing function
####################################

def get_aperture_section(im, ap_center_interp, ap_width=15):
    """ get aperture section and coordinates """
    n_rows, n_cols = im.shape
    im_xx, im_yy = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

    # cut image to get this aperture region
    ap_center_interp_floor = np.floor(ap_center_interp).astype(int)
    # ap_center_interp_rmdr= ap_center_interp-ap_center_interp_floor

    ap_im_nrows = n_rows
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


def extract_profile(ap_im, ap_im_xx_cor, profile_smoothness=1e-2, n_chunks=8,
                    ap_width=15., profile_oversample=10.):
    """ extract profile from aperture image ap_im """
    # 0. determine chunk length
    n_rows = ap_im.shape[0]
    chunk_len = np.int(n_rows / n_chunks)
    chunk_centers = chunk_len * (np.arange(n_chunks) + 0.5)
    chunk_start = chunk_len * np.arange(n_chunks)

    # 1. normalize image along x
    ap_im_norm = ap_im / np.sum(ap_im, axis=1).reshape(-1, 1)

    # 2. extract profile (quite good so far) for each chunk
    prof_x = np.arange(-ap_width - 1, ap_width + 1, 1 / profile_oversample)
    profs = []
    for istart in chunk_start:
        # smooth the profile
        sp_prof_init = SmoothSpline(
            ap_im_xx_cor[istart:istart + chunk_len].flatten(),
            ap_im_norm[istart:istart + chunk_len].flatten(),
            p=profile_smoothness)
        prof_init = sp_prof_init(prof_x)
        profs.append(prof_init)
    # head and tail
    profs.insert(0, profs[0])
    profs.append(profs[-1])
    profs = np.array(profs)
    profs_y = np.hstack((-1, chunk_centers, n_rows + 1))

    # 3. interpolate profiles along lambda
    # [assuming profile varies along dispersion axis]
    ap_y = np.arange(n_rows)
    profs_interp = np.array([
        interp1d(profs_y, profs[:, icol], kind="linear")(ap_y) \
        for icol in range(profs.shape[1])]).T

    # 4. interpolate profiles along x
    profs_rebin = np.array([interp1d(prof_x, profs_interp[i], kind="linear", bounds_error=False, fill_value=0)(ap_im_xx_cor[i]) for i in range(n_rows)])

    # force positive
    profs_rebin = np.where(profs_rebin <= 0, 0., profs_rebin)
    # normalize
    profs_rebin /= np.sum(profs_rebin, axis=1).reshape(-1, 1)

    return profs_rebin


def extract_from_profile(ap_im, prof_recon, var=None):
    """ given profile, extract spectrum """
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
    """ extract an aperture given the trace """

    # 1. get aperture section
    ap_im, ap_im_xx, ap_im_yy, ap_im_xx_cor = get_aperture_section(
        im, ap_center_interp, ap_width=ap_width)

    # set negative values to 0
    ap_im = np.where(ap_im > 0, ap_im, 0)

    # error image
    ap_im_err = np.sqrt(ap_im / gain + ron ** 2.)

    # 2. extract profile (quite good so far) for each chunk
    prof_recon = extract_profile(
        ap_im, ap_im_xx_cor, profile_smoothness=profile_smoothness,
        n_chunks=n_chunks, ap_width=ap_width,
        profile_oversample=profile_oversample)

    # 3. extract using profile
    spec_extr1 = extract_from_profile(ap_im, prof_recon, var=None)
    err_extr1 = extract_from_profile(ap_im_err, prof_recon, var=None)

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
    err_extr2 = extract_from_profile(ap_im_err, prof_recon_masked, var=None)

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
        # ----- second extraction (after sigma-clipping -----
        spec_extr2=spec_extr2,
        err_extr2=err_extr2,
        # ----- inconsistent pixels between 1&2 -----
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
    )


def extract_sum(ap_im, ap_im_xx_cor, ap_width=15):
    """ extract spectrum using simple sum """
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

def extract_all(im, ap, n_chunks=8, ap_width=15,
                profile_oversample=10, profile_smoothness=1e-2,
                num_sigma_clipping=10, gain=1., ron=0):
    """ extract all apertures """
    n_ap = ap.n_ap

    spec_extr = []
    err_extr = []

    spec_extr1 = []
    err_extr1 = []

    spec_extr2 = []
    err_extr2 = []

    spec_sum = []
    err_sum = []

    mask_extr = []

    for i in range(n_ap):
        r = extract_aperture(im, ap.ap_center_interp[i],
                             n_chunks=n_chunks, ap_width=ap_width,
                             profile_oversample=profile_oversample,
                             profile_smoothness=profile_smoothness,
                             num_sigma_clipping=num_sigma_clipping,
                             gain=gain, ron=ron)
        spec_extr.append(r["spec_extr"])
        err_extr.append(r["err_extr"])
        spec_extr1.append(r["spec_extr1"])
        err_extr1.append(r["err_extr1"])
        spec_extr2.append(r["spec_extr2"])
        err_extr2.append(r["err_extr2"])
        spec_sum.append(r["spec_sum"])
        err_sum.append(r["err_sum"])
        mask_extr.append(r["mask_extr"])

    return dict(
        spec_extr=np.array(spec_extr),
        err_extr=np.array(err_extr),
        spec_extr1=np.array(spec_extr1),
        err_extr1=np.array(err_extr1),
        spec_extr2=np.array(spec_extr2),
        err_extr2=np.array(err_extr2),
        spec_sum=np.array(spec_sum),
        err_sum=np.array(err_sum),
        mask_extr=np.array(mask_extr),
    )


####################################
# make normalized FLAT image
####################################

def local_filter1(x, kw=5, method="mean"):
    """ 1d mean/median filter """
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


def make_normflat(im, ap, max_dqe=0.04, min_snr=20, smooth_blaze=5,
                  n_chunks=8, ap_width=15,
                  profile_oversample=10, profile_smoothness=1e-2,
                  num_sigma_clipping=20, gain=1., ron=0):
    """ make normalized FLAT image """
    n_ap = ap.n_ap
    im_recon = np.zeros_like(im)
    blaze = []
    for i in range(n_ap):
        r = extract_aperture(im, ap.ap_center_interp[i],
                             n_chunks=n_chunks, ap_width=ap_width,
                             profile_oversample=profile_oversample,
                             profile_smoothness=profile_smoothness,
                             num_sigma_clipping=num_sigma_clipping,
                             gain=gain, ron=ron)
        # smooth blaze function
        this_blaze_smoothed1 = local_filter1(r["spec_extr1"], kw=np.int(smooth_blaze),
                                             method="median")
        this_blaze_smoothed2 = local_filter1(this_blaze_smoothed1, kw=np.int(smooth_blaze),
                                             method="mean")
        blaze.append(this_blaze_smoothed2)
        im_recon[r["ap_im_yy"], r["ap_im_xx"]] = r["prof_recon"] * this_blaze_smoothed2.reshape(-1, 1)
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
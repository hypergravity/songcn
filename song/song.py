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
- Fri Feb 24 16:00:00 2017

Modifications
-------------
-

Aims
----
- Song class

"""

import os
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from joblib import Parallel, delayed, dump, load
from matplotlib import pyplot as plt

from twodspec import extract
from twodspec.aperture import Aperture
from twodspec.ccd import CCD
from twodspec.deprecated import stella  # should be removed in future
from . import __path__
from . import thar
from .utils import scan_files


class SongSlit(object):
    """ Song Slit class """
    slit = -1
    flat = None
    aprec = None
    background = None
    blaze = None
    norm = None
    thar_list = {}
    star_list = {}

    def extract_thar(self, img_thar):
        # return wave to thar_list
        pass

    def extract_star(self, img_star):
        pass


class Song(Table):
    """ represent SONG configuration """

    # define all image types, used for classification
    ALL_IMGTYPE = [
        "BIAS", "FLAT", "FLATI2", "THAR", "THARI2", "STAR", "STARI2", "TEST"]

    # configuration for:
    kwargs_scan = dict(n_jobs=2, verbose=True)
    # 1. reading CCD images
    kwargs_read = dict(hdu=0, gain=1., ron=0., unit='adu', trim=None, rot90=0)
    # 2. combine images
    kwargs_combine = dict(method="median")
    # 3. scattered-light background
    kwargs_background_flat = dict(q=(30, 1), kernel_size=(17, 17), sigma=(11, 7))
    kwargs_background_star = dict(q=(45, 45), kernel_size=(17, 17), sigma=(11, 7))

    kwargs_normflat = dict(max_dqe=0.04, min_snr=20, smooth_blaze=5,
                           n_chunks=8, ap_width=15, profile_oversample=10,
                           profile_smoothness=1e-2, num_sigma_clipping=20,
                           gain=1., ron=0)  # root directory
    kwargs_extract = dict(n_chunks=8, ap_width=15, profile_oversample=10,
                          profile_smoothness=1e-2, num_sigma_clipping=20,
                          gain=1., ron=0)
    # dirpath = ""
    # data & work directory
    dir_data = ""
    dir_work = ""

    # unique slits
    unique_slits = []

    # master BIAS
    master_bias = None
    master_ron = None

    # master FLAT for each slit
    master_flats = dict()

    # THAR/STAR/FLATI2/TEST will be processed separately
    thar_list = dict()
    thari2_list = dict()
    star_list = dict()
    stari2_list = dict()
    flati2_list = dict()
    test_list = dict()

    # load ThAr template in Song class
    print("@SONG: [ThAr] loading ThAr template ...")
    thar_line_list = np.loadtxt(__path__[0] + "/calibration/thar.dat")
    thar_solution_temp = (
        fits.getdata(__path__[0] + "/calibration/thar_template.fits", ext=1),
        fits.getdata(__path__[0] + "/calibration/thar_template.fits", ext=2),
        fits.getdata(__path__[0] + "/calibration/thar_template.fits", ext=3))

    def __init__(self, *args, **kwargs):
        super(Song, self).__init__(*args, **kwargs)

    def new(self):
        """ initiate a new blank Song object """
        s = self._init_from_dir(self.dir_data, self.dir_work, self.date,
                                **self.kwargs_scan)
        s.kwargs_scan = self.kwargs_scan
        s.kwargs_read = self.kwargs_read
        s.kwargs_combine = self.kwargs_combine
        s.kwargs_background = self.kwargs_background

        return s

    @staticmethod
    def _init_from_dir(dir_data, dir_work, date="", n_jobs=-1, verbose=True):
        """ initiate from a directory path

        Parameters
        ----------
        dir_data: string
            data directory
        dir_work: string
            work directory
        date:
            optional
        n_jobs=-1, verbose=True:
            joblib.Parallel arguments

        Returns
        -------
        Song object
        """

        try:
            assert os.path.exists(dir_data)
        except AssertionError as ae:
            print("@SONG: {} doesn't exist!")
            raise (ae)

        # if dir_work does not exist, make it
        if not os.path.exists(dir_work):
            os.mkdir(dir_work)

        print("@SONG: scanning files ...")
        s = Song(scan_files(dir_data, n_jobs=n_jobs, verbose=verbose))

        s.dir_data = dir_data
        s.dir_work = dir_work
        s.date = date

        s.kwargs_scan["n_jobs"] = n_jobs
        s.kwargs_scan["verbose"] = verbose

        # get unique slits
        s.unique_slits = list(s.unique_config(cfgkeys=("SLIT")))

        return s

    def pipeline_bias(self):
        # BIAS: up to 120 frame
        self.master_bias, self.master_ron = self.ezmaster(
            {"IMAGETYP": "BIAS"}, n_select=120, method_select='all',
            method_combine='median', std=True)
        """ write to disk """
        self.master_bias_fp = "{}/masterbias_{}.fits".format(self.dir_work, self.date)
        self.master_bias.write(self.master_bias_fp, overwrite=True)
        print("@SONG: combined bias written to *{}*".format(self.master_bias_fp))
        return

    def pipeline_flat(self, slits="all"):

        if slits == "all":
            slits = self.unique_slits
        else:
            slits = [slits]

        for slit in slits:

            print()
            print()
            print("@SONG: processing FLAT for SLIT = ", slit)

            # 1.combine FLAT
            print("@SONG: combining FLAT ...")
            flat = self.ezmaster({"SLIT": slit, "IMAGETYP": "FLAT"},
                                 n_select=120, method_select='all',
                                 method_combine='median')
            # 2.substract bias
            print("@SONG: substracting BIAS ...")
            flat = flat - self.master_bias

            # 3.write to disk
            flat_fp = self.dir_work + "/masterflat_{}_slit{}.fits".format(
                self.date, slit)
            print("@SONG: writing to {} ...".format(flat_fp))
            flat.write(flat_fp, overwrite=True)

            # 4.choose best sigma (at least 50 orders, else raise error)
            sigmas = np.arange(1, 10, 0.3)
            n_ap = Parallel(n_jobs=-1, verbose=0)(
                delayed(_try_trace_apertures)(np.array(flat), sigma_)
                for sigma_ in sigmas)
            n_ap = np.array(n_ap)
            try:
                assert np.sum(n_ap >= 50) > 0 and np.max(n_ap) < 52
            except AssertionError as ae:
                print("@SONG: the n_ap: ", np.unique(n_ap))
                raise (ae)
            sigma_best = sigmas[np.argmax(n_ap)]
            print("@SONG: best sigma: {:0.2f}".format(sigma_best))

            # 4'. trace apertures
            print("@SONG: tracing apertures ...")
            ap = Aperture.trace(flat, method="canny", sigma=sigma_best)
            ap.polyfit(4)

            # figure of apertures
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(np.log10(flat.data))
            plt.plot(ap.ap_center.T, ap.x, 'w')
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.title("SLIT {}".format(slit))
            fig.tight_layout()

            # figure of horizontal slice
            fig = plt.figure(figsize=(12, 6))
            plt.plot(flat[1024])
            plt.vlines(ap.ap_center[:, 1023], 0, 10000)
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.title("SLIT {} | Slice of row 1024".format(slit))
            fig.tight_layout()

            # 5.background
            print("@SONG: computing background ...")
            bg = CCD(ap.background(flat, **self.kwargs_background_flat))
            bg_fp = self.dir_work + "/masterflat_{}_slit{}_bg.fits".format(self.date, slit)
            bg.write(bg_fp, overwrite=True)
            flat = flat - bg

            print("@SONG: computing blaze function & sentitivity ...")
            blz, norm = extract.make_normflat(np.array(flat), ap, **self.kwargs_normflat)
            blz_fp = self.dir_work + "/masterblz_{}_slit{}.fits".format(self.date, slit)
            norm_fp = self.dir_work + "/masternorm_{}_slit{}.fits".format(self.date, slit)
            blz = CCD(blz)
            norm = CCD(norm)
            print("@SONG: writing blaze function to {}...".format(blz_fp))
            blz.write(blz_fp, overwrite=True)
            print("@SONG: writing sentivity to {}...".format(norm_fp))
            norm.write(norm_fp, overwrite=True)

            self.master_flats[slit] = dict(
                # slit
                slit=slit,          # slit number
                # flat
                flat=flat,          # flat-bias
                flat_fp=flat_fp,    # flat fp
                # sigma
                sigma_best=sigma_best,
                # apertures
                ap=ap,
                # background
                bg=bg,
                bg_fp=bg_fp,
                # blaze function
                blz=blz,
                blz_fp=blz_fp,
                # sentivity
                norm=norm,
                norm_fp=norm_fp,
            )
            print("@SONG: finishing processing SLIT {}".format(slit))
        return

    def pipeline_thar(self, poly_order=(5, 10)):

        # subtable of ThAr images
        ind_thar = self["IMAGETYP"] == "THAR"
        print("@SONG: in total {} ThAr images found!".format(np.sum(ind_thar)))
        sthar = self["fps", "SLIT", "FILE", "MJD-MID"][ind_thar]
        print(sthar)

        # for each ThAr image
        for i_thar in range(len(sthar)):

            # 0.check ThAr existence
            file = sthar["FILE"][i_thar]
            if file in list(self.thar_list.keys()):
                print("@SONG: skipping {} !".format(file))
                continue
            else:
                # process this ThAr image
                print("@SONG: processing {} ...".format(file))

            # 1.read ThAr
            slit = sthar["SLIT"][i_thar]
            mjdmid = sthar["MJD-MID"][i_thar]
            thar_fp = sthar["fps"][i_thar]
            thar_fn = os.path.basename(thar_fp)
            im_thar = self.read(thar_fp)
            im_meta = im_thar.meta
            im_thar = im_thar.data

            # 2.sensitivity correction (de-norm)
            im_thar_denm = (im_thar - self.master_bias) / self.master_flats[slit]["norm"]
            #im_thar_denm_err = np.sqrt(np.abs(im_thar_denm)) / self.master_flats[slit]["norm"]

            # 3.extract ThAr spectrum
            ap = self.master_flats[slit]["ap"]
            rextr = extract.extract_all(im_thar_denm, ap, **self.kwargs_extract)
            thar_sp = rextr["spec_sum"]
            thar_err = rextr["err_sum"]
            """ extract all apertures """

            thar1d_simple = np.copy(np.flipud(thar_sp))

            # 4.calibration
            calibration_results = thar.calibrate(
                thar1d_simple, self.thar_solution_temp, self.thar_line_list,
                poly_order=(5, 10), slit=slit)
            sgrid_fitted_wave = calibration_results[0]
            wave_final = np.copy(np.flipud(sgrid_fitted_wave))

            # save figures
            fig = calibration_results[2]
            fig.savefig(self.dir_work + "thar{}_{:s}".format(
                slit, thar_fn.replace(".fits", "_diagnostics.svg")))
            plt.close(fig)
            fig = calibration_results[3]
            fig.savefig(self.dir_work + "thar{}_{:s}".format(
                slit, thar_fn.replace(".fits", "_used_lines.svg")))
            plt.close(fig)

            # 5.ThAr results: [wave] [sp] [sp_err]
            exheader = OrderedDict([
                ("---PC---", "---PROCESSING---"),
                ("author", "Bo Zhang"),
                ("court", "home"),])
            thar_header_ = OrderedDict([
                ("--HDUS--", "---HDUs---"),
                ("HDU0", "wavelength solution"),
                ("HDU1", "1D ThAr spectrum"),
                ("HDU2", "1D ThAr spectrum error")])
            im_meta.update(exheader)
            im_meta.update(thar_header_)
            h0 = fits.hdu.PrimaryHDU(wave_final, fits.Header(im_meta))  # ThAr wavelength
            # h1 = fits.hdu.ImageHDU(wave_final)  # ThAr wavelength
            h1 = fits.hdu.ImageHDU(thar_sp)  # ThAr spectrum
            h2 = fits.hdu.ImageHDU(thar_err)  # ThAr error spectrum
            hl = fits.HDUList([h0, h1, h2])
            hl.writeto(self.dir_work + "thar{}_{:s}".format(slit, thar_fn),
                       overwrite=True)
            # figure(); plot(wave_final.T, thar_sp.T)

            # record ThAr to self.thar_list
            self.thar_list[thar_fn] = dict(filename=thar_fn,
                                           mjdmid=mjdmid,
                                           slit=slit,
                                           wave_final=wave_final,
                                           thar1d_simple=thar1d_simple)

    def pipeline_star(self, key="STAR", n_jobs=-1, verbose=10):

        # subtable of ThAr images
        ind_star = self["IMAGETYP"] == key
        sub_star = np.where(ind_star)[0]
        print("@SONG: in total {} {} images found!".format(np.sum(ind_star), key))
        sstar = self["fps", "SLIT", "FILE", "MJD-MID"][ind_star]
        print(sstar)

        # prepare calibration data
        if key == "STAR":
            star_list = self.star_list
        elif key == "STARI2":
            star_list = self.stari2_list
        elif key == "FLATI2":
            star_list = self.flati2_list
        else:
            raise(AssertionError("@SONG: key *{}* is not valid!".format(key)))

        # skip processed files
        ind_unprocessed = np.zeros((len(sstar),), dtype=bool)
        for i_star in range(len(sstar)):
            # check STAR existence
            file = sstar["FILE"][i_star]
            if file not in list(star_list):
                # process this STAR image
                ind_unprocessed[i_star] = True
        print("==========================================================")
        print("@SONG: {} files will be skipped!".format(
            np.sum(np.logical_not(ind_unprocessed))))
        print("==========================================================")
        print(sstar[np.logical_not(ind_unprocessed)])
        print("==========================================================")
        print("")
        print("")
        print("==========================================================")
        print("@SONG: {} files will be processed!".format(
            np.sum(ind_unprocessed)))
        print("==========================================================")
        print(sstar[ind_unprocessed])
        print("==========================================================")
        print("")
        print("")
        sstar = Table(sstar[ind_unprocessed])

        r = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self._pipeline_a_star)(i_star) for i_star in sub_star[ind_unprocessed])

        if key == "STAR":
            self.star_list.extend(r)
        elif key == "STARI2":
            self.stari2_list.extend(r)
        elif key == "FLATI2":
            self.flati2_list.extend(r)

        # # record results
        # for i_star in range(len(sstar)):
        #     star_list[sstar["FILE"][i_star]] = r[i_star]

        # if key == "STAR":
        #     self.star_list = star_list
        # elif key == "STARI2":
        #     self.stari2_list = star_list
        # elif key == "FLATI2":
        #     self.flati2_list = star_list

        return

    def select(self, cond_dict=None, method="all", n_select=10,
               returns=("fps"), verbose=False):
        """ select some images from list

        Parameters
        ----------
        cond_dict: dict
            the dict of colname:value pairs
        method: string, {"all", "random", "top", "bottom"}
            the method adopted
        n_images:
            the number of images that will be selected
            if n_images is larger than the number of images matched conditions,
            then n_images is forced to be n_matched
        returns:
            the column name(s) of the column that will be returned
            if returns == 'sub', return the subs of selected images
        verbose:
            if True, print result

        Returns
        -------
        the Song instance

        Examples
        --------
        >>> s.list_image({"IMAGETYP":"STAR"}, returns=["OBJECT"])
        >>> s.select({"IMAGETYP":"THAR", "SLIT":6}, method="all", n_select=200,
        >>>          returns="sub", verbose=False)

        """

        # determine the matched images
        ind_match = np.ones((len(self),), dtype=bool)
        if cond_dict is None or len(cond_dict) < 1:
            print("@SONG: no condition is specified!")
        for k, v in cond_dict.items():
            ind_match = np.logical_and(ind_match, self[k] == v)

        # if no image found
        n_matched = np.sum(ind_match)
        if n_matched < 1:
            print("@SONG: no images matched!")
            return None

        sub_match = np.where(ind_match)[0]
        # determine the number of images to select
        n_return = np.min([n_matched, n_select])

        if verbose:
            print("@SONG: conditions are ", cond_dict)
            print("@SONG: {0} matched & {1} selected & {2} will be returned"
                  "".format(n_matched, n_select, n_return))

        # select according to method
        assert method in {"all", "random", "top", "bottom"}
        sub_rand = np.arange(0, n_matched, dtype=int)
        if method is "all":
            n_return = n_matched
        elif method is "random":
            np.random.shuffle(sub_rand)
            sub_rand = sub_rand[:n_return]
        elif method is "top":
            sub_rand = sub_rand[:n_return]
        elif method is "bottom":
            sub_rand = sub_rand[-n_return:]
        sub_return = sub_match[sub_rand]

        # constructing result to be returned
        if returns is "sub":
            result = sub_return
        else:
            result = self[returns][sub_return]

        # verbose
        if verbose:
            print("@SONG: these are all images selected")
            # here result is a Table
            print(result.__repr__())

        return result

    # #################################### #
    # simplified methods to select subsets
    # currently, just use select() method
    # #################################### #

    def ezselect_rand(self, cond_dict, n_select=10, returns="sub",
                      verbose=False):
        return self.select(cond_dict=cond_dict, returns=returns,
                           method="random", n_select=n_select, verbose=verbose)

    def ezselect_all(self, cond_dict, n_select=10, returns="sub",
                     verbose=False):
        return self.select(cond_dict=cond_dict, returns=returns,
                           method="all", n_select=n_select, verbose=verbose)

    # TODO: this method will either be updated/deleted
    def list_image(self, imagetp="FLAT", kwds=None, max_print=None):
        list_image(self, imagetp=imagetp, return_col=None, kwds=kwds,
                   max_print=max_print)
        return

    # #################################### #
    # methods to summarize data
    # #################################### #
    def unique_config(self, cfgkeys=("SLIT", "IMAGETYP")):

        result = np.asarray(np.unique(self[cfgkeys]))
        print("@SONG: {0} unique config found!".format(len(result)),
              list(result))
        return result

    def describe(self, cfgkeys=("SLIT", "IMAGETYP")):
        """

        Parameters
        ----------
        cfgkeys: tuple
            a pair of keys, default is ("SLIT", "IMAGETYP")

        Returns
        -------
        summary in Table format

        """
        # initialize result Table
        col0 = [Column(np.unique(self[cfgkeys[0]]), cfgkeys[0])]
        cols = [Column(np.zeros_like(col0[0], dtype=int), key2val) for key2val
                in np.unique(self[cfgkeys[1]])]
        col0.extend(cols)
        result = Table(col0)

        # do statistics & assign to result Table
        unique_result = np.unique(self[cfgkeys], return_counts=True)
        for keyvals_unique, count in zip(*unique_result):
            result[keyvals_unique[1]][
                result[cfgkeys[0]] == keyvals_unique[0]] = count

        return result

    # TODO: to add more info in summary
    @property
    def summary(self, colname_imagetype="IMAGETYP", return_data=False):
        """

        Parameters
        ----------
        colname_imagetype: string
            the keyword name for image type, default is "IMAGETYP"
        return_data: bool
            if True, return the summary data

        Returns
        -------
        unique images

        """

        u, uind, uinv, ucts = np.unique(self[colname_imagetype],
                                        return_counts=True, return_index=True,
                                        return_inverse=True)
        # print summary information
        print("=====================================================")
        print("[SUMMARY] {:s}".format(self.dirpath))
        print("=====================================================")
        for i in range(len(u)):
            print("{:10s} {:d}".format(u[i], ucts[i]))
        print("=====================================================")
        self.describe().pprint()
        print("=====================================================")

        # return results
        if return_data:
            return u, uind, uinv, ucts

    def read(self, fp):
        return CCD.read(fp, **self.kwargs_read)

    def reads(self, fps, method="median", std=False):
        return CCD.reads(fps, method=method, std=std, **self.kwargs_read)

    def ezmaster(self, cond_dict={"IMAGETYP":"BIAS"}, n_select=10,
                 method_select="top", method_combine="median", std=False):
        """

        Parameters
        ----------
        imgtype: string
            {"BIAS", "FLAT", "FLATI2", "THAR", "THARI2",
             "STAR", "STARI2", "TEST"}
        n_select: int
            number of images will be selected
        method_select:
            scheme of selection
        method_combine:
            method of combining

        Returns
        -------
        combined image

        """

        assert method_select in {"random", "top", "bottom", "all"}

        # if any cond_dict key does not exist in song
        try:
            for k in cond_dict.keys():
                assert k in self.colnames
        except:
            print("@SONG: key not found: {0}".format(k))
            raise(ValueError())

        # find fps of matched images
        fps = self.select(cond_dict, method=method_select, n_select=n_select)
        print("@SONG: *ezmaster* working on ", fps)

        # combine all selected images
        return self.reads(fps, method=method_combine, std=std)

    # #################################### #
    # save & dump method
    # #################################### #
    def dump(self, fp):
        print("@SONG: save to {0} ...".format(fp))
        dump(self, fp)
        return

    def draw(self, save=None, figsize=(20, 10), return_fig=False):
        """ a description of observation """

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ind = self["IMAGETYP"] == "BIAS"
        ax.plot(self["MJD-MID"][ind], self["SLIT"][ind], "o", mfc="none", mec='g',
                ms=20, label="BIAS")

        ind = self["IMAGETYP"] == "FLAT"
        ax.plot(self["MJD-MID"][ind], self["SLIT"][ind], "^", mfc="none", mec='r',
                ms=20, label="FLAT")

        ind = self["IMAGETYP"] == "FLATI2"
        ax.plot(self["MJD-MID"][ind], self["SLIT"][ind], "^", mfc="none", mec='b',
                ms=20, label="FLATI2")

        ind = self["IMAGETYP"] == "THAR"
        ax.plot(self["MJD-MID"][ind], self["SLIT"][ind], "D", mfc="none", mec='c',
                ms=20, mew=2, label="THAR")

        ind = self["IMAGETYP"] == "STAR"
        ax.plot(self["MJD-MID"][ind], self["SLIT"][ind], "s", mfc="none", mec='m',
                ms=20, label="STAR")

        ind = self["IMAGETYP"] == "STARI2"
        ax.plot(self["MJD-MID"][ind], self["SLIT"][ind], "s", mfc="none", mec='y',
                ms=20, label="STARI2")

        ind = self["IMAGETYP"] == "TEST"
        ax.plot(self["MJD-MID"][ind], self["SLIT"][ind], "s", mfc="none", mec='gray',
                ms=20, label="TEST")

        # sub table for texting
        tsub = self["MJD-MID", "SLIT", "IMAGETYP", "OBJECT"]
        tsub.sort("MJD-MID")
        trep = find_all_repeats(tsub)

        for i in range(len(trep)):
            if trep['label'][i] not in \
                    ["BIAS", "FLAT", "FLATI2", "THAR", "THARI2"]:
                fontcolor = "r"
            else:
                fontcolor = (0, 0, 0, .8)

            if np.mod(i, 2) == 0:
                tick_height = 0.3
                text_kwargs = dict(horizontalalignment='left',
                                   verticalalignment='bottom', rotation=50,
                                   fontsize=12, color=fontcolor)
            else:
                tick_height = -0.3
                text_kwargs = dict(horizontalalignment='left',
                                   verticalalignment='top', rotation=-50,
                                   fontsize=12, color=fontcolor)

            if trep['c'][i] > 1:
                tick_x = [trep["mjd1"][i], trep["mjd1"][i], trep["mjd2"][i],
                          trep["mjd2"][i]]
                tick_y = [trep["slit"][i], trep["slit"][i] + tick_height,
                          trep["slit"][i] + tick_height, trep["slit"][i]]
                ax.plot(tick_x, tick_y, 'k')
                ax.text(trep["mjd1"][i] * .5 + trep["mjd2"][i] * .5,
                        trep["slit"][i] + tick_height,
                        trep['label'][i] + " x {}".format(trep['c'][i]),
                        **text_kwargs)
            else:
                tick_x = [trep["mjd1"][i], trep["mjd1"][i]]
                tick_y = [trep["slit"][i], trep["slit"][i] + tick_height]
                ax.plot(tick_x, tick_y, 'k')
                ax.text(trep["mjd1"][i], trep["slit"][i] + tick_height,
                        trep['label'][i] + " x {}".format(trep['c'][i]),
                        **text_kwargs)

        ax.legend(loc=3)
        ax.set_xlabel("MJD-MID")
        ax.set_ylabel("#SLIT")
        ax.set_yticks(np.arange(9) + 1)
        ax.set_ylim(-.5, 10.5)

        mjd_min = np.floor(tsub['MJD-MID'].min())
        mjd_max = np.ceil(tsub['MJD-MID'].max())
        nmjd = np.int(np.round(mjd_max - mjd_min))
        ax.set_xlim(mjd_min, mjd_max)
        ax.grid(True, axis="y", ydata=np.arange(11))

        xlim1 = ax.get_xlim()
        ax2 = ax.twiny()
        ax2.set_xlabel("BJ Time")
        ax2.set_xlim(xlim1)
        ax2.set_xticks(mjd_min + np.linspace(0, nmjd, 24 * nmjd + 1))
        ax2.set_xticklabels(np.hstack((np.roll(np.arange(24), 12).reshape(
            -1, 1).repeat(nmjd, axis=1).T.flatten(), 12)))
        ax2.grid(True, axis='x', linestyle='--')

        ax.set_xlim(tsub['MJD-MID'].min() - .1, tsub['MJD-MID'].max() + .1)
        ax2.set_xlim(tsub['MJD-MID'].min() - .1, tsub['MJD-MID'].max() + .1)
        fig.tight_layout()

        if save is not None:
            fig.savefig(save)
        if return_fig:
            return fig
        else:
            plt.close(fig)
            return

    def _pipeline_a_star(self, i_star):

        # read STAR
        file = self["FILE"][i_star]
        slit = self['SLIT'][i_star]
        mjdmid = self['MJD-MID'][i_star]
        star_fp = self['fps'][i_star]
        star_fn = os.path.basename(star_fp)
        n_ap = self.master_flats[slit]["ap"].n_ap
        im_star = self.read(star_fp)
        im_meta = im_star.meta
        im_star = im_star.data

        # find ThAr
        thar_keys = list(self.thar_list.keys())
        # check slit
        ind_good = np.zeros_like(thar_keys, dtype=bool)
        for i in range(len(thar_keys)):
            if self.thar_list[thar_keys[i]]["slit"] == slit:
                ind_good[i] = True
        thar_keys = np.array(thar_keys)[ind_good]

        thar_mjdmid = np.array([self.thar_list[_]["mjdmid"] for _ in thar_keys])
        # ThAr before STAR
        sub_bfr = np.where(mjdmid > thar_mjdmid)[0]
        if len(sub_bfr)>0:
            ind_bfr = np.argmax(thar_mjdmid[sub_bfr])
            ind_tharbfr = sub_bfr[ind_bfr]
            print("@SONG: mjdmid: ", mjdmid)
            for i in range(len(thar_keys)):
                print("@SONG: thar: ", thar_keys[i],
                      " mjd: ", self.thar_list[thar_keys[i]]["mjdmid"],
                      " slit: ", self.thar_list[thar_keys[i]]["slit"])
            tharbfr = self.thar_list[thar_keys[ind_tharbfr]]["wave_final"]
            tharbfr_file = thar_keys[ind_tharbfr]
            print("@SONG: STAR[{}] THAR0[{}]".format(file, tharbfr_file))
        else:
            tharbfr = np.zeros((n_ap, 2048), dtype=float)
            tharbfr_file = ""
            print("@SONG: Couldn't find ThAr before observation!")
        # ThAr after STAR
        sub_aft = np.where(mjdmid < thar_mjdmid)[0]
        if len(sub_aft) == 0:
            tharaft = np.zeros_like(tharbfr)
            tharaft_file = ""
            print("@SONG: Couldn't find ThAr after observation!")
        else:
            ind_aft = np.argmin(thar_mjdmid[sub_aft])
            ind_tharaft = sub_aft[ind_aft]
            tharaft = self.thar_list[thar_keys[ind_tharaft]]["wave_final"]
            tharaft_file = thar_keys[ind_tharaft]
            print("@SONG: STAR[{}] THAR1[{}]".format(file, tharaft_file))

        # sensitivity correction
        im_star_denm = (im_star - self.master_bias) / self.master_flats[slit]['norm']
        #im_star_denm_err = np.sqrt(np.abs(im_star_denm)) / flats[slit]['norm']

        # scattered light substraction
        ap = self.master_flats[slit]['ap']
        bg = ap.background(im_star_denm, **self.kwargs_background_star)

        # extract STAR
        rextr = extract.extract_all(im_star_denm-bg, self.master_flats[slit]["ap"],
                                    **self.kwargs_extract)
        star_sp = rextr["spec_sum"]
        star_err = rextr["err_sum"]

        star_optsp = rextr["spec_extr1"]
        star_opterr = rextr["err_extr1"]

        star_robsp = rextr["spec_extr2"]
        star_roberr = rextr["err_extr2"]

        star_mask = rextr["mask_extr"]

        """ STAR results: [wave] [sp] [sp_err]"""
        exheader = OrderedDict([
            ("---PC---", "---PROCESSING---"),
            ("author", "Bo Zhang"),
            ("court", "home"),])
        star_header_ = OrderedDict([
            ("-LAYERS-", "-LAYERS-"),
            ("layer0", "optimal extracted spectrum"),
            ("layer1", "simple extracted spectrum"),
            ("layer2", "blaze function"),
            ("layer3", "wavelength before observation"),
            ("layer4", "wavelength after observation"),
            ("layer5", "error of optimal extracted spectrum"),
            ("layer6", "error of simple extracted spectrum"),
            ("layer7", "robust extracted spectrum"),
            ("layer8", "error of robust extracted spectrum"),
            ("layer9", "mask of extracted spectrum"),
            ("tharbfr", tharbfr_file),
            ("tharaft", tharaft_file)])

        im_meta.update(exheader)
        im_meta.update(star_header_)
        data = np.array([star_optsp,  # opt spec
                         star_sp,  # simple spec
                         self.master_flats[slit]['blz'],  # blaze
                         tharbfr,  # wave before
                         tharaft,  # wave after
                         star_opterr,  # opt error
                         star_err,
                         star_mask])  # simple error

        h0 = fits.hdu.PrimaryHDU(data, fits.Header(im_meta))
        hl = fits.HDUList([h0])
        hl.writeto(self.dir_work + "pstar{}_{:s}".format(slit, star_fn),
                   overwrite=True)

        return file


def _try_trace_apertures(flat, sigma_):
    return Aperture.trace(flat, method="canny", sigma=sigma_, verbose=False).n_ap


# used in draw()
def find_repeats(tsub, i1=0):
    c = 0
    for i in range(i1, len(tsub)):
        if tsub["SLIT", "IMAGETYP", "OBJECT"][i] == \
                tsub["SLIT", "IMAGETYP", "OBJECT"][i1]:
            c += 1
        else:
            break

    if tsub["IMAGETYP"][i1] in ["BIAS", "FLAT", "FLATI2", "THAR", "THARI2"]:
        this_label = tsub["IMAGETYP"][i1]
    else:
        this_label = tsub["OBJECT"][i1]
    this_mjd1 = tsub["MJD-MID"][i1]
    this_mjd2 = tsub["MJD-MID"][i1 + c - 1]
    this_slit = tsub["SLIT"][i1]

    return this_mjd1, this_mjd2, this_slit, this_label, c


# used in draw()
def find_all_repeats(tsub):
    repeats = []

    i1 = 0
    while i1 < len(tsub):
        repeats.append(find_repeats(tsub, i1))
        i1 += repeats[-1][-1]

    from astropy.table import Table
    return Table(np.array(repeats),
                 names=["mjd1", "mjd2", "slit", "label", "c"],
                 dtype=[float, float, int, str, int])


# random choice
def random_ind(n, m):
    """ from n choose m randomly """
    return np.argsort(np.random.rand(n,))[:m]


# list images of a specified type
def list_image(t, imagetp="FLAT", return_col=None, kwds=None, max_print=None):
    """ list images with specified IMAGETYP value

    Examples
    --------
    >>> list_image(t2, imagetp="STAR", kwds=["OBJECT"])

    Parameters
    ----------
    t: Table
        catalog of files, generated by *scan_files*
    imagetp: string
        IMAGETYP value
    kwds: list
        optional. additional columns to be displayed
    max_print:
        max line number

    Returns
    -------

    """
    ind_mst = np.where(t["IMAGETYP"] == imagetp)[0]

    if max_print is not None:
        if max_print > len(ind_mst):
            max_print = len(ind_mst)
    else:
        max_print = len(ind_mst)

    print("@SONG: these are all images of type %s" % imagetp)
    print("+ --------------------------------------------------")
    if isinstance(kwds, str):
        kwds = [kwds]
    if kwds is None or kwds == "":
        for i in range(max_print):
            print("+ %04d - %s" % (i, t["fps"][ind_mst[i]]))
    else:
        assert isinstance(kwds, list) or isinstance(kwds, tuple)
        for kwd in kwds:
            try:
                assert kwd in t.colnames
            except AssertionError:
                print("kwd", kwd)
                raise AssertionError()

        for i in range(max_print):
            s = "+ %04d - %s" % (i, t["fps"][ind_mst[i]])
            for kwd in kwds:
                s += "  %s" % t[kwd][ind_mst[i]]
            print(s)
    print("+ --------------------------------------------------")

    if return_col is not None:
        return t[return_col][ind_mst]


# CCD operations
def read_image(fp, hdu=0, gain=1., ron=0., unit='adu', trim=None, rot90=0):
    """ read image """
    ccds = CCD.read(fp, hdu=hdu, gain=gain, ron=ron, unit=unit, trim=trim,
                    rot90=rot90)
    return ccds


# combine CCDs
def combine_images(fps, method="median", std=False,
                   hdu=0, gain=1., ron=0., unit='adu', trim=None, rot90=0):
    """ combine images

    Parameters
    ----------
    fps:
        file paths
    method:
        mean or median
    std:
        False if you don't want std
        True if you want std, this is useful to estimate Read-Out-Noise
    kwargs:
        CCD.read() keyword args
    """
    ccds = [CCD.read(fp, hdu=hdu, gain=gain, ron=ron, unit=unit, trim=trim,
                     rot90=rot90) for fp in fps]

    if method == "mean":
        ccd_comb = CCD.mean(ccds)
    elif method == "median":
        ccd_comb = CCD.median(ccds)
    else:
        raise ValueError("@SONG: bad value for method [{}]".format(method))

    if not std:
        return ccd_comb
    else:
        ccd_std = CCD.std(ccds)
        return ccd_comb, ccd_std

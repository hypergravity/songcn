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
- Fri Mar 27 21:34:55 2020

Aims
----
- Song class

Notes
-----
The SONG-China project defines file names with local time. i.e.,
>>> import time
>>> time.localtime()

"""

import glob
import os
from collections import OrderedDict

import joblib
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from astropy.time import Time
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from twodspec import extract, thar
from twodspec.aperture import Aperture
from twodspec.ccd import CCD
from .slit import Slit
from .utils import scan_files


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
    # 3. scattered-light background for FLAT and STAR
    kwargs_background_flat = dict(q=(30, 1), kernel_size=(17, 17), sigma=(11, 7))
    kwargs_background_star = dict(q=(45, 45), kernel_size=(17, 17), sigma=(11, 7))
    # 4. normalize FLAT
    kwargs_normflat = dict(max_dqe=0.04, min_snr=20, smooth_blaze=5,
                           n_chunks=8, profile_oversample=10,
                           profile_smoothness=1e-2, num_sigma_clipping=20,
                           gain=1., ron=0, n_jobs=1)  # root directory
    # 5. extract 1D spectrum
    kwargs_extract = dict(n_chunks=8, profile_oversample=10,
                          profile_smoothness=1e-2, num_sigma_clipping=20,
                          gain=1., ron=0, n_jobs=1)

    # data & work directory
    rootdir = ""
    rawdir = ""
    extdir = ""
    rawdir1 = ""
    subdir = ""
    datejd0 = 0
    date = ""
    jdnight = True
    node = 2

    # unique slits
    unique_slits = []
    slits = []

    # master BIAS
    master_bias_fp = ""
    master_bias = None
    master_ron = None

    # master FLAT for each slit
    master_flats = dict()

    # THAR/STAR/FLATI2/TEST will be processed separately
    thar_list = dict()
    thari2_list = dict()

    star_list = list()
    stari2_list = list()
    flati2_list = list()
    test_list = list()

    # load ThAr template in Song class
    # print("@SONG: [ThAr] loading ThAr template ...")
    # thar_line_list = np.loadtxt(__path__[0] + "/calibration/thar.dat")
    # thar_solution_temp = (
    #     fits.getdata(__path__[0] + "/calibration/thar_template.fits", ext=1),
    #     fits.getdata(__path__[0] + "/calibration/thar_template.fits", ext=2),
    #     fits.getdata(__path__[0] + "/calibration/thar_template.fits", ext=3))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def new(self):
        """ initiate a new blank Song object """
        s = self.init(self.rootdir, self.date, self.jdnight, self.kwargs_scan["n_jobs"], self.kwargs_scan["verbose"], self.subdir, self.node)
        s.kwargs_read = self.kwargs_read
        s.kwargs_combine = self.kwargs_combine
        s.kwargs_background = self.kwargs_background
        return s

    @staticmethod
    def init(rootdir="/Users/cham/projects/song/star_spec", date="20191030", jdnight=True,
             n_jobs=-1, verbose=True, subdir="night", node=2):
        """ initiate from a directory path

        Parameters
        ----------
        rootdir: string
            root directory of data
        date: string
            optional
        jdnight: bool
            if True, use jd to define a night
        n_jobs:
            joblib.Parallel arguments
        verbose:
            joblib.Parallel arguments
        subdir:
            night / day
        node:
            1: Tenerife, 2: China

        Returns
        -------
        Song object
        """

        # directory path
        datejd0 = date2jd0(date)
        datedir = "{}/{}/{}".format(rootdir, date, subdir)
        rawdir = "{}/{}/{}/raw".format(rootdir, date, subdir)
        extdir = "{}/{}/{}/ext".format(rootdir, date, subdir)
        rawdir1 = "{}/{}/{}/raw".format(rootdir, dateplus1(date), subdir)

        # assert directories exist
        assert os.path.exists(datedir)  # date dir
        assert os.path.exists(rawdir)  # raw dir
        # if jdnight:
        #     assert os.path.exists(rawdir1)
        # if extdir does not exist, make it
        if not os.path.exists(extdir):
            print("@SONG: making directory [{}]..".format(extdir))
            os.mkdir(extdir)

        # define a night, glob files
        fps = glob.glob(rawdir + "/s{}_*.fits".format(node))
        if jdnight:
            fps.extend(glob.glob(rawdir1 + "/s{}_*.fits".format(node)))
        fps = np.array(fps)
        fps.sort()

        # extract jd
        if jdnight:
            jds = np.array([fp2jd(fp) for fp in fps])
            fps = fps[(jds >= datejd0) & (jds < datejd0 + 1)]

        # scan files
        print("@SONG: scanning files ...")
        s = Song(scan_files(fps, n_jobs=n_jobs, verbose=verbose))

        # add info
        s.rawdir = rawdir
        s.rawdir1 = rawdir1
        s.rootdir = rootdir
        s.extdir = extdir
        s.datejd0 = datejd0
        s.date = date
        s.jdnight = jdnight
        s.subdir = subdir
        s.node = node
        s.kwargs_scan["n_jobs"] = n_jobs
        s.kwargs_scan["verbose"] = verbose

        # get unique slits
        s.unique_slits = list(s.unique_config(cfgkeys=("SLIT")))

        return s

    def pipeline_bias(self, n_select=120, method_select='random'):
        """ process bias
        Parameters
        ----------
        n_select:
            number of frames used
        method_select:
            {"random", "top", "bottom", "all"}
        """
        self.master_bias, self.master_ron = self.ezmaster(
            {"IMAGETYP": "BIAS"}, n_select=n_select, method_select=method_select,
            method_combine='median', std=True)
        # write to disk
        self.master_bias_fp = "{}/masterbias_{}.fits".format(self.extdir, self.date)
        print("@SONG: master bias written to {}".format(self.master_bias_fp))
        self.master_bias.write(self.master_bias_fp, overwrite=True)
        return

    def pipeline_flat(self, slits="all", n_jobs_trace=1, verbose=10):

        if slits == "all":
            slits = self.unique_slits
        else:
            slits = [slits]

        for slit in slits:
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
            flat_fp = self.extdir + "/masterflat_{}_slit{}.fits".format(self.date, slit)
            print("@SONG: writing to {} ...".format(flat_fp))
            flat.write(flat_fp, overwrite=True)

            # 4. trace apertures
            ap = Aperture.trace(flat, method="naive", ap_width=15, sigma=7, maxdev=7, polydeg=4)

            # # figure of apertures
            # fig = plt.figure(figsize=(8, 8))
            # plt.imshow(np.log10(flat.data))
            # plt.plot(ap.ap_center.T, ap.x, 'w')
            # plt.xlabel("X coordinate")
            # plt.ylabel("Y coordinate")
            # plt.title("SLIT {}".format(slit))
            # fig.tight_layout()

            # # figure of horizontal slice
            # fig = plt.figure(figsize=(12, 6))
            # plt.plot(flat[1024])
            # plt.vlines(ap.ap_center[:, 1023], 0, 10000)
            # plt.xlabel("X coordinate")
            # plt.ylabel("Y coordinate")
            # plt.title("SLIT {} | Slice of row 1024".format(slit))
            # fig.tight_layout()

            # 5.background
            print("@SONG: computing background ...")
            bg = CCD(ap.background(flat, **self.kwargs_background_flat))
            bg_fp = self.extdir + "/masterflat_{}_slit{}_bg.fits".format(self.date, slit)
            bg.write(bg_fp, overwrite=True)
            flat = flat - bg

            # 6. calculate blaze functions and sensitivity
            print("@SONG: computing blaze function & sentitivity ...")
            # blz_bg, norm_bg = extract.make_normflat(np.array(flat), ap, **self.kwargs_normflat)
            blaze, sensitivity = ap.make_normflat(flat, **self.kwargs_normflat)

            blaze = CCD(blaze)
            blaze_fp = self.extdir + "/masterblaze_{}_slit{}.fits".format(self.date, slit)
            print("@SONG: writing blaze function to {}...".format(blaze_fp))
            blaze.write(blaze_fp, overwrite=True)

            sensitivity = CCD(sensitivity)
            sensitivity_fp = self.extdir + "/mastersens_{}_slit{}.fits".format(self.date, slit)
            print("@SONG: writing sentivity to {}...".format(sensitivity_fp))
            sensitivity.write(sensitivity_fp, overwrite=True)

            self.master_flats[slit] = dict(
                # slit
                slit=slit,  # slit number
                # flat
                flat=flat,  # flat-bias
                flat_fp=flat_fp,  # flat fp
                # apertures
                ap=ap,
                # background
                bg=bg,
                bg_fp=bg_fp,
                # blaze function
                blaze=blaze,
                blaze_fp=blaze_fp,
                # sentivity
                sensitivity=sensitivity,
                sensitivity_fp=sensitivity_fp,
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
                poly_order=poly_order, slit=slit)
            sgrid_fitted_wave = calibration_results[0]
            wave_final = np.copy(np.flipud(sgrid_fitted_wave))

            # save figures
            fig = calibration_results[2]
            fig.savefig(self.extdir + "thar{}_{:s}".format(
                slit, thar_fn.replace(".fits", "_diagnostics.pdf")))
            plt.close(fig)
            fig = calibration_results[3]
            fig.savefig(self.extdir + "thar{}_{:s}".format(
                slit, thar_fn.replace(".fits", "_used_lines.pdf")))
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
            hl.writeto(self.extdir + "thar{}_{:s}".format(slit, thar_fn),
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
            delayed(self._pipeline_a_star)(self["fps"][i_star])
            for i_star in sub_star[ind_unprocessed])

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
        n_select:
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
        """ find number of unique config """
        result = np.asarray(np.unique(self[cfgkeys]))
        print("@SONG: {0} unique config found!".format(len(result)), list(result))
        self.describe().pprint()
        return result

    def describe(self, cfgkeys=("SLIT", "IMAGETYP")):
        """ generate a table of stats on image configs

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
        print("[SUMMARY] {:s}".format(self.extdir))
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
        """ read a CCD frame """
        return CCD.read(fp, **self.kwargs_read)

    def reads(self, fps, method="median", std=False):
        """ actually a combine method """
        return CCD.reads(fps, method=method, std=std, **self.kwargs_read)

    def ezmaster(self, cond_dict={"IMAGETYP": "BIAS"}, n_select=10,
                 method_select="top", method_combine="median", std=False):
        """

        Parameters
        ----------
        cond_dict: dict
            conditions on {"BIAS", "FLAT", "FLATI2", "THAR", "THARI2", "STAR", "STARI2", "TEST"}
        n_select: int
            number of images will be selected
        method_select:
            scheme of selection
        method_combine:
            method of combining
        std:
            if True, return average and std

        Returns
        -------
        combined image

        """

        assert method_select in {"random", "top", "bottom", "all"}

        # if any cond_dict key does not exist in song
        try:
            for k in cond_dict.keys():
                assert k in self.colnames
        except Exception:
            raise ValueError("@SONG: key not found: {0}".format(k))

        # find fps of matched images
        fps = self.select(cond_dict, method=method_select, n_select=n_select)
        print("@SONG: *ezmaster* working on ", fps)

        # combine all selected images
        return self.reads(fps, method=method_combine, std=std)

    def draw(self, save=None, figsize=(15, 6), return_fig=False):
        """ a description of observation """
        # ljd = self["JD-MID"] + 8 / 24
        # lhour = (np.mod(ljd, 1) + 0.5) * 24

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # sub table for text
        tsub = self["JD-MID", "SLIT", "IMAGETYP", "OBJECT"]
        # tsub["JD-MID"] = lhour
        tsub.sort("JD-MID")
        trep = find_all_repeats(tsub)

        _ms = 15
        ind = self["IMAGETYP"] == "BIAS"
        ax.plot(tsub["JD-MID"][ind], tsub["SLIT"][ind], "o", mfc="none", mec='g', ms=_ms, label="BIAS")
        ind = self["IMAGETYP"] == "FLAT"
        ax.plot(tsub["JD-MID"][ind], tsub["SLIT"][ind], "^", mfc="none", mec='r', ms=_ms, label="FLAT")
        ind = self["IMAGETYP"] == "FLATI2"
        ax.plot(tsub["JD-MID"][ind], tsub["SLIT"][ind], "^", mfc="none", mec='b', ms=_ms, label="FLATI2")
        ind = self["IMAGETYP"] == "THAR"
        ax.plot(tsub["JD-MID"][ind], tsub["SLIT"][ind], "D", mfc="none", mec='c', ms=_ms, mew=2, label="THAR")
        ind = self["IMAGETYP"] == "STAR"
        ax.plot(tsub["JD-MID"][ind], tsub["SLIT"][ind], "s", mfc="none", mec='m', ms=_ms, label="STAR")
        ind = self["IMAGETYP"] == "STARI2"
        ax.plot(tsub["JD-MID"][ind], tsub["SLIT"][ind], "s", mfc="none", mec='y', ms=_ms, label="STARI2")
        ind = self["IMAGETYP"] == "TEST"
        ax.plot(tsub["JD-MID"][ind], tsub["SLIT"][ind], "s", mfc="none", mec='gray', ms=_ms, label="TEST")

        # annotate the repeats
        for i in range(len(trep)):
            if trep['label'][i] not in ["BIAS", "FLAT", "FLATI2", "THAR", "THARI2"]:
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
                tick_x = [trep["jd1"][i], trep["jd1"][i], trep["jd2"][i], trep["jd2"][i]]
                tick_y = [trep["slit"][i], trep["slit"][i] + tick_height, trep["slit"][i] + tick_height, trep["slit"][i]]
                ax.plot(tick_x, tick_y, 'k')
                ax.text(trep["jd1"][i] * .5 + trep["jd2"][i] * .5, trep["slit"][i] + tick_height,
                        trep['label'][i] + " x {}".format(trep['c'][i]), **text_kwargs)
            else:
                tick_x = [trep["jd1"][i], trep["jd1"][i]]
                tick_y = [trep["slit"][i], trep["slit"][i] + tick_height]
                ax.plot(tick_x, tick_y, 'k')
                ax.text(trep["jd1"][i], trep["slit"][i] + tick_height,
                        trep['label'][i] + " x {}".format(trep['c'][i]), **text_kwargs)

        ax.legend(loc=3)
        ax.set_xlabel("Julian Day")
        ax.set_ylabel("Slit Number")
        ax.set_yticks(np.arange(9) + 1)
        _ylim = self["SLIT"].min() - 2, self["SLIT"].max() + 2
        ax.set_ylim(_ylim)
        ax.set_xticks(np.linspace(self.datejd0 - 8 / 24, self.datejd0 - 8 / 24 + 1, 13))
        ax.set_xlim(self.datejd0 - 8 / 24, self.datejd0 - 8 / 24 + 1)
        ax.grid(True, axis="y", ydata=np.arange(11))

        ax2 = ax.twiny()
        ax2.set_xlabel("{}-{}-{} Beijing local time (UTC+8 Hour)".format(self.date[:4], self.date[4:6], self.date[6:8]))
        ax2.set_xticks(np.linspace(12, 36, 13))
        ax2.set_xticklabels(["{:.0f}".format(_) for _ in np.mod(np.linspace(12, 36, 13), 24)])
        ax2.set_xlim(12, 36)
        ax2.grid(True, axis='x', linestyle='--')

        fig.tight_layout()

        if save is True:
            # save to default
            figpath = self.extdir+"/summary_{}.pdf".format(self.date)
            print("@SONG: saving summary figure to {}".format(figpath))
            fig.savefig(figpath)
        elif save is not None:
            figpath = save
            print("@SONG: saving summary figure to {}".format(figpath))
            fig.savefig(save)

        if return_fig:
            return fig
        else:
            plt.close(fig)
            return

    def _pipeline_a_star(self, star_fp):

        # read STAR
        hdr = fits.getheader(star_fp)
        slit = hdr["SLIT"]
        mjdmid = hdr["MJD-MID"]

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
            print("@SONG: STAR[{}] THAR0[{}]".format(star_fn, tharbfr_file))
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
            print("@SONG: STAR[{}] THAR1[{}]".format(star_fn, tharaft_file))

        # sensitivity correction
        im_star_denm = (im_star - self.master_bias) / self.master_flats[slit]['norm']
        # im_star_denm_err = np.sqrt(np.abs(im_star_denm)) / flats[slit]['norm']

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
                         star_robsp,
                         star_roberr,
                         star_mask])  # simple error

        h0 = fits.hdu.PrimaryHDU(data, fits.Header(im_meta))
        hl = fits.HDUList([h0])
        hl.writeto(self.extdir + "pstar{}_{:s}".format(slit, star_fn),
                   overwrite=True)

        return star_fn

    def daily(self, ipcprofile="default"):
        """ daily pipeline """
        joblib.dump(self, "{}/{}_song.dump".format(self.extdir, self.date))
        print("===========================================")
        print("@Song: unique slits are ", self.unique_slits)
        print("===========================================")

        for slit in self.unique_slits:
            this_slit = Slit(slit=slit, extdir=self.extdir)
            # bias
            fps_bias = self["fps"][self.ezselect_all({"IMAGETYP": "BIAS", "SLIT": slit})]
            this_slit.proc_bias(fps_bias)
            # flat
            fps_flat = self["fps"][self.ezselect_all({"IMAGETYP": "FLAT", "SLIT": slit})]
            this_slit.proc_flat(fps_flat)
            # thar
            fps_thar = list(self["fps"][self.ezselect_all({"IMAGETYP": "THAR", "SLIT": slit})][:])
            this_slit.proc_thar(fps_thar, ipcprofile=ipcprofile)
            # star
            fps_star = list(self["fps"][self.ezselect_all({"IMAGETYP": "STAR", "SLIT": slit})][:])
            this_slit.proc_star(fps_star, ipcprofile=ipcprofile)

            joblib.dump(this_slit, "{}/{}_slit{}.dump".format(self.extdir, self.date, slit))
            print("===========================================")
        print("DONE!~")
        print("===========================================")
        return


def _try_trace_apertures(flat, sigma_):
    try:
        ap = Aperture.trace(flat, method="canny", sigma=sigma_, verbose=False)
        return ap.n_ap
    except Exception as e_:
        return -1


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
    this_mjd1 = tsub["JD-MID"][i1]
    this_mjd2 = tsub["JD-MID"][i1 + c - 1]
    this_slit = tsub["SLIT"][i1]
    return this_mjd1, this_mjd2, this_slit, this_label, c


def find_all_repeats(tsub):
    repeats = []
    i1 = 0
    while i1 < len(tsub):
        repeats.append(find_repeats(tsub, i1))
        i1 += repeats[-1][-1]
    return Table(np.array(repeats),
                 names=["jd1", "jd2", "slit", "label", "c"],
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
    ccds = CCD.read(fp, hdu=hdu, gain=gain, ron=ron, unit=unit, trim=trim, rot90=rot90)
    return ccds


# combine CCDs
def combine_images(fps, method="median", std=False, hdu=0, gain=1., ron=0., unit='adu', trim=None, rot90=0, **kwargs):
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


# convert date string
# print(date2jd0("20191031"))
def date2jd0(s="20191031"):
    ds = "{}-{}-{}T12:00:00".format(s[0:4],s[4:6],s[6:8])
    return Time(ds,format="isot").jd


# print(jd02date(2458788.0+1))
def jd02date(jd):
    ds = Time(jd,format="jd").isot
    return ds[:10].replace("-", "")


# print(dateplus1("20191031"))
def dateplus1(s="20191031"):
    return jd02date(date2jd0(s)+1)


def fp2jd(fp="s2_2019-10-30T19-56-00.fits"):
    fn = os.path.basename(fp)
    dtisot = "{}:{}:{}".format(fn[3:16], fn[17:19], fn[20:22])
    return Time(dtisot, format="isot").jd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 12:28:52 2017

@author: Bo Zhang (bozhang@nao.cas.cn)

This module is to wrap some of the functions in STELLA (Ritter et al. 2014),
e.g., optimal extraction.

The scheme is to run the C executables, such as optextract through subprocess.Popen().

"""

import os
import subprocess
from tempfile import NamedTemporaryFile

import ccdproc
import numpy as np
from astropy.io import fits

from .aprec import AprecList

COMMAND_LIST = {"optextract", "extractsum", "makenormflat"}

# ######################### #
# simple wrappers of STELLA #
# ######################### #


def command_assert(cmd):
    """ assert a command in the list of wrapped commands """
    assert cmd in COMMAND_LIST


def print_stdout(bts):
    """ print stdout on screen """
    str_list = bts.decode("utf-8").split("\n")
    for str_ in str_list:
        print(str_)
    return


def command_help(cmd='optextract', shell=False):
    """
    
    Parameters
    ----------
    cmd: string
        command name
    shell: bool
        if True, shell function is enabled

    Returns
    -------
    
    Examples
    --------
    >>> command_help('extractsum')
    
    """
    assert cmd in COMMAND_LIST
    sp = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE)
    print_stdout(sp.stdout)
    return


def command_run(cmd, **kwargs):
    """ to run a bash command
    
    Parameters
    ----------
    cmd: 
        command and args
    kwargs: 
        keyword arguments passed to subprocess.run()

    Returns
    -------

    """
    s = subprocess.run(cmd, stdout=subprocess.PIPE, **kwargs)
    if s.returncode == 0:
        # success
        return 
    else:
        # fail
        print_stdout(s.stdout)
        raise (RuntimeError(
            "@Cham: something went wrong when executing command '%s'" % (
            " ".join(s.args))))


def value2string(val):
    """ to convert val to string """
    if val is None:
        return ""
    else:
        return str(val)

    
def args2command(*args):
    """ to convert positional arguments to string list """
    try:
        assert None not in args
        assert "" not in args        
    except:
        print("args:", args)
        raise(ValueError("None values not allowed in args!"))
    return [str(_).strip() for _ in args]


def kwargs2command(**kwargs):
    """ to convert keyword arguments to string list """
    cmd = []
    for k, v in kwargs.items():
        if v is not None:
            cmd.append("{}={}".format(str(k).strip(), str(v).strip()))
    return cmd


# ############################# #
#    wrapped STELLA commands    #
# ############################# #

def extractsum(*args, **kwargs):
    """
    USAGE: extractsum
    <char[] FitsFileName_In>
    <char[] DatabaseFileName_In> 
    <char[] FitsFileName_Out> 
    <double Gain> 
    [ERR_IN=char[]] 
    [ERR_OUT_EC=char[]] 
    [AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]] 
    [APERTURES=<char[] ApertureFile_In]
    """
    if len(args) + len(kwargs) == 0:
        command_help("extractsum")
        return
    cmd = ["extractsum"]
    cmd.extend(args2command(*args))
    cmd.extend(kwargs2command(**kwargs))
    print("@stella.py: executing command ""{}""".format(cmd))
    command_run(cmd)
    return


def optextract(*args, **kwargs):
    """
    USAGE: optextract
    <char[] (@)FitsFileName_In>
    <char[] (@)DatabaseFileName_In>
    <char[] (@)FitsFileName_Out>
    <double Gain>
    <double ReadOutNoise>
    [TELLURIC=int[0 - none, 1 - Piskunov, 2 - LinFit]]
    [MAX_ITER_SF=int] 
    [MAX_ITER_SKY=int]
    [MAX_ITER_SIG=int]
    [SWATH_WIDTH=int] 
    [SF_SMOOTH=double] 
    [SP_SMOOTH=int] 
    [WING_SMOOTH_FACTOR=double] 
    [ERR_IN=char[](@)] 
    [ERR_OUT_2D=char[](@)] 
    [ERR_OUT_EC=char[](@)] 
    [ERRFIT_OUT_EC=char[](@)] 
    [SKY_OUT_EC=char[](@)]
    [SKYFIT_OUT_EC=char[](@)] 
    [SKY_OUT_2D=char[](@)] 
    [SKYFIT_OUT_2D=char[](@)]
    [SKY_ERR_OUT_EC=char[](@)] 
    [SKYFIT_ERR_OUT_EC=char[](@)]
    [PROFILE_OUT=char[](@)] 
    [IM_REC_OUT=char[](@)] 
    [REC_FIT_OUT=char[](@)] 
    [MASK_OUT=char[](@)] 
    [SPFIT_OUT_EC=char[](@)] 
    [EC_FROM_PROFILE_OUT=char[](@)]
    [AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]]
    [XCOR_PROF=int] 
    [APERTURES=char[](@)]

    FitsFileName_In: image to extract
    DatabaseFileName_In: aperture-definition file to use for extraction
    FitsFileName_Out: output filename containing extracted spectra
    Gain: CCD gain
    ReadOutNoise: CCD readout noise
    TELLURIC: 0 - none, 1 - Piskunov, 2 - LinFit
    MAX_ITER_SF: maximum number of iterations calculating the slit function 
        (spatial profile)
    MAX_ITER_SKY: maximum number of iterations calculating the sky 
        (TELLURIC = 2 only)
    MAX_ITER_SIG: maximum number of iterations rejecting cosmic-ray hits
    SWATH_WIDTH: width of swath (bin) for which an individual profile shall be
        calculated
    SMOOTH_SF: Width of median SlitFunc smoothing
    SMOOTH_SP: Width of median Spectrum/Blaze smoothing
    WING_SMOOTH_FACTOR: Width of median SlitFunc-Wing smoothing
    ERR_IN: input image containing the uncertainties in the pixel values of 
        FitsFileName_In
    ERR_OUT_2D: output uncertainty image - same as ERR_IN, but with detected 
        cosmic-ray hits set to 10,000
    ERR_OUT_EC: output file containing the uncertainties in the extracted 
        spectra's pixel values from ExtractErrors
    ERRFIT_OUT_EC: output file containing the uncertainties in the extracted 
        spectra's pixel values from Fit
    SKY_OUT_EC: output sky-only spectra (TELLURIC > 0 only)
    SKYFIT_OUT_EC: output sky-only spectra (TELLURIC > 0 only)
    SKY_OUT_2D: reconstructed sky-only image
    SKYFIT_OUT_2D: reconstructed sky-only image
    SKY_ERR_OUT_EC: uncertainties in the calculated sky-only values 
        (only differs from SKYFIT_ERR_OUT_EC if TELLURIC==1
    SKYFIT_ERR_OUT_EC: uncertainties in the calculated sky-only values from Fit
    PROFILE_OUT: reconstructed image of the spatial profiles
    IM_REC_OUT: reconstructed input image from the profile-fitting/extraction
    SPFIT_OUT_EC: extracted spectra from linear fit of spatial profiles to 
        input spectra with 3-sigma rejection (ignoring mask), with sky if 
        TELLURIC>0, without sky if TELLURIC=0
    REC_FIT_OUT: reconstructed input image for SPFIT_OUT_EC
    MASK_OUT: output mask with detected cosmic-ray hits set to 0, good pixels 
        set to 1
    EC_FROM_PROFILE_OUT: extracted spectra from simply multiplying the input 
        image with the profile image as weight and summing up
    AREA: Area from which to extract spectra if center of aperture is in 
        specified area
    XCOR_PROF: How many cross-correlation runs from -1 pixel to +1 pixel 
        compared to XCenter?
    APERTURES: input filename containing a list of apertures to extract
    """
    if len(args) + len(kwargs) == 0:
        command_help("optextract")
        return
    cmd = ["optextract"]
    cmd.extend(args2command(*args))
    cmd.extend(kwargs2command(**kwargs))
    print("@stella.py: executing command ""{}""".format(cmd))
    command_run(cmd)
    return


def makenormflat(*args, **kwargs):
    """
    
    USAGE: makenormflat 
    <char[] FitsFileName_In> 
    <char[] DatabaseFileName_In> 
    <char[] NormalisedFlat_Out> 
    <char[] ProfileFits_Out> 
    <char[] BlazeFits_Out> 
    <double Gain> 
    <double ReadOutNoise> 
    <int SmoothSP> 
    <double MinSNR> 
    [<int SwathWidth>] 
    [ERR_IN=<char[] ErrFileName_In>]
    [SMOOTH_SF=<double smooth_slit_func>]
    [AREA=[<int xmin>,<int xmax>,<int ymin>,<int ymax>]]
    [MASK_OUT=<char[] Mask_Out>] 
    [RECONSTRUCTED_IMAGE_OUT=<char[] ReconstructedImage_Out>]
    Parameter 1: <char[] FitsFileName_In>: input flat fits file name
    Parameter 2: <char[] DatabaseFileName_In>: input Database file name
    Parameter 3: <char[] NormalisedFlat_Out>: output fits file name
    Parameter 4: <char[] ProfileFits_Out>: output profile image fits file name
    Parameter 5: <char[] BlazeFits_Out>: output blaze functions fits file name
    Parameter 6: <double Gain>: CCD gain
    Parameter 7: <double ReadOutNoise>: CCD read-out noise
    Parameter 8: <int SmoothSP>: smooth blaze function over SmoothSP pixels
    Parameter 9: <double MinSNR>: minimum S/N to use for each pixel of the 
        normalised flat. Pixels with lower S/N are set to unity
    Parameter 10: [<int SwathWidth>]: optional, length of swath to be used for 
        the determination of the spatial profile
    Parameter 11: [ERR_IN=<char[] ErrFileName_In>]: optional, name of fits file
        containing the uncertainty for the FitsFileName_In. If not set, the 
        errors will be calculated as the square root of FitsFileName_In
    Parameter 12: [SMOOTH_SF=<double smooth_slit_func>]: optional, smoothing 
        factor for the spatial profile (default = 1/10 with 10 pixels 
        oversampling)
    Parameter 13: [AREA=[<int xmin>,<int xmax>,<int ymin>,<int ymax>]]: 
        optional, area on the CCD to calculate and remove scattered light
    Parameter 14: [MASK_OUT=<char[] Mask_Out>]: optional, output fits file 
        containing the mask for the detected bad-pixels / cosmic-rays
    Parameter 15: [RECONSTRUCTED_IMAGE_OUT=<char[] ReconstructedImage_Out>]: 
        optional, output fits file containing the reconstructed flat
    example: makenormflat flat.fits database/apflat normalizedFlat.fits 
        flat_prof.fits flat_blaze.fits 3.8 10.5 5 100 300 SMOOTH_SF=0.5 
        MASK_OUT=flat_mask.fits RECONSTRUCTED_IMAGE_OUT=flat_rec.fits
    
    """
    if len(args) + len(kwargs) == 0:
        command_help("makenormflat")
        return
    cmd = ["makenormflat"]
    cmd.extend(args2command(*args))
    cmd.extend(kwargs2command(**kwargs))
    print("@stella.py: executing command ""{}""".format(cmd))
    command_run(cmd)
    return


# ##################################### #
# wrap STELLA functions at memory level #
# ##################################### #
PREFIX = "_twodspec_"


def write_image(im, filename=None, dir_work="", prefix="_twodspec_",
                suffix=".fits", delete=False):
    if filename is None:
        # tempfile in dir_work
        tf_im = NamedTemporaryFile(
            dir=dir_work, prefix=prefix, suffix=suffix, delete=delete)
        ccdproc.CCDData(im, unit='adu').write(tf_im.name, clobber=True)
        print("writing to {}".format(tf_im.name))
        tf_im.close()
        return tf_im.name
    else:
        # filename
        ccdproc.CCDData(im, unit='adu').write(filename, clobber=True)
        print("writing to {}".format(filename))
        return filename


def temp_image(dir_work=None):
    tf_im = NamedTemporaryFile(
        dir=dir_work, prefix=PREFIX, suffix=".fits", delete=False)
    return tf_im.name


def extractsum_prime(dir_work, im_in, database, err_in=None, err_out=None,
                     gain=1., **kwargs):
    """
    USAGE: extractsum
    <char[] FitsFileName_In>
    <char[] DatabaseFileName_In>
    <char[] FitsFileName_Out>
    <double Gain>
    [ERR_IN=char[]]
    [ERR_OUT_EC=char[]]
    [AREA=[int(xmin),int(xmax),int(ymin),int(ymax)]]
    [APERTURES=<char[] ApertureFile_In]
    """
    if dir_work[-1] != "/":
        dir_work += "/"

    if err_in is None:
        err_in = np.sqrt(np.abs(im_in))

    tf_im_in = write_image(im_in, filename=None, dir_work=dir_work)
    tf_im_out = temp_image(dir_work=dir_work)
    tf_err_in = write_image(err_in, filename=None, dir_work=dir_work)
    tf_err_out = temp_image(dir_work=dir_work)

    extractsum(tf_im_in, database, tf_im_out, gain,
               ERR_IN=tf_err_in, ERR_OUT_EC=tf_err_out, **kwargs)

    sp = ccdproc.CCDData.read(tf_im_out, unit="adu").data
    sp_err = ccdproc.CCDData.read(tf_err_out, unit="adu").data
    os.remove(tf_im_in)
    os.remove(tf_im_out)
    os.remove(tf_err_in)
    os.remove(tf_err_out)

    return sp, sp_err


def optextract_prime(dir_work, im_in, database, err_in=None, im_rec=None,
                     gain=1., ron=10., **kwargs):

    if dir_work[-1] != "/":
        dir_work += "/"

    if err_in is None:
        err_in = np.sqrt(np.abs(im_in))

    tf_im_in = write_image(im_in, filename=None, dir_work=dir_work)
    tf_im_out = temp_image(dir_work=dir_work)
    tf_err_in = write_image(err_in, filename=None, dir_work=dir_work)
    tf_err_out = temp_image(dir_work=dir_work)

    try:
        # try to extract all apertures
        if im_rec:
            tf_im_rec = temp_image(dir_work=dir_work)
            optextract(tf_im_in, database, tf_im_out, gain, ron,
                       ERR_IN=tf_err_in, ERR_OUT_EC=tf_err_out,
                       IM_REC_OUT=tf_im_rec, **kwargs)
            sp = ccdproc.CCDData.read(tf_im_out, unit="adu").data
            sp_err = ccdproc.CCDData.read(tf_err_out, unit="adu").data
            im_rec = ccdproc.CCDData.read(tf_im_rec, unit="adu").data
            os.remove(tf_im_in)
            os.remove(tf_im_out)
            os.remove(tf_err_in)
            os.remove(tf_err_out)
            os.remove(tf_im_rec)
            return sp, sp_err, im_rec
        else:
            optextract(tf_im_in, database, tf_im_out, gain, ron,
                       ERR_IN=tf_err_in, ERR_OUT_EC=tf_err_out, **kwargs)
            sp = ccdproc.CCDData.read(tf_im_out, unit="adu").data
            sp_err = ccdproc.CCDData.read(tf_err_out, unit="adu").data
            os.remove(tf_im_in)
            os.remove(tf_im_out)
            os.remove(tf_err_in)
            os.remove(tf_err_out)
            return sp, sp_err

    except RuntimeError as re_:
        # extract apertures one by one

        # split AprecList
        al = AprecList.read(database)
        al_fps = al.split(database+"_{:04d}")

        # extract each aperture
        sp = []
        sp_err = []
        for iap in range(al.nap):
            try:
                optextract(tf_im_in, al_fps[iap], tf_im_out, gain, ron,
                           ERR_IN=tf_err_in, ERR_OUT_EC=tf_err_out, **kwargs)
                sp.append(fits.getdata(tf_im_out).flatten())
                sp_err.append(fits.getdata(tf_err_out).flatten())

            except RuntimeError:
                print(tf_im_out)
                print(sp)
                print(sp_err)
                sp.append(None)
                sp_err.append(None)

        # replace None values
        isnotnone = [_ is not None for _ in sp]
        print("sp:", sp)
        if np.any(isnotnone):
            # default array
            defarr = sp[np.where(isnotnone)[0][0]] * np.nan
            for i in range(len(sp)):
                if sp[i] is None:
                    sp[i] = defarr
                    sp_err[i] = defarr
        else:
            print()
            raise RuntimeError("@STELLA: all apertures are bad!")

        return np.array(sp), np.array(sp_err)


def makenormflat_prime(dir_work, im_in, database, err_in=None, gain=1.,
                       ron=10., smoothsp=5, minsnr=15, bg=None, maxdev=None,
                       **kwargs):

    if dir_work[-1] != "/":
        dir_work += "/"

    if err_in is None:
        err_in = np.sqrt(np.abs(im_in))

    tf_im_in = write_image(im_in, filename=None, dir_work=dir_work)
    # tf_im_out = temp_image(dir_work=dir_work)
    tf_err_in = write_image(err_in, filename=None, dir_work=dir_work)
    tf_im_sst = temp_image(dir_work=dir_work)
    # tf_im_rec = temp_image(dir_work=dir_work)
    tf_im_prf = temp_image(dir_work=dir_work)
    tf_im_blz = temp_image(dir_work=dir_work)

    makenormflat(tf_im_in, database, tf_im_sst, tf_im_prf, tf_im_blz, gain,
                 ron, smoothsp, minsnr, ERR_IN=tf_err_in, **kwargs)
                 # RECONSTRUCTED_IMAGE_OUT=tf_im_rec,
    blaze = ccdproc.CCDData.read(tf_im_blz, unit="adu").data
    # im_rec_ = ccdproc.CCDData.read(tf_im_rec, unit="adu").data
    sensitivity = ccdproc.CCDData.read(tf_im_sst, unit="adu").data

    os.remove(tf_im_in)
    os.remove(tf_err_in)
    os.remove(tf_im_sst)
    # os.remove(tf_im_rec)
    os.remove(tf_im_prf)
    os.remove(tf_im_blz)

    if bg is None:
        # no background is added back
        if maxdev is not None:
            sensitivity = np.where(
                np.abs(sensitivity-1) >= maxdev, 1.0, sensitivity)
        return blaze, sensitivity
    else:
        # background is added back to calculate sensitivity
        sensitivity = np.where(
            sensitivity == 1.0, 1.0, im_in / ((im_in - bg) / sensitivity + bg))
        if maxdev is not None:
            sensitivity = np.where(
                np.abs(sensitivity-1) >= maxdev, 1.0, sensitivity)
        return blaze, sensitivity


# #%%
#
# command_run(['optextract','tmp_s2_2017-01-13T12-17-47.fits','masteraper_20170113_slit8.txt','tmp_s2_2017-01-13T12-17-47_ext.fits',' 1.0',' 10.0','ERR_IN=tmp_s2_2017-01-13T12-17-47.fits','SMOOTH_SF=0.1'])
#
#
# s = subprocess.run(["cat", ], stdout=subprocess.PIPE, input=b"abc")
#
# subprocess.run(["ls", "-l", "/dev/null"], stdout=subprocess.PIPE)
#
# cmd = [" tmp_s2_2017-01-13T12-17-47.fits", "masteraper_20170113_slit8.txt", "tmp_s2_2017-01-13T12-17-47_ext.fits", "1.0", "10.0", "ERR_IN=tmp_s2_2017-01-13T12-17-47.fits", "SMOOTH_SF=0.1"]
# cmd = ["optextract tmp_s2_2017-01-13T12-17-47.fits masteraper_20170113_slit8.txt tmp_s2_2017-01-13T12-17-47_ext.fits 1.0 10.0 ERR_IN=tmp_s2_2017-01-13T12-17-47.fits SMOOTH_SF=0.1"]
# cmd = ["optextract",]
#
#
# optextract(*cmd)
#
#
# s = subprocess.run(cmd, stdout=subprocess.PIPE)
# s = command_run(cmd)
# print_stdout(s.stdout)
#
# s = subprocess.run("optextract", shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# s.stdout
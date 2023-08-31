import os.path
import glob
from astropy.io import fits
import shutil

dir_star_spec = "/Users/cham/projects/song/star_spec/20191031/night/raw"
dir_unit_test = "/Users/cham/projects/song/unittest/20191031"

fl = glob.glob(dir_star_spec + "/*.fits")
fl.sort()
fl_thar = []
fl_star = []
print(f"{len(fl)} fits files found")

for f in fl:
    # get header
    h = fits.getheader(f)
    print(
        f,
        h["IMAGETYP"],
        h["EXPTIME"],
        h["SLIT"],
    )

    # copy file
    shutil.copy2(
        f, os.path.join(dir_unit_test, h["IMAGETYP"].lower(), os.path.basename(f))
    )

# combine bias & dark
import numpy as np

fl_bias = glob.glob(os.path.join(dir_unit_test, "bias", "*.fits"))
bias_mean = np.mean(np.array([fits.getdata(f) for f in fl_bias]), axis=0)
fits.PrimaryHDU(data=bias_mean).writeto(os.path.join(dir_unit_test, "bias.fits"))

fl_flat = glob.glob(os.path.join(dir_unit_test, "flat", "*.fits"))
flat_mean = np.mean(np.array([fits.getdata(f) for f in fl_flat]), axis=0)
fits.PrimaryHDU(data=flat_mean).writeto(os.path.join(dir_unit_test, "flat.fits"))

fits.PrimaryHDU(data=flat_mean - bias_mean).writeto(
    os.path.join(dir_unit_test, "flat-bias.fits")
)

star = fits.getdata(os.path.join(dir_unit_test, "star.fits"))
fits.PrimaryHDU(data=star - bias_mean).writeto(
    os.path.join(dir_unit_test, "star-bias.fits")
)
thar = fits.getdata(os.path.join(dir_unit_test, "thar.fits"))
fits.PrimaryHDU(data=thar - bias_mean).writeto(
    os.path.join(dir_unit_test, "thar-bias.fits")
)

# # check images
# import matplotlib.pyplot as plt
#
# plt.imshow(np.log10(star - bias_mean))
# plt.imshow(thar)
# plt.imshow(np.log10(thar - bias_mean))
#
# plt.plot(star[1000])
# plt.plot(star[1000] - bias_mean[1000])
#
# plt.plot(flat_mean[1000])
# plt.plot(flat_mean[1000] - bias_mean[1000])

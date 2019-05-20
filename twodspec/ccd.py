
import os
import numpy as np
from astropy.io import fits
from collections import OrderedDict


class CCD(np.ndarray):
    """ CCD class, to cope with ccd images

    Note
    ----
    This class is a subclass of numpy.ndarray.

    References
    ----------
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    https://docs.scipy.org/doc/numpy/reference/arrays.classes.html

    """

    # ccd info
    gain = 1
    ron = 0
    unit = "adu"

    # file info, only when read from fits file
    fp = ""
    hdu = 0
    header = None
    meta = OrderedDict()

    # post-processing info
    _trim = None
    _rot90 = 0

    extr_attr_list = ["gain", "ron", "unit",            # ccd info
                      "fp", "hdu", "header", "meta",    # file info
                      "_trim", "_rot90"]                # post-proc info

    ###########################
    # print info
    ###########################
    def pprint(self):
        s = ("< twodspec.ccd.CCD instance {} x {} >\n"
             "gain  = {}\n"
             "ron   = {}\n"
             "unit  = {}\n"
             "trim  = {}\n"
             "rot90 = {}\n"
             "fp    = {}\n"
             "{}".format(*self.shape,
                         self.gain, self.ron, self.unit, self._trim,
                         self._rot90, self.fp,
                         self.__repr__()))
        print(s)
        return

    @property
    def info(self):
        """ print basic info

        Usage:
        ------
        ccd.info

        """
        s = ("< twodspec.ccd.CCD instance {} x {} >\n"
             "gain  = {}\n"
             "ron   = {}\n"
             "unit  = {}\n"
             "trim  = {}\n"
             "rot90 = {}\n"
             "fp    = {}\n"
             "hdu   = {}".format(*self.shape,
                                 self.gain, self.ron, self.unit, self._trim,
                                 self._rot90, self.fp, self.hdu))
        print(s)
        return

    ###########################
    # to inherit numpy.ndarray
    ###########################
    def __new__(subtype, data, gain=1., ron=0., unit="adu",
                trim=None, rot90=0, header=None, meta=None,
                offset=0, strides=None, order="C", info=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__

        data = np.array(data, dtype=np.float, order=order)

        # substantiate
        ccd = super(CCD, subtype).__new__(subtype, shape=data.shape,
                                          dtype=data.dtype, buffer=data,
                                          offset=offset, strides=strides,
                                          order=order)

        # set info
        ccd.gain = gain
        ccd.ron = ron
        ccd.unit = unit

        ccd._trim = trim
        ccd._rot90 = rot90

        ccd.header = header
        ccd.meta = OrderedDict(header)

        return ccd

    # def __array_finalize__(self, obj):
    #     if obj is None:
    #         return

    # def __array_wrap__(self, out_arr, context=None):
    #     print('In __array_wrap__:')
    #     print('   self is %s' % repr(self))
    #     print('   arr is %s' % repr(out_arr))
    #     # then just call the parent
    #     return super(CCD, self).__array_wrap__(self, out_arr, context)

    ###########################
    # read from fits
    ###########################
    @staticmethod
    def read(fp="", hdu=0, gain=1., ron=0., unit="adu", trim=None, rot90=0):
        if os.path.exists(fp):
            # read fits
            hdu_list = fits.open(fp)

            # get data
            data = hdu_list[hdu].data
            header = hdu_list[hdu].header
            meta = OrderedDict(header)

            # trim
            if trim is not None:
                l, r, t, b = trim
                data = data[l:r + 1, t:b + 1]

            # rotate
            data = np.rot90(data, rot90)

            # initialize CCD
            ccd = CCD(data, gain=gain, ron=ron, unit=unit,
                      trim=trim, rot90=rot90, header=header, meta=meta)
            ccd.fp = fp

            return ccd

        else:
            raise ValueError("@CCD.read(): file not found! [{}]".format(fp))
    
    def trim(self, trim=(0, 2047, 0, 2047)):
        """ trim CCD data

        Parameters
        ----------
        trim : [left, right, top, bottom]
            the trimed section
        """
        l, r, t, b = trim
        trimed_data = self[l:r+1, t:b+1]
        trimed_data.copy_info(self)
        return trimed_data

    def rot90(self, k):
        """ Rotate self by 90 degrees in the counter-clockwise direction. <-\

        Parameters
        ----------
        k : integer
            Number of times the array is rotated by 90 degrees.
        """
        rotated_data = np.rot90(self, k)
        rotated_data.copy_info(self)
        return rotated_data

    ###########################
    # copy option
    ###########################
    def copy_info(self, ccd1):
        """ copy info from an other CCD instance """
        for k in self.extr_attr_list:
            self.__setattr__(k, ccd1.__getattribute__(k))

    def copy(self):
        return np.copy(self)

    ###########################
    # get config
    ###########################
    @property
    def config(self):
        return dict(hdu=self.hdu,
                    gain=self.gain,
                    ron=self.ron,
                    unit=self.unit,
                    trim=self._trim,
                    rot90=self._rot90)

    ###########################
    # arithmetic options
    ###########################
    def subtract(self, ccd1):
        """ ccd2 = self - ccd1 """
        ccd2 = self/ccd1
        ccd2.copy_info(self)
        return ccd2

    def devide(self, ccd1):
        """ ccd2 = self / ccd1 """
        ccd2 = self / ccd1
        ccd2.copy_info(self)
        return ccd2

    @staticmethod
    def combine(ccds, method="median"):
        """ combine ccd frames """
        ccds = CCD(ccds)
        if method == "median":
            return np.median(ccds, axis=0)
        elif method == "mean":
            return np.mean(ccds, axis=0)
        else:
            raise ValueError("@CCD.combine: bad method [{}]".format(method))

    @staticmethod
    def mean(ccds):
        return np.mean(ccds, axis=0)

    @staticmethod
    def median(ccds):
        return np.median(ccds, axis=0)

    @staticmethod
    def std(ccds):
        return np.std(ccds, axis=0)

    def write(self, fp, overwrite=True):
        phdu = fits.PrimaryHDU(data=self, header=fits.Header(self.meta))
        phdu.writeto(fp, overwrite=overwrite)
        return


def test2():
    fp = "/home/cham/PycharmProjects/songcn/twodspec/data/s2_2017-01-13T16-43-05.fits"
    # check image --> consistent with fits.getdata & ccdproc.CCDData.read
    print("---")
    data = fits.getdata(fp)
    print(data)

    print("---")
    from ccdproc import CCDData
    ccddata = CCDData.read(fp, unit="adu")
    print(ccddata)

    print("---")
    for i in range(4):
        ccd = CCD.read(fp, rot90=i, gain=5)
        print(ccd)
    print("---")


def test1():
    fp = "/home/cham/PycharmProjects/songcn/twodspec/data/s2_2017-01-13T16-43-05.fits"
    ccd = CCD.read(fp, rot90=0, gain=5)
    print(ccd)

    print(ccd.info)
    print(ccd.config)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(ccd, origin="upper")


if __name__ == "__main__":

    test1()
    test2()

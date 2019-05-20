#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:24:50 2017

@author: cham


DEPRECATED: 2019-05-20

"""

import re
import numpy as np
from astropy.time import Time


def float_int(value):
    """ to convert value to float then to int, useful when convert string """
    try:
        return np.int(value)
    except ValueError:
        return np.int(np.float(value))


# Just to be consistent with IRAF... I guess this is not useful
class Bgrec():
    """ IRAF background record """

    def __init__(self):
        self._xmin = -10.
        self._xmax = 10.
        self._function = "chebyshev"
        self._order= 1
        self._sample = "-10:-6, 6:10"
        self._naverage = -3
        self._niterate = 0
        self._low_reject = 3.
        self._high_reject = 3.
        self._grow = 0.

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, value):
        self._xmin = np.float(value)

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, value):
        self._xmax = np.float(value)

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        self._function = str(value)

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        assert float_int(value) >= 0
        self._order = float_int(value)

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, value):
        # hard to assert type
        self._sample = value

    @property
    def naverage(self):
        return self._naverage

    @naverage.setter
    def naverage(self, value):
        self._naverage = float_int(value)

    @property
    def niterate(self):
        return self._niterate

    @niterate.setter
    def niterate(self, value):
        self._niterate = float_int(value)

    @property
    def low_reject(self):
        return self._low_reject

    @low_reject.setter
    def low_reject(self, value):
        self._low_reject = np.float(value)

    @property
    def high_reject(self):
        return self._high_reject

    @high_reject.setter
    def high_reject(self, value):
        self._high_reject = np.float(value)

    @property
    def grow(self):
        return self._grow

    @grow.setter
    def grow(self, value):
        self._grow = np.float(value)

    def tostring(self, sep="\n", extra_space=4):

        str_ = []
        # 0. background
        str_.append("background")
        # 1. xmin
        str_.append("    xmin {}".format(self.xmin))
        # 2. xmax
        str_.append("    xmax {}".format(self.xmax))
        # 3. function
        str_.append("    function {}".format(self.function))
        # 4. order
        str_.append("    order {}".format(self.order))
        # 5. sample
        str_.append("    sample {}".format(self.sample))
        # 6. naverage
        str_.append("    naverage {}".format(self.naverage))
        # 7. niterate
        str_.append("    niterate {}".format(self.niterate))
        # 8. low_reject
        str_.append("    low_reject {}".format(self.low_reject))
        # 9. high_reject
        str_.append("    high_reject {}".format(self.high_reject))
        # 10. grow
        str_.append("    grow {}".format(self.grow))

        if extra_space:
            str_ = [" "*extra_space + _ for _ in str_]

        if sep is not None:
            str_ = sep.join(str_)

        return str_


class Cvrec(object):
    """ IRAF curve record """

    def __init__(self):
        self._n_element = 9
        self._function = 1
        self._order = 5
        self._range = (0., 2048.)
        self._coefs = np.zeros((5,), float)

    @property
    def n_element(self):
        return self._n_element

    @n_element.setter
    def n_element(self, value):
        self._n_element = float_int(value)

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, value):
        self._function = str(value)

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = float_int(value)

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, value):
        self._range = np.float(value[0]), np.float(value[1])

    @property
    def coefs(self):
        return self._coefs

    @coefs.setter
    def coefs(self, value):
        self._coefs = np.array(value, dtype=float)
        self.n_element = len(self.coefs) + 4

    def tostring(self, sep="\n", extra_space=4):
        try:
            # check n_elements
            assert float_int(self.n_element) == (len(self.coefs) + 4)
        except AssertionError:
            raise(AssertionError("n_element in Cvrec instance not consistent!"))

        str_ = []
        # 0. n_element
        str_.append("curve {}".format(self.n_element))
        # 1. function
        str_.append("    {}".format(self.function))
        # 2. order
        str_.append("    {}".format(self.order))
        # 3. range
        str_.append("    {}".format(self.range[0]))
        str_.append("    {}".format(self.range[1]))
        # 4. coefs
        for _ in self.coefs:
            str_.append("    {}".format(_))

        if extra_space:
            str_ = [" "*extra_space + _ for _ in str_]

        if sep is not None:
            str_ = sep.join(str_)

        return str_


class Aprec(object):
    """ IRAF aperture definition record """

    def __init__(self):
        self.comment = "Mon 15:29:12 22-May-2017"
        self.time = None
        # availabel format in astropy.time.Time:
        # ['datetime', 'iso', 'isot', 'yday', 'fits', 'byear_str', 'jyear_str']
        #
        # n = time.Time.now()
        # n.replicate(format='datetime').value
        # Out[33]: datetime.datetime(2017, 5, 22, 7, 32, 38, 167897)
        # n.replicate(format='iso').value
        # Out[34]: '2017-05-22 07:32:38.168'
        # n.replicate(format='isot').value
        # Out[35]: '2017-05-22T07:32:38.168'
        # n.replicate(format='yday').value
        # Out[36]: '2017:142:07:32:38.168'
        # n.replicate(format='fits').value
        # Out[37]: '2017-05-22T07:32:38.168(UTC)'
        # n.replicate(format='byear_str').value
        # Out[38]: 'B2017.389'

        self._image = ""  # file path of the image
        self._apID = 0
        self._center = (0., 0.)
        self._beamID = 0
        self._low = (-2047., -5.)
        self._high = (2048., 5.)
        self._axis = 1

        self.bgrec = Bgrec()
        self.cvrec = Cvrec()

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = str(value)

    @property
    def apID(self):
        return self._apID

    @apID.setter
    def apID(self, value):
        self._apID = float_int(value)

    @property
    def beamID(self):
        return self._beamID

    @beamID.setter
    def beamID(self, value):
        self._beamID = float_int(value)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = np.float(value[0]), np.float(value[1])

    @property
    def low(self):
        return self._low

    @low.setter
    def low(self, value):
        self._low = np.float(value[0]), np.float(value[1])

    @property
    def high(self):
        return self._high

    @high.setter
    def high(self, value):
        self._high = np.float(value[0]), np.float(value[1])

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, value):
        assert 1 <= float_int(value) <= 3
        self._axis = float_int(value)

    def tostring(self, comment=None, sep="\n"):

        if comment is None:
            comment = Time.now().replicate(format='fits').value

        str_ = []
        # 1. comment
        str_.append("# {}".format(comment))
        # 2. begin
        str_.append("begin aperture {} {} {} {}".format(
            self.image, self.apID, self.center[0], self.center[1]))
        # 3. image
        str_.append("    image {}".format(self.image))
        # 4. apID
        str_.append("    aperture {}".format(self.apID))
        # 5. beam
        str_.append("    beam {}".format(self.beamID))
        # 6. center
        str_.append("    center {} {}".format(self.center[0], self.center[1]))
        # 7. low
        str_.append("    low {} {}".format(self.low[0], self.low[1]))
        # 8. high
        str_.append("    high {} {}".format(self.high[0], self.high[1]))
        # 9. background
        str_.extend(self.bgrec.tostring(sep=None, extra_space=4))
        # 10. axis
        str_.append("    axis {}".format(self.axis))
        # 11. curve
        str_.extend(self.cvrec.tostring(sep=None, extra_space=4))

        if sep is not None:
            str_ = sep.join(str_)
        return str_

    @staticmethod
    def from_string(str_list):
        """ initiate from string list """
        # find start: the first char is "#"
        start_found = False
        for i in range(len(str_list)):
            if str_list[i][0] == "#":
                start_found = True
                break
        if not start_found:
            raise(ValueError("# doesn't exist in input string list!"))

        # replace special chars
        str_list = [_.replace("\t", " ") for _ in str_list]
        str_list = [_.replace("\n", " ") for _ in str_list]

        # initialize Aprec
        ar = Aprec()

        # start is found
        ar.comment = str_list[i][1:].strip()

        # image, apID, center
        s = re.split(" +", str_list[i + 1].strip())
        assert s[0] == "begin" and s[1] == "aperture"
        ar.image = s[2]
        ar.apID = s[3]
        ar.center = s[4], s[5]

        # confirm image
        s = re.split(" +", str_list[i + 2].strip())
        image = s[1]
        assert ar.image == image

        # confirm apID
        s = re.split(" +", str_list[i + 3].strip())
        assert s[0] == "aperture"
        apID = float_int(s[1])
        assert ar.apID == apID

        # beam
        s = re.split(" +", str_list[i + 4].strip())
        assert s[0] == "beam"
        ar.beamID = float_int(s[1])

        # confirm center
        s = re.split(" +", str_list[i + 5].strip())
        assert s[0] == "center"
        center = np.float(s[1]), np.float(s[2])
        assert ar.center == center

        # low
        s = re.split(" +", str_list[i + 6].strip())
        assert s[0] == "low"
        ar.low = s[1], s[2]

        # high
        s = re.split(" +", str_list[i + 7].strip())
        assert s[0] == "high"
        ar.high = s[1], s[2]

        # BACKGROUND ==============================
        s = re.split(" +", str_list[i + 8].strip())
        assert s[0] == "background"

        # xmin
        s = re.split(" +", str_list[i + 9].strip())
        assert s[0] == "xmin"
        ar.bgrec.xmin = s[1]

        # xmax
        s = re.split(" +", str_list[i + 10].strip())
        assert s[0] == "xmax"
        ar.bgrec.xmax = s[1]

        # function
        s = re.split(" +", str_list[i + 11].strip())
        assert s[0] == "function"
        ar.bgrec.function = s[1]

        # order
        s = re.split(" +", str_list[i + 12].strip())
        assert s[0] == "order"
        ar.bgrec.order = s[1]

        # sample
        s = re.split(" +", str_list[i + 13].strip())
        assert s[0] == "sample"
        ar.bgrec.sample = "".join(s[1:])

        # naverage
        s = re.split(" +", str_list[i + 14].strip())
        assert s[0] == "naverage"
        ar.bgrec.naverage = s[1]

        # niterate
        s = re.split(" +", str_list[i + 15].strip())
        assert s[0] == "niterate"
        ar.bgrec.niterate = s[1]

        # low_reject
        s = re.split(" +", str_list[i + 16].strip())
        assert s[0] == "low_reject"
        ar.bgrec.low_reject = s[1]

        # high_reject
        s = re.split(" +", str_list[i + 17].strip())
        assert s[0] == "high_reject"
        ar.bgrec.high_reject = s[1]

        # grow
        s = re.split(" +", str_list[i + 18].strip())
        assert s[0] == "grow"
        ar.bgrec.grow = s[1]

        # AXIS =====================================
        s = re.split(" +", str_list[i + 19].strip())
        assert s[0] == "axis"
        ar.axis = s[1]

        # CURVE ====================================
        # n_element
        s = re.split(" +", str_list[i + 20].strip())
        assert s[0] == "curve"
        ar.cvrec.n_element = s[1]

        # function
        s = re.split(" +", str_list[i + 21].strip())
        ar.cvrec.function = s[0]

        # order
        s = re.split(" +", str_list[i + 22].strip())
        ar.cvrec.order = s[0]

        # range
        s1 = re.split(" +", str_list[i + 23].strip())
        s2 = re.split(" +", str_list[i + 24].strip())
        ar.cvrec.range = s1[0], s2[0]

        # coefs
        s = [re.split(" +", str_list[_].strip())[0] for _ in range(i+25, i+ar.cvrec.n_element+25-4)]
        ar.cvrec.coefs = s

        return ar


class AprecList():
    """ IRAF aperture list: a list of Aprec instances """

    def __init__(self):
        self._data = []
        self._nap = 0

    @property
    def nap(self):
        return self._nap

    @nap.setter
    def nap(self, value):
        assert np.int(value) == len(self.data)
        self._nap = np.int(value)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        try:
            assert isinstance(value, list)
            for _ in value:
                assert isinstance(_, Aprec)
        except AssertionError as ae:
            # print(value)
            # print(type(_))
            raise(ae)
        self._data = value
        self.nap = len(value)

    @data.getter
    def data(self):
        return self._data

    @staticmethod
    def read(filepath):
        """ read from a text file """
        with open(filepath, "r+") as f:
            s = f.readlines()

        iscomment = [_[0] == "#" for _ in s]
        subcomment = np.where(iscomment)[0]
        print("@AprecList: {} apertures recognized!".format(len(subcomment)))
        arlist = []
        for i in range(len(subcomment) - 1):
            arlist.append(
                Aprec.from_string(s[subcomment[i]:subcomment[i + 1]]))
        try:
            arlist.append(Aprec.from_string(s[subcomment[-1]:]))
        except:
            print("@AprecList: the last aperture not recognized!")
            pass

        al = AprecList()
        al.data = arlist
        return al

    def write(self, filepath):
        """ write to a text file """
        if self.nap < 1:
            print("@AprecList: N(aperture) < 1, nothing has been done!")
            return
        else:
            with open(filepath, "w+") as f:
                for ar in self.data:
                    f.write(ar.tostring())
                    f.write("\n\n")
            print("@AprecList: {} apertures written into file {}".format(
                self.nap, filepath))
            return

    def split(self, fp_fmt):
        """ split an AprecList into single apertures
        
        Parameters
        ----------
        fp_fmt: string
            file path format

        Returns
        -------
        fps

        """

        fps = [fp_fmt.format(iap) for iap in range(self.nap)]
        for iap in range(self.nap):

            al = AprecList()
            al.data = [self.data[iap]]
            al.data[0].apID = 1
            al.write(fps[iap])

        return fps


def _test1():
    """ test read a single aperture """
    with open("/home/cham/PycharmProjects/hrs/ar.txt", 'r') as f:
        s = f.readlines()
    from twodspec.aprec import Aprec
    ar = Aprec.from_string(s)
    print(ar.tostring())


def _test2():
    """ test read a list of apertures """
    from twodspec.aprec import AprecList
    al = AprecList.read("/home/cham/PycharmProjects/hrs/ar.txt")
    al.data[0].tostring()


def _test3():
    """ test read & write a list of apertures """
    from twodspec.aprec import AprecList
    al = AprecList.read("/home/cham/PycharmProjects/hrs/testdata/Dropbox"
                        "/masteraper_20170113_slit8.txt")
    al.write("/home/cham/PycharmProjects/hrs/testdata/Dropbox/"
             "masteraper_20170113_slit8_.txt")


if __name__ == "__main__":
    _test3()

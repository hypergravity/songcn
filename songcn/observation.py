import glob
import re
from collections import OrderedDict

import joblib
import numpy as np
from astropy import table
from astropy.io import fits
from astropy.time import Time

from .config import SONGCN_PARAMS


class SongCalendar(table.Table):
    """
    Initiated from all song observations.
    """

    def __init__(self, *args, **kwargs):
        super(SongCalendar, self).__init__(*args, **kwargs)

    @staticmethod
    def from_dir(root="/Users/cham/projects/song/star_spec", n_jobs=4):
        # glob files in root
        fps = glob.glob(root + "/*/night/raw/*.fits", recursive=True)
        fps.sort()
        print(f"{len(fps)} files found!")
        meta = OrderedDict(
            root=root,
            fps=fps,
        )

        # gather basic info
        basic_info = SongFile.gather_basic_info(fps, n_jobs=4)

        # stats for all IMAGETYP in each day
        date_list = np.unique(basic_info["jd1date"])
        info_list = []
        for date in date_list:
            od = OrderedDict(
                DATE=date,
                N_FILE=sum(basic_info["jd1date"] == date),
            )
            for k in SONGCN_PARAMS["IMAGETYP"]:
                od[k] = sum((basic_info["jd1date"] == date) & (basic_info["IMAGETYP"] == k))
            info_list.append(od)
        sc = SongCalendar(info_list, meta=meta)
        return sc

    def get_one_night(self, date="20191029"):
        night_list = list(self["DATE"])
        if date in night_list:
            basic_info = SongFile.gather_basic_info(self.meta["fps"], n_jobs=4)
            return SongNight(basic_info[basic_info["jd1date"] == date])
        else:
            print("Available DATEs are: ", list(self["DATE"]))
            raise ValueError(f"The given date {date} is not available!")


class SongNight(table.Table):
    def __init__(self, *args, **kwargs):
        super(SongNight, self).__init__(*args, **kwargs)

    # def from_dir(self, dir_path):
    #     """ from a night dir """
    #     pass

    def select(self,
               cond_dict={"IMAGETYP": "THAR", "SLIT": 6},
               method="random",
               n_select=120
               ):
        """ select some images from list

        Parameters
        ----------
        cond_dict: dict
            the dict of colname:value pairs
        method: str, {"all", "random", "top", "bottom"}
            the selection method
        n_select:
            the number of images that will be selected
            if n_images is larger than the number of images matched conditions,
            then n_images is forced to be n_matched

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
            # print("@SONG: no images matched!")
            return []

        sub_match = np.where(ind_match)[0]
        # determine the number of images to select
        n_return = np.min([n_matched, n_select])

        # select according to method
        assert method in {"all", "random", "top", "bottom"}
        sub_rand = np.arange(0, n_matched, dtype=int)
        if method == "all":
            n_return = n_matched
        elif method == "random":
            np.random.shuffle(sub_rand)
            sub_rand = sub_rand[:n_return]
        elif method == "top":
            sub_rand = sub_rand[:n_return]
        elif method == "bottom":
            sub_rand = sub_rand[-n_return:]
        sub_return = sub_match[sub_rand]

        # constructing result to be returned
        return self["path"][sub_return]

    def master_bias(self):
        fps = self.select({"IMAGETYP":"THAR", "SLIT":6})
        return fps

    def master_flat(self):
        fps = self["path"][self["IMAGETYP"] == "FLAT"]
        return fps

    @staticmethod
    def catalog_spectra(file_paths):
        pass

    def get_image(self):
        pass


class SongFile:
    def __init__(self, path='/Users/cham/projects/song/star_spec/20191031/night/raw/s2_2019-10-31T17-22-11.fits'):
        super().__init__()
        m = re.search(
            r"(\d+)/night/raw/"
            r"s(?P<node>\d+)_(?P<yyyy>\d+)-(?P<mm>\d+)-(?P<dd>\d+)T(?P<HH>\d+)-(?P<MM>\d+)-(?P<SS>\d+).fits",
            path)
        header = fits.getheader(path)
        if m is not None:
            self.isregular = True
            self.path = path
            # self.node = m.group("node")
            # self.time = Time(
            #     "{}-{}-{}T{}:{}:{}".format(
            #         m.group("yyyy"), m.group("mm"), m.group("dd"), m.group("HH"), m.group("MM"), m.group("SS")
            #     ),
            #     format="isot"
            # )
        else:
            self.isregular = False
            self.path = path
        self.telescope = header["TELESCOP"]
        self.time = Time(header["JD-MID"], format="jd")
        self.jd1 = int(self.time.jd1)
        self.jd1date = Time(self.jd1, format="jd").to_datetime().strftime("%Y%m%d")

    def __getitem__(self, item):
        """ Keywords can be accessed in a simple way """
        header = fits.getheader(self.path)
        return header[item]

    @property
    def header(self):
        return fits.getheader(self.path)

    @property
    def data(self):
        return fits.getdata(self.path)

    def info(self):
        return fits.info(self.path)

    def basic_dict(self):
        """ Return basic info dict. """
        header = self.header
        return OrderedDict(
            path=self.path,
            isregular=self.isregular,
            telescope=self.telescope,
            jd1=self.jd1,
            jd1date=self.jd1date,
            jdmid=header["JD-MID"],
            SLIT=header["SLIT"],
            I2POS=header["I2POS"],
            IMAGETYP=header["IMAGETYP"],
            EXPTIME=header["EXPTIME"],
            OBJECT=header["OBJECT"],
            OBJNAME=header["OBJ-NAME"],
            OBJRA=header["OBJ-RA"],
            OBJDEC=header["OBJ-DEC"],
        )

    def full_dict(self):
        """ Return full info dict. """
        d = OrderedDict(
            path=self.path,
            isregular=self.isregular,
            telescope=self.telescope,
            jd1=self.jd1,
            jd1date=self.jd1date,
        )
        header = self.header
        for k in header.keys():
            if not k.startswith("---"):
                d[k] = header[k]
        return d

    @staticmethod
    def gather_basic_info(fps, n_jobs=4):
        def get_one_entry(path):
            return SongFile(path).basic_dict()

        return table.Table(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(get_one_entry)(_) for _ in fps
            )
        )

    @staticmethod
    def gather_full_table(fps, n_jobs=4):
        def get_one_entry(path):
            return SongFile(path).full_dict()

        return table.Table(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(get_one_entry)(_) for _ in fps
            )
        )


def test_songfile():
    sf = SongFile()
    print(sf["IMAGETYP"])
    print(sf.header)
    print(sf.data)
    sf.info()
    print(sf.basic_dict())
    print(sf.full_dict())


def test_songcalendar():
    sc = SongCalendar.from_dir()
    sc.pprint_all(align="<")
    sc.show_in_browser()


if __name__ == "__main__":
    test_songfile()

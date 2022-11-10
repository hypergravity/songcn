import os
import re
from astropy.time import Time
from astropy.io import fits


class SongCalendar:
    """
    Initiated from all song observations.
    """

    def __init__(self, root="/Users/cham/projects/song/star_spec"):
        pass


class SongNight:
    def __init__(self, file_list):
        pass

    def from_dir(self, dir_path):
        pass

    def from_mjd(self, root_path):
        pass

    @staticmethod
    def catalog_spectra(file_paths):
        pass

    def get_image(self):
        pass


class SongFile:
    def __init__(self, path='/Users/cham/projects/song/star_spec/20191031/night/raw/s2_2019-10-31T17-22-11.fits'):
        super().__init__()
        self.path = path
        self.node, obstime, _ext = re.split(r"[_.]", os.path.basename(path))
        yyyy, mm, dd, HH, MM, SS = re.split(r"[-T]", obstime)
        self.time = Time(f"{yyyy}-{mm}-{dd}T{HH}:{MM}:{SS}", format="isot")

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


def test_songfile():
    sf = SongFile()
    print(sf["IMAGETYP"])
    print(sf.header)
    print(sf.data)
    sf.info()


if __name__ == "__main__":
    test_songfile()

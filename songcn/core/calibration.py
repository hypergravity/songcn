"""
Wavelength calibration, possibly flux calibration in the future.
"""


class Calibrator:
    def __init__(self):
        pass

    def cal_wave(self):
        pass

    def cal_flux(self):
        raise NotImplementedError("This is not implemented in present.")


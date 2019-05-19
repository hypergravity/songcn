### **songcn**

The SONG-China project data processing pipeline.

*SmoothingSpline* is from https://github.com/wafo-project/pywafo


## structures


** song **
- *extract.py* \
    ??
- *master.py* \
    master for song images
- *song.py*\
    song image collection management
- *thar.py*\
    ThAr wavelength calibration module 

** twodspec **

- *aperture.py* \
    the aperture class
- *aperture_old.py* \
    deprecated
- *aprec.py*\
    interface to IRAF aperture records
- *calibratio.py*\
    wavelength calibration module    
- *ccd.py*\
    basic CCD operations
- *ccdproc_mod.py*\
    modified ccdproc.CCDData class, deprecated
- *config.py*\
    configuration class
- *extract.py*\
    spectral extraction module
- *normalization.py*\
    deprecated
- *pyreduce.py*\
    python version of REDUCE, deprecated
- *scatter.py*\
    scattered-light substraction module
- *stella.py*\
    wrapper of **stella** (Ritter et al. 2014)
- *trace*\
    trace aperture

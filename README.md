## **songcn**

**SONG** stands for **S**tellar **O**bservations **N**etwork **G**roup.

This package, **songcn**, is designed for the [**SONG-China**](http://song.bao.ac.cn/) project.

The affliated **song** package is the SONG-China project data processing pipeline.
The affliated **twodspec** is to provide basic operations for raw 2D spectral data.

## author
Bo Zhang, [bozhang@nao.cas.cn](mailto:bozhang@nao.cas.cn)

## home page
- [https://github.com/hypergravity/songcn](https://github.com/hypergravity/songcn)
- [https://pypi.org/project/songcn](https://pypi.org/project/songcn)

## install
- for the latest **stable** version: `pip install -U songcn`
- for the latest **github** version: `pip install -U git+git://github.com/hypergravity/songcn`

## structures

**song**

- *song.py* \
    song image collection management
- *thar.py* \
    ThAr wavelength calibration module for SONG.
    Loads templates.

**twodspec**

- *aperture.py* \
    the aperture class
- *background.py* \
    background modelling (scattered light substraction)
- *calibration.py* \
    wavelength calibration module    
- *ccd.py* \
    basic CCD operations
- *extract.py* \
    spectral extraction module
- *trace* \
    trace aperture


## acknowledgements

*SmoothingSpline* is from https://github.com/wafo-project/pywafo

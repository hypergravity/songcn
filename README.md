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

## tutorial

```python
# step1: import Song
from song import Song

# step2: initialize Song object by scanning files
s = Song.init(rootdir="/Users/cham/projects/song/star_spec", date="20191030", jdnight=True, n_jobs=-1, verbose=True, subdir="night", node=2)
# usually you can just modify rootdir and date
# rootdir: the root directory
# date: date of the observation
# jdnight: if True, split data at 12:00 next day; if False, only use the data in this directory
# n_jobs: number of jobs for scanning files, -1 for all
# verbose: if True, print information
# subdir: "night"
# node: Danish-->1, Delingha-->2. 

# step3: run the daily pipeline
s.daily(proc_slits='all', sharebias=True, star=True, stari2=False, flati2=False, ipcprofile='default')
# proc_slits: a list of slit names, i.e., [5, 8]. or just "all"
# sharebias: if True, share bias images when processing different slit data
# star, stari2, flati2: if set to True, process corresponding images
# ipcprofile: the ipcluster profile name. if not None, you should type 
#             `ipcluster start --n=8 --profile=default` to start ipcluster 
```

The verbose information:

```buildoutcfg
@SONG: scanning files ...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
[Parallel(n_jobs=-1)]: Done  18 tasks      | elapsed:    0.7s
[Parallel(n_jobs=-1)]: Done 260 out of 260 | elapsed:    1.1s finished
@SONG: 1 unique config found! [5]
SLIT BIAS FLAT STAR THAR
---- ---- ---- ---- ----
   5  120  120   11    9
===========================================
@Song: unique slits are  [5]
===========================================
@SONG: no images matched!
@SONG: no images matched!
@Slit[5]: processing bias ...Done!
@Slit[5]: processing flat ...
@Slit[5]: tracing orders ...
@Aperture: tracing apertures using [naive] method  >>>  51 apertures found!
@Slit[5]: modeling background ...
@Slit[5]: extracting blaze & sensitivity ...Done!
@Slit[5]: cleared tws ...
@Slit[5]: dispatching 9 thar to ipcluster (profile=default, nproc=8) ...
@Slit[5]: dispatching 11 star to ipcluster (profile=default, nproc=8) ...
@Slit[5]: Done!)
saved to files:
========
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-30T20-37-34.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-30T20-39-11.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-30T20-44-31.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-30T20-46-38.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-30T23-36-39.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-30T23-46-46.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-31T00-09-56.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-31T00-12-02.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-31T00-15-42.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-31T00-27-30.fits
/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-31T00-37-37.fits
========
===========================================
DONE!~
===========================================
```

The generated files are:
```
tstar_s2_2019-10-31T00-37-37.fits   # reduced spectrum
tstar_s2_2019-10-31T00-27-30.fits
tstar_s2_2019-10-31T00-15-42.fits
tstar_s2_2019-10-31T00-12-02.fits
tstar_s2_2019-10-31T00-09-56.fits
tstar_s2_2019-10-30T23-46-46.fits
tstar_s2_2019-10-30T23-36-39.fits
tstar_s2_2019-10-30T20-46-38.fits
tstar_s2_2019-10-30T20-44-31.fits
tstar_s2_2019-10-30T20-39-11.fits
tstar_s2_2019-10-30T20-37-34.fits
summary.svg                         # summary figure
20191030_song.dump                  # Song object
20191030_slit5.dump                 # Slit object
```
To access the dump files, use joblib.load() to load data.
To access the reduced spectra,
```python
from astropy.table import Table
spec = Table.read("/Users/cham/projects/song/star_spec/20191030/night/ext/tstar_s2_2019-10-30T20-37-34.fits")
print(spec.colnames)
['blaze', 'bvc', 'err', 'err1', 'err2', 'err_sum', 'exptime', 'jdmid', 'mask', 'flux', 'flux1', 'flux2', 'flux_sum', 'wave', 'wave_rms']
# 'blaze'           [51,2048] blaze function
# 'bvc'             float barycenter velocity correction 
# 'err'             [51,2048] flux error
# 'err1'            [51,2048] optimal extracted 1 flux error
# 'err2'            [51,2048] optimal extracted 2 flux error (clipped)
# 'err_sum'         [51,2048] simple sum flux error
# 'exptime'         float exptime
# 'jdmid'           float mid-jd
# 'mask'            [51,2048] True for bad pixels
# 'flux'            [51,2048] combined flux
# 'flux1'           [51,2048] optimal extracted 1 flux
# 'flux2'           [51,2048] optimal extracted 2 flux
# 'flux_sum'        [51,2048] simple sum flux
# 'wave'            [51,2048] wavelength
# 'wave_rms'        float rms of the ThAr calibration
```
 
The Slit object contains bg, tws etc...
slit.tws has colnames: 
`['fp', 'jdmid', 'exptime', 'wave_init', 'wave_solu', 'tlines', 'nlines', 'rms', 'pf1', 'pf2']`


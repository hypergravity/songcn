import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='songcn',
    version='0.0.1',
    author='Bo Zhang',
    author_email='bozhang@nao.cas.cn',
    description='The SONG-China project.',  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/hypergravity/songcn',
    #packages=setuptools.find_packages(),
    packages=['song', 'twodspec', 'twodspec/extern'],
    license='MIT',
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python :: 3.7",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Scientific/Engineering :: Astronomy"],
    package_dir={'regli': 'regli'},
    package_data={"song": ['calibration/thar_template/*',
                           'calibration/*'],
                  "": ["LICENSE"]},
    include_package_data=True,
    requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'skimage',
              'joblib', 'ccdproc', 'tqdm']
)

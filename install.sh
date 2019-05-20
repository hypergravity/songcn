#!/usr/bin/env bash
rm -rf build
rm -rf dist
rm -rf ./*.egg-info

#python setup.py build_ext --inplace
#python setup.py install

python setup.py sdist
pip install dist/*.tar.gz
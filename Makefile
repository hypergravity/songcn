all: install clean 

clean:
	rm -rf build songcn.egg-info

install:
	pip install .
	rm -rf build songcn.egg-info

upload:
	rm -rf dist/
	python setup.py sdist bdist_wheel
	twine upload dist/*

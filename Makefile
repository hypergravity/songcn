all: install clean 

clean:
	rm -rf build songcn.egg-info

install:
	pip install .

upload:
	rm -rf dist/
	python setup.py sdist bdist_wheel
	twine upload dist/*

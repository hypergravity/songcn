import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [
        req.strip()
        for req in f.readlines()
        if not req.startswith("#") and req.__contains__("==")
    ]

setuptools.setup(
    name="songcn",
    version="0.1.2",
    author="Bo Zhang",
    author_email="bozhang@nao.cas.cn",
    description="The SONG-China data processing pipeline.",  # short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/hypergravity/songcn",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    package_data={
        "": ["LICENSE"],
        "songcn": ["config/*"],
    },
    # include_package_data=True, commented to include data!
    requires=requirements,
)

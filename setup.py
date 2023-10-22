"""Setup file for package."""
#!/user/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data
NAME = "movie-recommender-model"
DESCRIPTION = "Example recommendation model using data from 100 movies project."
URL = "https://github.com/AceDitko/MovieRecommender"
AUTHOR = "Jake Oliver"
REQUIRES_PYTHON = ">=3.11"

long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
print(Path(__file__).resolve().parent)
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR = ROOT_DIR / "regression_model"
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


# What packages are required for this model to be executed?
def list_reqs(fname: str = "requirements.txt"):
    """List contents of requirements file."""
    with open(REQUIREMENTS_DIR / fname, encoding="utf-8") as fd:
        for i in fd.read():
            i
        return fd.read().splitlines()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests")),
    package_data={"regression_model": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: /pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)

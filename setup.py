import pathlib
from setuptools import setup, find_packages
from fcapsy_experiments import __version__, __author__, __email__, __license__

setup(
    name="fcapsy-experiments",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="Package of experiments for fcapsy library.",
    keywords="fca formal concept analysis experiments",
    license=__license__,
    url="https://github.com/mikulatomas/fcapsy_experiments",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "fcapsy",
        "plotly",
        "pandas",
        "binsdpy",
        "sklearn",
        "fuzzycorr",
        "scipy",
        "numpy",
    ],
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

__author__ = 'Tomáš Mikula'
__email__ = 'mail@tomasmikula.cz'
__version__ = '0.1.0'
__license__ = 'MIT license'

with open('README.md') as readme_file:
    readme = readme_file.read()

# Requirements for end-user
requirements = [
    'fcapy']


setup(
    author=__author__,
    author_email=__email__,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Package of experiment for fcapy library.",
    install_requires=requirements,
    license=__license__,
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='fca formal concept analysis experiments',
    name='fcapy_experiments',
    packages=find_packages(
        include=['fcapy_experiments', 'fcapy_experiments.*']),
    url='https://github.com/mikulatomas/fcapy_experiments',
    version=__version__,
    zip_safe=False,
)

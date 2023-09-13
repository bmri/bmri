import os
import urllib.request
from setuptools import setup, Command, find_packages
from setuptools.command.install import install

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(
    name='bmri',
    version='0.0.4',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=requirements
)



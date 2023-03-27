import os
import urllib.request
from setuptools import setup, Command, find_packages
from setuptools.command.install import install

# model_url  = 'https://drive.google.com/uc?export=download&id=1h5-i2D9QdS80WrlDX65WnGl0gwdUqAKV'    
# model_name = 'model_brain_extraction_cGAN_128x128_z128_weights_dl.h5'
# model_dir  = '~/cGAN-BET/saved_models/'

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    


    

setup(
    name='bmri',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=requirements,
#     cmdclass={
#         'install': CustomInstallCommand
#     }
)



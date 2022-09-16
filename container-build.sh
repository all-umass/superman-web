#!/usr/bin/env bash

pip install --upgrade pip

# TODO: manage these with apt instead; see README.md
pip install h5py==2.10.0 matplotlib==3.1.3 pandas==1.0.0 PyWavelets==1.1.1 PyYAML==5.3 tornado==4.4.2

# The superman package from PyPI is missing a file; this is the easiest way to install.
cd ~
git clone git@github.com:all-umass/superman.git
cd superman
pip install -e .

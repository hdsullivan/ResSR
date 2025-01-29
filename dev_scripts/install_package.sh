#!/bin/bash
# This script just installs ResSR along with all requirements
# for the package, demos, and documentation.
# However, it does not remove the existing installation of ResSR.

conda activate ResSR
conda install -c conda-forge gdal
cd ..
pip install -r requirements.txt
pip install -e .
pip install -r demo/requirements.txt
pip install -r docs/requirements.txt 
cd dev_scripts


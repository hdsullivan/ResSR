.. docs-include-ref

ResSR
=====

This is the official implementation of ResSR [1]. 
ResSR is a computationally efficient MSI-SR method that achieves high-quality reconstructions by using a closed-form spectral decomposition along with a spatial residual correction. 
ResSR applies singular value decomposition to identify correlations across spectral bands, uses pixel-wise computation to upsample the MSI, and then applies a residual correction process to correct the high-spatial frequency components of the upsampled bands.  
While ResSR is formulated as the solution to a spatially-coupled optimization problem, we use pixel-wise regularization and derive an approximate closed-form solution, resulting in a pixel-wise algorithm with a dramatic reduction in computation that achieves state-of-the-art reconstructions. 

[1] Duba-Sullivan, H., Reid, E. J., Voisin, S., Bouman, C. A., & Buzzard, G. T. (2024). ResSR: A Computationally Efficient Residual Approach to Super-Resolving Multispectral Images. arXiv preprint arXiv:2408.13225.


Installing
----------
1. Clone or download the repository.

    .. code-block::

        git clone git@code.ornl.gov:sullivanhe/ressr.git

2. Install the conda environment and package. Note that GDAL is a required package and should be installed using conda forge (https://gdal.org/en/stable/download.html). 

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh

    b. Option 2: Manual install

        Create a new conda environment named ``ResSR`` using the following commands:

        .. code-block::

            conda create --name ResSR python=3.9.19
            conda activate ResSR
            conda install -c conda-forge gdal
            pip install -r requirements.txt
            pip install -e .
            pip install -r docs/requirements.txt 

3. Anytime you want to use this package, this ``ResSR`` environment should be activated with the following:

    .. code-block::

        conda activate ResSR


Running Demo(s)
---------------
1. Demo without downloading any data
    This demo will generate and super-resolve a random MSI, enabling repository testing without downloading any data.  
    To run this demo,  navigate to the ``demo/`` directory and run the following command

        .. code-block::

            python random_data_recon.py

2. Demos on APEX Simulated Sentinel-2 MSI
    Before running this demo, you first need to download and uncompress APEX_demo_data.tgz from https://engineering.purdue.edu/~bouman/data_repository/. 
    Once uncompressed, this directory is ~0.6 GB and consists of the APEX simulated dataset originally downloaded from https://github.com/lanha/SupReME.  

    After downloading the data, navigate to the ``demo/`` directory and udpate the ``parameters_apex.yaml`` file with the path to the APEX_demo_data directory.
    To reconstruct the APEX dataset with ResSR, run the following command

        .. code-block::

            python recon_with_ressr.py -opt parameters_apex.yaml

    The results will be saved in the ``demo/results/`` directory. 

    To reconstruct with the exact iterative formulation of ResSR as discussed in Section III.F of the paper, run the following command:

        .. code-block::

            python recon_with_ressr_iterative.py -opt parameters_apex.yaml

3. Demos on Sentinel-2 MSIs
    Before running this demo, you first need to download and uncompress Sentinel-2_demo_data.tgz from https://engineering.purdue.edu/~bouman/data_repository/. 
    Once uncompressed, this directory is ~25 GB and consists of the real Sentinel-2 dataset curated for our experimental results (containing 19 real Sentinel-2 MSIs). 

    After downloading the data, navigate to the ``demo/`` directory and udpate the ``parameters_sentinel2.yaml`` file with the path to the Sentinel-2_demo_data directory.
    To reconstruct the Sentinel-2 dataset with ResSR, run the following command

        .. code-block::

            python recon_with_ressr.py -opt parameters_sentinel2.yaml

    The results will be saved in the ``demo/results/`` directory. 

    To reconstruct with the exact iterative formulation of ResSR as discussed in Section III.F of the paper, run the following command:

        .. code-block::

            python recon_with_ressr_iterative.py -opt parameters_sentinel2.yaml

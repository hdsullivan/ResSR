.. docs-include-ref

ResSR
=====

This is the official implementation of ResSR [1]. 
ResSR is a computationally efficient MSI-SR method that achieves high-quality reconstructions by using a closed-form spectral decomposition along with a spatial residual correction. 
ResSR applies singular value decomposition to identify correlations across spectral bands, uses pixel-wise computation to upsample the MSI, and then applies a residual correction process to correct the high-spatial frequency components of the upsampled bands.  
While ResSR is formulated as the solution to a spatially-coupled optimization problem, we use pixel-wise regularization and derive an approximate closed-form solution, resulting in a pixel-wise algorithm with a dramatic reduction in computation that achieves state-of-the-art reconstructions. 

[1] Duba-Sullivan, H., Reid, E. J., Voisin, S., Bouman, C. A., & Buzzard, G. T. (2024). ResSR: A Residual Approach to Super-Resolving Multispectral Images. arXiv preprint arXiv:2408.13225.


Installing
----------
1. *Clone or download the repository:*

    .. code-block::

        git clone git@code.ornl.gov:sullivanhe/ressr.git

2. Install the conda environment and package

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh

    b. Option 2: Manual install

        1. *Create conda environment:*

            Create a new conda environment named ``ResSR`` using the following commands:

            .. code-block::

                conda create --name ResSR python=3.9.19
                conda activate ResSR
                conda install -c conda-forge gdal
                pip install -r requirements.txt
                pip install -e .
                pip install -r demo/requirements.txt
                pip install -r docs/requirements.txt 

            Anytime you want to use this package, this ``ResSR`` environment should be activated with the following:

            .. code-block::

                conda activate ResSR


        2. *Install ResSR package:*

            Navigate to the main directory ``ResSR/`` and run the following:

            .. code-block::

                pip install .

            To allow editing of the package source while using the package, use

            .. code-block::

                pip install -e .


Running Demo(s)
---------------

    To run the demo(s) provided for ResSR, you first need to download the necessary data.
    
    After downloading the data, navigate to the ``demo/`` directory and udpate the ``parameters.yaml`` file with the path to the data directory.
    To run reconstruct all images with the data directory with ResSR, run the following command

        .. code-block::

            python recon_with_ressr.py -opt parameters.yaml
    
    The results will be saved in the ``demo/results/`` directory. 

    To reconstruct with the exact iterative formulation of ResSR as discussed in Section III.F of the paper, run the following command:

        .. code-block::

            python recon_with_ressr_iterative.py -opt parameters.yaml

    If you would like to run a demo without downloading any data, you can run it on a random image by running the following command:

        .. code-block::

            python random_data_recon.py -opt parameters.yaml -random
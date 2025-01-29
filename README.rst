.. docs-include-ref

ResSR
=====

ResS is a computationally efficient MSI-SR method that achieves high-quality reconstructions by using a closed-form spectral decomposition along with a spatial residual correction. 
ResSR applies singular value decomposition to identify correlations across spectral bands, uses pixel-wise computation to upsample the MSI, and then applies a residual correction process to correct the high-spatial frequency components of the upsampled bands.  
While ResSR is formulated as the solution to a spatially-coupled optimization problem, we use pixel-wise regularization and derive an approximate closed-form solution, resulting in a pixel-wise algorithm with a dramatic reduction in computation that achieves state-of-the-art reconstructions. 

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

    To run the demo(s) provided for ResSR, navigate to the ``demo/`` directory and run the following command

        .. code-block::

            python demo/recon_with_ressr.py -opt parameters.yaml

    The demo script will read the parameters from the ``parameters.yaml`` file and run the ResSR algorithm on the provided data. 
    You can modify the parameters in the ``demos/parameters.yaml`` file to test different configurations.
    The results will be saved in the ``demo/results/`` directory.
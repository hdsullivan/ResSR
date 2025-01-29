============
Installation 
============

The ``ResSR`` package currently is only available to download and install from source through GitHub.


Downloading and installing from source
-----------------------------------------

1. Download the source code:

  In order to download the python code, move to a directory of your choice and run the following two commands.

    | ``git clone https://github.com/cabouman/ResSR.git``
    | ``cd ResSR``


2. Create a Virtual Environment:

  It is recommended that you install to a virtual environment.
  If you have Anaconda installed, you can run the following:

    | ``conda create --name ResSR python=ResSR``
    | ``conda activate ResSR``

  Install the dependencies using:

    ``pip install -r requirements.txt``

  Install the package using:

    ``pip install .``

  or to edit the source code while using the package, install using

    ``pip install -e .``

  Now to use the package, this ``ResSR`` environment needs to be activated.


3. Install:

You can verify the installation by running ``pip show ResSR``, which should display a brief summary of the packages installed in the ``ResSR`` environment.
Now you will be able to use the ``ResSR`` python commands from any directory by running the python command ``import ResSR``.


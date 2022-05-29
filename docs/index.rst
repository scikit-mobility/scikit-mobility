===================
Scikit-mobility
===================

scikit-mobility is a library for human mobility analysis in Python. The library allows to:

* represent trajectories and mobility flows with proper data structures, TrajDataFrame and FlowDataFrame.

* manage and manipulate mobility data of various formats (call detail records, GPS data, data from Location Based Social Networks, survey data, etc.);

* extract human mobility metrics and patterns from data, both at individual and collective level (e.g., length of displacements, characteristic distance, origin-destination matrix, etc.)

* generate synthetic individual trajectories using standard mathematical models (random walk models, exploration and preferential return model, etc.)

* generate synthetic mobility flows using standard migration models (gravity model, radiation model, etc.)

* assess the privacy risk associated with a mobility dataset



Installation
============

.. note::
  Full instructions to install the library are available in the `scikit-mobilty repository <https://github.com/scikit-mobility/scikit-mobility>`_.


Installation with pip (python >= 3.7 required)
-----------------------------------------------

#. Create an environment `skmob`

   .. code-block:: console

     python3 -m venv skmob

#. Activate

   .. code-block:: console

     source skmob/bin/activate

#. Install skmob

   .. code-block:: console

     pip install scikit-mobility

#. OPTIONAL to use `scikit-mobility` on the jupyter notebook

Activate the virutalenv:

   .. code-block:: console

     source skmob/bin/activate

Install jupyter notebook:

   .. code-block:: console

     pip install jupyter

Run jupyter notebook

   .. code-block:: console

     jupyter notebook

(Optional) install the kernel with a specific name

   .. code-block:: console

     ipython kernel install --user --name=skmob


Installation with conda - miniconda
-----------------------------------------------

#. Create an environment `skmob` and install pip

   .. code-block:: console

     conda create -n skmob pip python=3.7 rtree

#. Activate

   .. code-block:: console

     conda activate skmob

#. Install skmob

   .. code-block:: console

     conda install -c conda-forge scikit-mobility

#. OPTIONAL to use `scikit-mobility` on the jupyter notebook

Install the kernel

   .. code-block:: console

     conda install jupyter -c conda-forge

Open a notebook and check if the kernel `skmob` is on the kernel list. If not, run the following:

On Mac and Linux

   .. code-block:: console

     env=$(basename `echo $CONDA_PREFIX`)
     python -m ipykernel install --user --name "$env" --display-name "Python [conda env:"$env"]"

On Windows

   .. code-block:: console
     python -m ipykernel install --user --name skmob --display-name "Python [conda env: skmob]"


You may run into dependency issues if you try to import the package in Python. If so, try installing the following packages as followed.

.. code-block:: console

  conda install -n skmob pyproj urllib3 chardet markupsafe


Known Issues
^^^^^^^^^^^^

the installation of package rtree could not work with pip within a conda environment. If so, try

.. code-block:: console

  pip install "rtree>=0.8,<0.9"

or install rtree with conda

.. code-block:: console

  conda install rtree


.. warning::
  scikit-mobility is an ongoing open-source project created by the research community. The library is in its first BETA release, as well as the documentation. In the case you find errors, or you simply have suggestions, please open an issue in the repository. We would love to hear from you!

.. toctree::
   :caption: API Reference
   :maxdepth: 2

   reference/data_structures
   reference/preprocessing
   reference/measures
   reference/models
   reference/io
   reference/privacy
   reference/data

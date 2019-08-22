===================
Installation
===================

scikit-mobility is a library for human mobility analysis in Python. The library allows to:

* represent trajectories and mobility flows with proper data structures, TrajDataFrame and FlowDataFrame.

* manage and manipulate mobility data of various formats (call detail records, GPS data, data from Location Based Social Networks, survey data, etc.);

* extract human mobility metrics and patterns from data, both at individual and collective level (e.g., length of displacements, characteristic distance, origin-destination matrix, etc.)

* generate synthetic individual trajectories using standard mathematical models (random walk models, exploration and preferential return model, etc.)

* generate synthetic mobility flows using standard migration models (gravity model, radiation model, etc.)

* assess the privacy risk associated with a mobility dataset

.. note::
   To install the library, please follow the guide in the `scikit-mobilty repository <https://github.com/scikit-mobility/scikit-mobility>`_. 

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


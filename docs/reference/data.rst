============
Data
============


The data module of scikit-mobility provides users with an easy way to: 1) Download ready-to-use mobility data (e.g., trajectories, flows, spatial tessellations, and auxiliary data); 2) Load and transform the downloaded dataset into standard skmob structures (TrajDataFrame, GeoDataFrame, FlowDataFrame, DataFrame); 3) Allow developers and contributors to add new datasets to the library.

.. currentmodule:: skmob.data

.. autosummary::

	skmob.data.load.list_datasets
	skmob.data.load.load_dataset
	skmob.data.load.get_dataset_info

.. automodule:: skmob.data.load
	:members: list_datasets, load_dataset, get_dataset_info

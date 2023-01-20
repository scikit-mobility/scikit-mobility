import glob
import importlib
import json
import os
from abc import ABC, abstractmethod

import pooch
from geopandas import GeoDataFrame
from pandas import DataFrame
from pooch import HTTPDownloader, Untar, Unzip

import skmob
from skmob.core.flowdataframe import FlowDataFrame
from skmob.core.trajectorydataframe import TrajDataFrame


class DatasetBuilder(ABC):
    def __init__(self):
        module = self.__class__.__name__
        info_ds = get_dataset_info(module)
        self.dataset_info = info_ds

    @abstractmethod
    def prepare(self, full_path_files):
        pass


def _skmob_downloader(url, known_hash, download_format=None, auth=(), show_progress=False):

    if download_format == "zip":
        processor = Unzip()
    elif download_format == "tar":
        processor = Untar()
    else:
        processor = None

    download_auth = HTTPDownloader(auth=auth, progressbar=show_progress)

    full_path_files = pooch.retrieve(
        url=url, path=pooch.os_cache("skmob_data"), known_hash=known_hash, processor=processor, downloader=download_auth
    )

    if type(full_path_files) != list:
        full_path_files = [full_path_files]

    return full_path_files


def load_dataset(name, drop_columns=False, auth=None, show_progress=False):

    """Load dataset

    Load one of the datasets that are present in the repository of scikit-mobility.

    Parameters
    ----------
    name: str
        the name of the dataset to load (e.g., foursquare_nyc)

    drop_columns: boolean, optional
        whether to keep additional columns when returning TrajDataFrame or FlowDataFrame object. The default is False.

    auth: (str, str), optional
        pair of strings (user, psw) used when the dataset loading requires an authentication. The default is None.

    show_progress: boolean, optional
        if True, show a progress bar. The default is False.

    Returns
    ----------
    TrajDataFrame/FlowDataFrame/GeoDataFrame/DataFrame
        an object containing the downloaded dataset

    Examples
    --------
    >>> import skmob
    >>> from skmob.data.load import load_dataset, list_datasets
    >>>
    >>> tdf_nyc = load_dataset("foursquare_nyc", drop_columns=True)
    >>> print(tdf_nyc.head())
       uid        lat        lng                  datetime
    0  470  40.719810 -74.002581 2012-04-03 18:00:09+00:00
    1  979  40.606800 -74.044170 2012-04-03 18:00:25+00:00
    2   69  40.716162 -73.883070 2012-04-03 18:02:24+00:00
    3  395  40.745164 -73.982519 2012-04-03 18:02:41+00:00
    4   87  40.740104 -73.989658 2012-04-03 18:03:00+00:00
    """

    # check parameters correctness

    if type(name) is not str:
        raise ValueError("The argument `name` must be a string.")
    if type(drop_columns) is not bool:
        raise ValueError("The argument `drop_columns` must be a boolean.")
    if auth is not None:
        if len(auth) != 2:
            raise ValueError("The argument `auth` must have length 2.")
        else:
            if type(auth[0]) != str or type(auth[1]) != str:
                raise ValueError("The argument `auth` must be a pair of strings.")
    if type(show_progress) is not bool:
        raise ValueError("The argument `show_progress` must be a boolean.")

    if not name.endswith(".py"):
        name = name + ".py"
    short_name = name[:-3]

    # import the module to download & prepare the dataset
    try:
        module = importlib.import_module("." + short_name, "skmob.data.datasets." + short_name)
    except ModuleNotFoundError:
        raise ValueError("Dataset name not found. Please use `list_datasets()` to list all the available datasets.")
        return

    # get the main class
    dataset_class = getattr(module, short_name)
    dataset_instance = dataset_class()

    # retrieve the dataset info (url, hash, license, etc)
    dataset_info = dataset_instance.dataset_info

    hash_value = None if dataset_info["hash"] == "" else dataset_info["hash"]

    if dataset_info["auth"] == "yes":
        if auth is None or len(auth) != 2:
            raise ValueError("`auth` should be a pair (username, password) used for the authentication.")
    else:
        auth = ()

    # download the dataset (if not in the cache)
    full_path_files = _skmob_downloader(
        dataset_info["url"],
        hash_value,
        auth=auth,
        download_format=dataset_info["download_format"],
        show_progress=show_progress,
    )

    # prepare the dataset
    dataset = dataset_instance.prepare(full_path_files)

    if type(dataset) is TrajDataFrame and drop_columns:
        dataset = dataset[["uid", "lat", "lng", "datetime"]]

    # insert the dataset information in the metadata variable _info
    if type(dataset) is TrajDataFrame:
        dataset._info = dataset_info
    elif type(dataset) is GeoDataFrame:
        dataset._info = dataset_info
    elif type(dataset) is DataFrame:
        dataset._info = dataset_info
    elif type(dataset) is FlowDataFrame:
        dataset._info = dataset_info

    # delete the instance
    del dataset_instance

    return dataset


def list_datasets(details=False, data_types=None):

    """List datasets

    List all the names of the datasets available in the data module of scikit-mobility.

    Parameters
    ----------
    details: boolean
        whether to return the full details of the dataset instead of the name only. The default is False.

    data_types: list of string, optional
        specify which dataset types to show. The default is None.


    Returns
    ----------
    an object listing the available datasets

    Examples
    --------
    >>> import skmob
    >>> from skmob.data.load import list_datasets
    >>>
    >>> list_datasets()
    ['flow_foursquare_nyc',
     'foursquare_nyc',
     'nyc_boundaries',
     'parking_san_francisco',
     'taxi_san_francisco']
    """

    if type(details) is not bool:
        raise ValueError("The argument `details` must be a boolean.")
    if data_types is not None:
        if not all(isinstance(item, str) for item in data_types):
            raise ValueError("The argument `data_types` must be a list of strings.")

    path_datasets = os.path.join(os.path.dirname(skmob.__file__), "data", "datasets")

    directories = glob.glob(os.path.join(path_datasets, "*"))
    name_datasets = []

    for d in directories:
        if "__pycache__" not in d and "__init__" not in d:
            if os.path.isdir(d):
                name_datasets.append(d.split(path_datasets)[1].replace("\\", "").replace("/", ""))

    if not details and data_types is None:
        return name_datasets
    else:

        if type(data_types) == str:
            data_types = [data_types]

        dict_datasets = {}

        for short_name in name_datasets:

            # retrieve the dataset info (url, hash, license, etc)
            dataset_info = get_dataset_info(short_name)

            # if data_types=None or the data_type info match the data_types add the dataset
            if data_types is not None:
                try:
                    d_type = dataset_info["data_type"]
                    if d_type in data_types:
                        dict_datasets[short_name] = dataset_info
                except KeyError:
                    return

            else:
                dict_datasets[short_name] = dataset_info

        if details:
            return dict_datasets
        else:
            return list(dict_datasets)


def get_dataset_info(name):

    """Get dataset info

    It returns the dataset information stored in the JSON file associated with the dataset.

    Parameters
    ----------
    name: str
        the name of the dataset of which to return the information

    Returns
    ----------
    dict:
        the information stored in the JSON file associated with the dataset

    Examples
    --------
    >>> import skmob
    >>> from skmob.data.load import get_dataset_info
    >>>
    >>> get_dataset_info("foursquare_nyc")
    {'name': 'Foursquare_NYC',
     'description': 'Dataset containing the Foursquare checkins of individuals moving in New York City',
     'url': 'http://www-public.it-sudparis.eu/~zhang_da/pub/dataset_tsmc2014.zip',
     'hash': 'cbe3fdab373d24b09b5fc53509c8958c77ff72b6c1a68589ce337d4f9a80235b',
     'auth': 'no',
     'data_type': 'trajectory',
     'download_format': 'zip',
     'sep': '\t',
     'encoding': 'ISO-8859-1'}
    """

    path_info = os.path.join(os.path.dirname(skmob.__file__), "data", "datasets", name, name + ".json")

    try:
        f = open(path_info)
    except FileNotFoundError:
        print("Missing .json file")
        return -1

    info_ds = json.load(f)

    return info_ds

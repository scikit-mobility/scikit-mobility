import pooch
import pandas
import importlib
import glob
import json
from abc import ABC, abstractmethod
from pooch import *
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.core.flowdataframe import FlowDataFrame
from geopandas import GeoDataFrame
from pandas import DataFrame


'''
Data types:
- trajectory
- flow
- tessellation
- auxiliar
'''


class DatasetBuilder(ABC):
    
    def __init__(self):              
        module = self.__class__.__name__
        info_ds = get_dataset_info(module)
        self.dataset_info = info_ds
    
    @abstractmethod
    def prepare(self, full_path_files):
        pass
    

def skmob_downloader(url, known_hash, download_format=None, auth=(), show_progress=False):
    
    if download_format=='zip': 
        processor = Unzip()
    elif download_format=='tar':
        processor = Untar()
    else:
        processor = None
    
    download_auth = HTTPDownloader(auth=auth, progressbar=show_progress)
    
    full_path_files = pooch.retrieve(url=url,
                                    path=pooch.os_cache("skmob"),
                                    known_hash=known_hash,
                                    processor=processor,
                                    downloader=download_auth)

    if type(full_path_files) != list:
        full_path_files = [full_path_files]
        
    return full_path_files
    

'''
load_dataset

parameters
----------
name: string 
    the name of the dataset to load (e.g., foursquare_nyc)
drop_columns: bool
    whether to keep additional columns when returning TrajectoryDataFrame/FlowDataFrame object. The default is False
auth: pair of strings in the form (username, password). The default is None.
show_progress: boolean – if True, show a progress bar. The default is True.

'''    

def load_dataset(name, drop_columns=False, auth=None, show_progress=False):
    
    if not name.endswith(".py"):
        name = name + ".py"
    short_name = name[:-3]

    # import the module to download & prepare the dataset
    try:
        module = importlib.import_module('.'+short_name, 'skmob.data.datasets.'+short_name)
    except ModuleNotFoundError:
        print("Dataset module not found")
        return    

    # get the main class
    dataset_class = getattr(module, short_name)
    dataset_instance = dataset_class()
    
    #retrieve the dataset info (url, hash, license, etc)
    dataset_info = dataset_instance.dataset_info
    
    hash_value = None if dataset_info['hash'] == '' else dataset_info['hash']
    
    if dataset_info['auth'] == "yes": 
        if auth == None or len(auth) != 2:
            raise ValueError("`auth` should be a pair (username, password) used for the authentication.")
    else:
        auth = ()
 
    #download the dataset (if not in the cache)
    full_path_files = skmob_downloader(dataset_info['url'], hash_value, auth=auth, download_format=dataset_info['download_format'],  show_progress=show_progress)
    
    #prepare the dataset
    dataset = dataset_instance.prepare(full_path_files)
    
    if type(dataset) is TrajDataFrame and drop_columns:
        dataset = dataset[['uid','lat','lng','datetime']]
        
    #insert the dataset information in the _metadata variable _info
    if type(dataset) is TrajDataFrame:
        dataset._info = dataset_info
    elif type(dataset) is GeoDataFrame:
        dataset._info = dataset_info
    elif type(dataset) is DataFrame:
        dataset._info = dataset_info
    elif type(dataset) is FlowDataFrame:
        dataset._info = dataset_info
        #dataset.attrs['_info'] = dataset_info
        
    #delete the instance
    del dataset_instance
    
    return dataset



def list_datasets(details=False, data_types=None):
    
    path_datasets = "./skmob/data/datasets"

    directories = glob.glob(path_datasets+"/*/")
    name_datasets = []
    
    for d in directories:
        if "__pycache__" not in d and  "__init__" not in d:
            name_datasets.append(d.split(path_datasets)[1].replace("\\",""))
                        
    if not details and data_types is None:
        return name_datasets
    else:
        
        if type(data_types) == str:
               data_types = [data_types]
        
        dict_datasets = {}
        
        for short_name in name_datasets:
            
            #retrieve the dataset info (url, hash, license, etc)
            dataset_info = get_dataset_info(short_name)
            
            #if data_types=None or the data_type info match the data_types add the dataset
            if data_types is not None: 
                try: 
                    d_type = dataset_info['data_type']
                    if d_type in data_types:
                        dict_datasets[short_name] = dataset_info
                except KeyError:
                    return;
                    
            else:
                dict_datasets[short_name] = dataset_info
                
        if details:
            return dict_datasets
        else:
            return list(dict_datasets)

        
def get_dataset_info(module_name):
    
    path_info = "./skmob/data/datasets/"+module_name+"/"+module_name+".json"
    
     #read the info stored in the .json
    try:
        f = open(path_info)
    except FileNotFoundError:
        print("Missing .json file")
        return -1

    info_ds = json.load(f)
    
    return info_ds
     
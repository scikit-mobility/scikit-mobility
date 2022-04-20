import skmob
import pandas
from skmob.data.load import DatasetBuilder


'''
*** Scikit-Mobility Pre-Processing dataset script (DPS) ***

'''


class dataset_name(DatasetBuilder): 
    
    # The constructor of the parent class (DatasetBuilder) is called
    # automatically, so the explicit call can be omitted
    
    '''
    `prepare(self, f_names)`.
    Assume that the paths of the files downloaded at the URL specified in the JSON file 
    are described by the argument `f_names` (represented as a list of strings) then:
        1. load the dataset
        2. pre-process it if necessary (e.g., adjust the timezone or delete/add something)
        3. convert it in the correct skmob format:
            trajectory -> TrajDataFrame
            flow -> FlowDataFrame
            shape -> GeoDataFrame
            auxiliar -> DataFrame
        4. return the dataset in the skmob format
    '''
    
    def prepare(self, f_names):
        pass
    
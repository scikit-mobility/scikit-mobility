import pandas
import skmob
from datetime import timedelta
from datetime import datetime
from skmob.data.load import DatasetBuilder


'''
*** Scikit-Mobility Pre-Processing dataset script (DPS) ***

'''


class taxi_san_francisco(DatasetBuilder):
    
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

        # keep only the filename of one taxi, e.g., new_abboip.txt
        fs = [f for f in f_names if 'new_abboip.txt' in f]

        # adjust the dataset in order to obtain a TrajDataFrame
        
        # to convert the datetime from UNIX to UTC
        mydateparser = lambda x: pandas.to_datetime(x, unit='s') + timedelta(minutes=-7*60)
        
        
        #read data
        raw_data = pandas.read_csv(fs[0], sep=self.dataset_info['sep'], 
                                   encoding=self.dataset_info['encoding'],
                                   names=['latitude', 'longitude', 'occupancy', 'time'],
                                   parse_dates=['time'], date_parser=mydateparser,
                                   header=None)
        
        #add the ID as a column
        raw_data['user_id'] = ['abboip']*len(raw_data)
        
        # convert the datetime
        tdf_dataset = skmob.TrajDataFrame(raw_data, latitude='latitude', 
                                          longitude='longitude', 
                                          user_id='user_id', 
                                          datetime='time')
        

        return tdf_dataset

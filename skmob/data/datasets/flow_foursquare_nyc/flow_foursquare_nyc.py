import pandas
import skmob
from skmob.data.load import DatasetBuilder
from skmob.tessellation import tilers



'''
*** Scikit-Mobility Pre-Processing dataset script (DPS) ***

'''


class flow_foursquare_nyc(DatasetBuilder):
    
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

        # keep only the filename of the NYC dataset
        fs = [f for f in f_names if 'TSMC2014_NYC.txt' in f]

        #read data
        raw_data = pandas.read_csv(fs[0], sep=self.dataset_info['sep'], 
                                   encoding=self.dataset_info['encoding'], 
                                   header=None)

        # create the trajdataframe
        tdf_dataset = skmob.TrajDataFrame(raw_data, latitude=4, 
                                          longitude=5, user_id=0, 
                                          datetime=7)

        # obtain a spatial tessellation of NYC using scikit-mobility
        tessellation = tilers.tiler.get('squared', base_shape="New York City", meters=4000)
        
        # create the flowdataframe
        fdf = tdf_dataset.to_flowdataframe(tessellation=tessellation, self_loops=True)

        return fdf

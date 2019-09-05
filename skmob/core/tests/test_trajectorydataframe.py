import skmob
import pandas as pd
from datetime import datetime
from skmob.utils.constants import UID, DATETIME, LATITUDE, LONGITUDE


EXPECTED_NUM_OF_COLUMNS_IN_TDF = 4


class TestTrajectoryDataFrame:

    def setup_method(self):
        self.default_data_list = [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
                                  [1, 39.984198, 116.319322, '2008-10-23 13:53:06'],
                                  [1, 39.984224, 116.319402, '2008-10-23 13:53:11'],
                                  [1, 39.984211, 116.319389, '2008-10-23 13:53:16']]
        self.default_data_df = pd.DataFrame(self.default_data_list, columns=['user', 'latitude', 'lng', 'hour'])
        self.default_data_dict = self.default_data_df.to_dict(orient='list')

    def perform_default_asserts(self, tdf):
        assert tdf._is_trajdataframe()
        assert tdf.shape == (4, EXPECTED_NUM_OF_COLUMNS_IN_TDF)
        assert tdf[UID][0] == 1
        assert tdf[DATETIME][0] == datetime(2008, 10, 23, 13, 53, 5)
        assert tdf[LATITUDE][0] == 39.984094
        assert tdf[LONGITUDE][3] == 116.319389

    def test_tdf_from_list(self):
        tdf = skmob.TrajDataFrame(self.default_data_list, latitude=1, longitude=2, datetime=3, user_id=0)
        self.perform_default_asserts(tdf)
        print(tdf.head())  # raised TypeError: 'BlockManager' object is not iterable

    def test_tdf_from_df(self):
        tdf = skmob.TrajDataFrame(self.default_data_df, latitude='latitude', datetime='hour', user_id='user')
        self.perform_default_asserts(tdf)

    def test_tdf_from_dict(self):
        tdf = skmob.TrajDataFrame(self.default_data_dict, latitude='latitude', datetime='hour', user_id='user')
        self.perform_default_asserts(tdf)

    def test_tdf_from_csv_file(self):
        tdf = skmob.TrajDataFrame.from_file('../../../tutorial/data/geolife_sample.txt.gz', sep=',')
        assert tdf._is_trajdataframe()
        assert tdf.shape == (217653, EXPECTED_NUM_OF_COLUMNS_IN_TDF)
        assert list(tdf[UID].unique()) == [1, 5]

    def test_timezone_conversion(self):
        tdf = skmob.TrajDataFrame(self.default_data_df, latitude='latitude', datetime='hour', user_id='user')
        tdf.timezone_conversion(from_timezone='Europe/London', to_timezone='Europe/Berlin')
        assert tdf[DATETIME][0] == pd.Timestamp('2008-10-23 14:53:05')

    def test_slicing_a_tdf_returns_a_tdf(self):
        tdf = skmob.TrajDataFrame(self.default_data_df, latitude='latitude', datetime='hour', user_id='user')
        assert type(tdf) == type(tdf[tdf[UID] == 1][:1])

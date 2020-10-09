import shapely
import skmob
import pandas as pd
from datetime import datetime
from skmob.utils.constants import UID, DATETIME, LATITUDE, LONGITUDE
import geopandas as gpd

EXPECTED_NUM_OF_COLUMNS_IN_FDF = 3
ORIGIN = '36001'
DESTINATION = ['36001', '36005', '36007', '36017', '36019', '36021', '36023',
       '36025', '36027', '36029', '36031', '36033', '36035', '36039',
       '36043']


class TestTrajectoryDataFrame:

    def setup_method(self):
        url_tess = '../../../tutorial/data/NY_counties_2011.geojson'

        self.tessellation = gpd.read_file(url_tess).rename(columns={'tile_id': 'tile_ID'})

        self.fdf = skmob.FlowDataFrame.from_file("../../../tutorial/data/NY_commuting_flows_2011.csv",
                                            tessellation=self.tessellation,
                                            tile_id='tile_ID',
                                            sep=",")
        self.fdf = self.fdf[:15]

    def test_tdf_from_df(self):
        fdf_local = skmob.FlowDataFrame.from_file("../../../tutorial/data/NY_commuting_flows_2011.csv",
                                            tessellation=self.tessellation,
                                            tile_id='tile_ID',
                                            sep=",")
        assert(len(fdf_local) == 1954)
        assert(fdf_local.shape == (1954,EXPECTED_NUM_OF_COLUMNS_IN_FDF))
        assert(isinstance(fdf_local, skmob.core.flowdataframe.FlowDataFrame))
        assert(fdf_local._is_flowdataframe())

    def test_get_flow(self):
        expected_flows = [121606,      5,     29,     11,     30,    728,     38,      6,
          183,     31,     10,     32,    105,   1296,      3]

        assert (len(expected_flows) == len(DESTINATION))

        for i in range(len(expected_flows)):
            assert expected_flows[i] == self.fdf.get_flow(ORIGIN, DESTINATION[i])

    def test_get_geometry(self):
        for i in range(len(DESTINATION)):
            assert (isinstance(self.fdf.get_geometry(DESTINATION[i]), shapely.geometry.polygon.Polygon))

    def test_origin(self):
        assert(self.fdf.origin.unique()[0] == ORIGIN)
        assert(len(self.fdf.origin.unique()) == 1)

    def test_destination(self):
        assert(len(list(self.fdf.destination)) == len(DESTINATION))

        for i in range(len(DESTINATION)):
            assert(self.fdf.destination[i] == DESTINATION[i])

    def test_has_flow_columns(self):
        assert (self.fdf._has_flow_columns())

import skmob
from skmob.utils import utils, constants
import geopandas as gpd
from skmob.models import Radiation
import numpy as np
import math

class TestRadiation:

    def setup_method(self):
        url_tess = '../../../tutorial/data/NY_counties_2011.geojson'

        self.tessellation = gpd.read_file(url_tess).rename(columns={'tile_id': 'tile_ID'})

        self.fdf = skmob.FlowDataFrame.from_file("../../../tutorial/data/NY_commuting_flows_2011.csv",
                                                 tessellation=self.tessellation,
                                                 tile_id='tile_ID',
                                                 sep=",")

        tot_outflows = self.fdf[self.fdf['origin'] != self.fdf['destination']].groupby(by='origin', axis=0)[
            'flow'].sum().fillna(
            0).values
        self.tessellation[constants.TOT_OUTFLOW] = tot_outflows

    def test_radiation(self):
        radiation = Radiation()

        np.random.seed(0)
        rad_flows = radiation.generate(self.tessellation,
                                       tile_id_column='tile_ID',
                                       tot_outflows_column='tot_outflow',
                                       relevance_column='population',
                                       out_format='flows_sample')

        expecter_36019 = [11648,  4232,  5598,  1596,   117,  1017,   354,   701,  1411,
                         270,   408,   135,   341,   410,   189,   287,    79,    23,
                          42,    38,    30,    37,   203,    18,    15,    37,    10,
                          23,    45,    69,     5,    41,    20,     9,     6,   100,
                           8,     4,    18,    34,     5,     3,     3,    73,    13,
                           7,     3,     1,    56,    48,     3,    43,    21,     1,
                          28,    38,     4,     3]

        output_36019 = rad_flows[rad_flows.origin == '36019']['flow'].values

        print(len(expecter_36019))
        print(len(output_36019))
        assert(len(expecter_36019) == len(output_36019))
        for i in range(len(output_36019)):
            assert(expecter_36019[i] == output_36019[i])



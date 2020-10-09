import skmob
from skmob.utils import utils, constants
import geopandas as gpd
from skmob.models import Gravity
import numpy as np
import math

class TestGravity:

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

    def test_gravity(self):
        gravity_singly = Gravity(gravity_type='singly constrained')

        np.random.seed(0)
        synth_fdf = gravity_singly.generate(self.tessellation,
                                            tile_id_column='tile_ID',
                                            tot_outflows_column='tot_outflow',
                                            relevance_column='population',
                                            out_format='flows')

        expecter_36019 = [ 101,   66, 1041,  151,   33,  106,  600,  512,  170,   77,  200,
                       3094, 1825,   70,  343,  115,  712, 1303, 1466,   59,  254,   40,
                         42, 1347,  296,   95,  179,   55,  147,  367, 1032,   89,   58,
                        366,  446,  128,  172,   23,  312, 1034,   98,  139,  388,   55,
                       1300,  100,  110,   53,  154,  557,  853,  447,   42,  432, 2001,
                        167,  291,  831, 1009, 1077, 1351]

        output_36019 = synth_fdf[synth_fdf.origin == '36019']['flow'].values

        assert(len(expecter_36019) == len(output_36019))
        for i in range(len(output_36019)):
            assert(expecter_36019[i] == output_36019[i])

        # FITTED
        gravity_singly_fitted = Gravity(gravity_type='singly constrained')
        gravity_singly_fitted.fit(self.fdf, relevance_column='population')

        np.random.seed(0)
        synth_fdf = gravity_singly_fitted.generate(self.tessellation,
                                            tile_id_column='tile_ID',
                                            tot_outflows_column='tot_outflow',
                                            relevance_column='population',
                                            out_format='flows')

        expecter_36019 = [ 136,  112,  576,  240,   77,  159,  616,  534,  297,  170,  177,
                           5039,  815,   87,  410,  147,  437,  588, 1831,   80,  420,   71,
                             73, 2434,  318,  126,  319,   84,  187,  285,  627,  160,  122,
                            310,  381,  308,  223,   57,  480,  786,  169,  190,  613,   95,
                            684,  201,  156,   89,  173,  953,  738,  558,  152,  342,  793,
                            245,  301,  844,  616, 1095,  675]

        output_36019 = synth_fdf[synth_fdf.origin == '36019']['flow'].values

        assert (math.isclose(gravity_singly_fitted.deterrence_func_args[0], -1.994715203191))
        assert (gravity_singly_fitted.deterrence_func_type == 'power_law')
        assert (math.isclose(gravity_singly_fitted.origin_exp, 1.0))
        assert (math.isclose(gravity_singly_fitted.destination_exp, 0.64717595522))

        assert (len(expecter_36019) == len(output_36019))
        for i in range(len(output_36019)):
            assert (expecter_36019[i] == output_36019[i])


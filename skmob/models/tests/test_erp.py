import skmob
from skmob.utils import utils, constants
import geopandas as gpd
from skmob.models.epr import DensityEPR
import numpy as np
import math

class TestERP:

    def setup_method(self):
        url_tess = '../../../tutorial/data/NY_counties_2011.geojson'

        self.tessellation = gpd.read_file(url_tess).rename(columns={'tile_id': 'tile_ID'})

    #def test_density_erp(self):




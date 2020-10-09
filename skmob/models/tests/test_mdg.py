import skmob
import pandas as pd
from skmob.models.markov_diary_generator import MarkovDiaryGenerator
from skmob.preprocessing import filtering, compression, detection, clustering
import numpy as np

class TestERP:

    def setup_method(self):
        url = '../../../tutorial/data/geolife_sample.txt.gz'

        df = pd.read_csv(url, sep=',', compression='gzip')
        tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')

        ctdf = compression.compress(tdf)
        stdf = detection.stops(ctdf)
        self.cstdf = clustering.cluster(stdf)

    #def test_density_erp(self):

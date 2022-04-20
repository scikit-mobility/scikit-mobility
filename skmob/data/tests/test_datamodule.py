import pytest
import numpy as np

# fix a random seed
np.random.seed(2)

# generate
@pytest.mark.parametrize('dataset_names', ['flow_foursquare_nyc', 'foursquare_nyc', 
                                           'nyc_boundaries', 'parking_san_francisco', 
                                           'taxi_san_francisco'])
@pytest.mark.parametrize('dataset_types', ['trajectory', 'flow', 'shape', 'auxiliar'])

def test_foo(dataset_names, dataset_types):
    print(dataset_names, dataset_types)
    assert 1==1
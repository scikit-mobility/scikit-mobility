from ...core.flowdataframe import FlowDataFrame
from ...utils import constants, gislib
import geopandas as gpd
from ...models import Radiation
import numpy as np
import shapely
import pytest

distfunc = gislib.getDistance

atol = 1e-12

# fix a random seed
np.random.seed(2)


def all_equal(a, b):
    return np.allclose(a, b, rtol=0., atol=atol)


# tessellation

tess_polygons = [[[7.481, 45.184],
                  [7.481, 45.216],
                  [7.526, 45.216],
                  [7.526, 45.184],
                  [7.481, 45.184]],
                 [[7.481, 45.216],
                  [7.481, 45.247],
                  [7.526, 45.247],
                  [7.526, 45.216],
                  [7.481, 45.216]],
                 [[7.526, 45.184],
                  [7.526, 45.216],
                  [7.571, 45.216],
                  [7.571, 45.184],
                  [7.526, 45.184]],
                 [[7.526, 45.216],
                  [7.526, 45.247],
                  [7.571, 45.247],
                  [7.571, 45.216],
                  [7.526, 45.216]]]

geom = [shapely.geometry.Polygon(p) for p in tess_polygons]
tessellation = gpd.GeoDataFrame(geometry=geom, crs="EPSG:4326")
tessellation = tessellation.reset_index().rename(columns={"index": constants.TILE_ID})

tot_outflow = np.random.randint(10, 20, size=len(tessellation))
relevance = np.random.randint(5, 10, size=len(tessellation))
tessellation[constants.TOT_OUTFLOW] = tot_outflow
tessellation[constants.RELEVANCE] = relevance

# # flows
# locs = tessellation[constants.TILE_ID].values
# ods = [[o, d] for o in locs for d in locs if o != d]
#
# # centroids
# centroid = [[lnla.y, lnla.x] for lnla in tessellation.geometry.centroid.values]
# # distance matrix
# distance = np.reshape([distfunc(laln0, laln1) for laln0 in centroid for laln1 in centroid], (len(locs), len(locs)))


# compute expected flows and probabilities

def correct_radiation():
    # TODO
    return 0


# generate

@pytest.mark.parametrize('out_format', ['flows', 'flows_sample', 'probabilities'])
def test_radiation_generate(out_format):

    # TODO: check correctness of results

    radiation = Radiation()
    rad_fdf = radiation.generate(tessellation, out_format=out_format)

    assert isinstance(rad_fdf, FlowDataFrame)



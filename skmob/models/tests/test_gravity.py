import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import pytest
import functools
from ...core.trajectorydataframe import FlowDataFrame
from ...utils import constants, gislib
from ...models.gravity import Gravity, exponential_deterrence_func, powerlaw_deterrence_func

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

# flows
locs = tessellation[constants.TILE_ID].values
ods = [[o, d] for o in locs for d in locs if o != d]

# centroids
centroid = [[lnla.y, lnla.x] for lnla in tessellation.geometry.centroid.values]
# distance matrix
distance = np.reshape([distfunc(laln0, laln1) for laln0 in centroid for laln1 in centroid], (len(locs), len(locs)))


# compute expected flows and probabilities for all models

def correct_gm(locs, tot_outflow, relevance, distance, detfunc, gravity_type, origin_exp=1.0, destination_exp=1.0,
               out_format='flows', exclude_selfflow=True):
    correct_flows = []

    if gravity_type == 'globally constrained':

        norm = 0.
        origs = []
        dests = []
        flows = []
        for o in locs:
            for d in locs:
                if not exclude_selfflow or o != d:
                    uflow = relevance[o] ** origin_exp * relevance[d] ** destination_exp * detfunc(distance[o, d])
                    norm += uflow
                    origs += [o]
                    dests += [d]
                    flows += [uflow]
        if out_format == 'flows':
            correct_flows += [[o, d, np.sum(tot_outflow) * f / norm] for o, d, f in zip(origs, dests, flows)]
        else:
            correct_flows += [[o, d, f / norm] for o, d, f in zip(origs, dests, flows)]

    else:

        for o in locs:
            norm = 0.
            dests = []
            flows = []
            for d in locs:
                if not exclude_selfflow or o != d:
                    uflow = relevance[o] ** origin_exp * relevance[d] ** destination_exp * detfunc(distance[o, d])
                    norm += uflow
                    dests += [d]
                    flows += [uflow]
            if out_format == 'flows':
                correct_flows += [[o, d, tot_outflow[o] * f / norm] for d, f in zip(dests, flows)]
            else:
                correct_flows += [[o, d, f / norm] for d, f in zip(dests, flows)]

    return FlowDataFrame(correct_flows, origin=0, destination=1, flow=2, tessellation=tessellation)


# generate

@pytest.mark.parametrize('deterrence_func_type_args', [['power_law', [-2]], ['exponential', [0.2]]])
@pytest.mark.parametrize('gravity_type', ['singly constrained', 'globally constrained'])
@pytest.mark.parametrize('origin_exp', [1.5])
@pytest.mark.parametrize('destination_exp', [2.0])
@pytest.mark.parametrize('out_format', ['flows', 'probabilities'])
def test_gravity_generate(deterrence_func_type_args, gravity_type, origin_exp, destination_exp, out_format):
    deterrence_func_type, deterrence_func_args = deterrence_func_type_args

    gm = Gravity(deterrence_func_type=deterrence_func_type,
                 deterrence_func_args=deterrence_func_args,
                 gravity_type=gravity_type,
                 origin_exp=origin_exp,
                 destination_exp=destination_exp)

    gmfdf = gm.generate(tessellation, out_format=out_format)

    # correct flows
    if deterrence_func_type == 'exponential':
        detfunc = functools.partial(exponential_deterrence_func, R=deterrence_func_args[0])
        exclude_selfflow = True #False
    else:
        detfunc = functools.partial(powerlaw_deterrence_func, exponent=deterrence_func_args[0])
        exclude_selfflow = True

    correct_fdf = correct_gm(locs, tot_outflow, relevance, distance, detfunc, gravity_type, origin_exp=origin_exp,
                 destination_exp=destination_exp, out_format=out_format, exclude_selfflow=exclude_selfflow)

    # compare
    fdf = pd.merge(gmfdf, correct_fdf, how='outer', on=[constants.ORIGIN, constants.DESTINATION]).fillna(0)

    # assert np.all(np.abs(fdf['flow_x'] - fdf['flow_y']).values < atol)
    assert all_equal(fdf['flow_x'].values, fdf['flow_y'].values)


# fit

@pytest.mark.parametrize('deterrence_func_type_args', [['power_law', [-2]], ['exponential', [0.2]]])
@pytest.mark.parametrize('gravity_type', ['singly constrained', 'globally constrained'])
@pytest.mark.parametrize('origin_exp', [1.0])
@pytest.mark.parametrize('destination_exp', [2.0])
@pytest.mark.parametrize('out_format', ['flows'])
def test_gravity_fit(deterrence_func_type_args, gravity_type, origin_exp, destination_exp, out_format):

    # TODO: check correctness of results

    deterrence_func_type, deterrence_func_args = deterrence_func_type_args

    # generate flows
    gm_gen = Gravity(deterrence_func_type=deterrence_func_type,
                     deterrence_func_args=deterrence_func_args,
                     gravity_type=gravity_type,
                     origin_exp=origin_exp,
                     destination_exp=destination_exp)

    gmfdf = gm_gen.generate(tessellation, out_format=out_format)

    # transform flows to integers
    gmfdf[constants.FLOW] = list(map(int, gmfdf[constants.FLOW].values + 0.5))

    # instantiate a new model and fit
    gm_fit = Gravity(deterrence_func_type=deterrence_func_type,
                     deterrence_func_args=deterrence_func_args,
                     gravity_type=gravity_type,
                     origin_exp=origin_exp,
                     destination_exp=destination_exp)

    fit_result = gm_fit.fit(gmfdf)

    assert fit_result is None


from ...utils import constants
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import pytest
from contextlib import ExitStack

from ...core.trajectorydataframe import TrajDataFrame
from ...models.gravity import Gravity
from ...models.epr import EPR, DensityEPR, SpatialEPR, Ditras
from ...models.markov_diary_generator import MarkovDiaryGenerator
from ...preprocessing import detection, clustering

atol = 1e-12

# fix a random seed
np.random.seed(2)


def all_equal(a, b):
    return np.allclose(a, b, rtol=0., atol=atol)


def global_variables():
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
    tessellation = tessellation

    gm = Gravity(gravity_type='singly constrained')
    gmfdf = gm.generate(tessellation, out_format='probabilities')
    odM = gmfdf.to_matrix()

    gcgm = Gravity(gravity_type='globally constrained')

    # instantiate a TrajDataFrame to fit the markov diary generator

    lats_lngs = np.array([[39.978253, 116.3272755],
                          [40.013819, 116.306532],
                          [39.878987, 116.1266865],
                          [40.013819, 116.306532],
                          [39.97958, 116.313649],
                          [39.978696, 116.3262205],
                          [39.98153775, 116.31079],
                          [39.978161, 116.3272425],
                          [38.978161, 115.3272425]])
    traj = pd.DataFrame(lats_lngs, columns=[constants.LATITUDE, constants.LONGITUDE])
    traj[constants.DATETIME] = pd.to_datetime([
        '20130101 8:34:04', '20130101 10:34:08', '20130105 10:34:08',
        '20130110 12:34:15', '20130101 1:34:28', '20130101 3:34:54',
        '20130101 4:34:55', '20130105 5:29:12', '20130115 00:29:12'])
    traj[constants.UID] = [1 for _ in range(5)] + [2 for _ in range(3)] + [3]
    tdf = TrajDataFrame(traj)
    stdf = detection.stops(tdf)
    cstdf = clustering.cluster(stdf)
    return tessellation, gm, gmfdf, gcgm, odM, cstdf

tessellation, gm, gmfdf, gcgm, odM, cstdf = global_variables()


# generate
@pytest.mark.parametrize('epr_model_type', [EPR, DensityEPR, SpatialEPR, Ditras])
@pytest.mark.parametrize('start_date', [pd.to_datetime('2019/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2019/01/02 08:00:00')])
@pytest.mark.parametrize('gravity_singly', [{}, gm, gcgm])
@pytest.mark.parametrize('n_agents', [1, 2])
@pytest.mark.parametrize('starting_locations', [None, 'random'])
@pytest.mark.parametrize('od_matrix', [None, odM])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
def test_epr_generate(epr_model_type, start_date, end_date, gravity_singly,
                      n_agents, starting_locations, od_matrix, random_state, show_progress):

    # TODO: check correctness of results

    if starting_locations == 'random':
        starting_locations = tessellation.sample(n=n_agents, replace=True)[constants.TILE_ID].values.tolist()

    # inititalize model
    if epr_model_type == Ditras:
        # create a markov diary generator
        mdg = MarkovDiaryGenerator()
        mdg.fit(cstdf, n_agents, lid=constants.CLUSTER)
        epr = epr_model_type(mdg)
    else:
        epr = epr_model_type()

    # generate flows
    with ExitStack() as stack:
        if gravity_singly != {}:
            if gravity_singly.gravity_type != 'singly constrained':
                stack.enter_context(pytest.raises(AttributeError))

        tdf = epr.generate(start_date, end_date,
                       spatial_tessellation=tessellation,
                       gravity_singly=gravity_singly,
                       n_agents=n_agents,
                       starting_locations=starting_locations,
                       od_matrix=od_matrix,
                       random_state=random_state,
                       show_progress=show_progress)

        assert isinstance(tdf, TrajDataFrame)

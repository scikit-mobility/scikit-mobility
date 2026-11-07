import geopandas as gpd
import numpy as np
import pytest
import shapely

from skmob.core.flowdataframe import FlowDataFrame
from skmob.models.radiation import Radiation, RadiationWithSelection, _survival_prob
from skmob.utils import constants, gislib

distfunc = gislib.getDistance

atol = 1e-12

# fix a random seed
np.random.seed(2)


def all_equal(a, b):
    return np.allclose(a, b, rtol=0.0, atol=atol)


# tessellation

tess_polygons = [
    [
        [7.481, 45.184],
        [7.481, 45.216],
        [7.526, 45.216],
        [7.526, 45.184],
        [7.481, 45.184],
    ],
    [
        [7.481, 45.216],
        [7.481, 45.247],
        [7.526, 45.247],
        [7.526, 45.216],
        [7.481, 45.216],
    ],
    [
        [7.526, 45.184],
        [7.526, 45.216],
        [7.571, 45.216],
        [7.571, 45.184],
        [7.526, 45.184],
    ],
    [
        [7.526, 45.216],
        [7.526, 45.247],
        [7.571, 45.247],
        [7.571, 45.216],
        [7.526, 45.216],
    ],
]

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


@pytest.mark.parametrize("out_format", ["flows", "flows_sample", "probabilities"])
def test_radiation_generate(out_format):

    # TODO: check correctness of results

    radiation = Radiation()
    rad_fdf = radiation.generate(tessellation, out_format=out_format)

    assert isinstance(rad_fdf, FlowDataFrame)


@pytest.mark.parametrize("out_format", ["flows", "probabilities"])
def test_radiation_with_selection_matches_standard_when_lambda_zero(out_format):
    # When lambda = 0, RadiationWithSelection must exactly match Radiation
    np.random.seed(42)
    rad_standard = Radiation()
    fdf_standard = rad_standard.generate(tessellation, out_format=out_format)

    np.random.seed(42)
    rad_selection = RadiationWithSelection(lambd=0.0)
    fdf_selection = rad_selection.generate(tessellation, out_format=out_format)

    assert isinstance(fdf_selection, FlowDataFrame)
    assert all_equal(
        fdf_standard[constants.FLOW].values, fdf_selection[constants.FLOW].values
    )


@pytest.mark.parametrize("lambd", [0.1, 0.5])
def test_radiation_with_selection_generate(lambd):
    rad = RadiationWithSelection(lambd=lambd)
    fdf_probs = rad.generate(tessellation, out_format="probabilities")

    assert isinstance(fdf_probs, FlowDataFrame)
    # Check that probabilities for each origin sum to 1.0
    prob_sums = fdf_probs.groupby(constants.ORIGIN)[constants.FLOW].sum()
    for _, p_sum in prob_sums.items():
        assert np.isclose(p_sum, 1.0, atol=1e-10)


def test_radiation_with_selection_invalid_lambda():
    with pytest.raises(ValueError):
        RadiationWithSelection(lambd=-0.1)

    with pytest.raises(ValueError):
        RadiationWithSelection(lambd=1.0)


def test_survival_prob_mathematical_properties():
    # When lambda = 0, survival_prob is 1 / (a + 1)
    m_i = 100.0
    s = 50.0
    a = s / m_i
    assert np.isclose(_survival_prob(s, m_i, 0.0), 1.0 / (a + 1.0))

    # At s = 0, a = 0, P_{>}(0) = (1 - lambda) / (1 * (1 - lambda)) = 1.0 for any lambda
    for param_lambda in [0.0, 0.2, 0.7]:
        assert np.isclose(_survival_prob(0.0, m_i, param_lambda), 1.0)

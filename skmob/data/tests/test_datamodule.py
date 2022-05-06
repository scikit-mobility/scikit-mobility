import pytest
from geopandas import GeoDataFrame
from pandas import DataFrame

from skmob.core.flowdataframe import FlowDataFrame
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.data.load import load_dataset


# generate
@pytest.mark.parametrize(
    "dataset_names", ["flow_foursquare_nyc", "foursquare_nyc", "nyc_boundaries", "parking_san_francisco"]
)
@pytest.mark.parametrize("dataset_types", ["trajectory", "flow", "shape", "auxiliar"])
def test_loading_existing_datasets(dataset_names):

    data = load_dataset(dataset_names)

    assert type(data) in [TrajDataFrame, FlowDataFrame, GeoDataFrame, DataFrame]

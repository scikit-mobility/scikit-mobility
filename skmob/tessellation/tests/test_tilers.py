import geopandas as gpd
from ...tessellation import tilers
import shapely
import pytest

poly = [[[116.1440758191,39.8846396072],
                  [116.3449987678,39.8846396072],
                  [116.3449987678,40.0430521004],
                  [116.1440758191,40.0430521004],
                  [116.1440758191,39.8846396072]]]
geom = [shapely.geometry.Polygon(p) for p in poly]
bbox = gpd.GeoDataFrame(geometry=geom, crs="EPSG:4326")


@pytest.mark.parametrize('tiler_type', ["squared"])
@pytest.mark.parametrize('base_shape', ['Beijing, China', bbox])
@pytest.mark.parametrize('meters', [15000])
def test_tiler_get(tiler_type, base_shape, meters):
    tessellation = tilers.tiler.get(tiler_type, base_shape=base_shape, meters=meters)
    assert isinstance(tessellation, gpd.GeoDataFrame)

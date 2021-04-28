import geopandas as gpd
from skmob.tessellation import tilers
import shapely
from shapely.geometry import Point, Polygon
import pytest

poly = [[[116.1440758191, 39.8846396072],
         [116.3449987678, 39.8846396072],
         [116.3449987678, 40.0430521004],
         [116.1440758191, 40.0430521004],
         [116.1440758191, 39.8846396072]]]
geom = [shapely.geometry.Polygon(p) for p in poly]
bbox = gpd.GeoDataFrame(geometry=geom, crs="EPSG:4326")


@pytest.mark.parametrize('tiler_type', ["squared", "h3_tessellation"])
@pytest.mark.parametrize('base_shape', ['Beijing, China', bbox])
@pytest.mark.parametrize('meters', [15000])
def test_tiler_get(tiler_type, base_shape, meters):
    tessellation = tilers.tiler.get(tiler_type, base_shape=base_shape, meters=meters)
    assert isinstance(tessellation, gpd.GeoDataFrame)

# Arrange
@pytest.fixture()
def h3_tess():
    return tilers.H3TessellationTiler()

@pytest.mark.parametrize("input_meters, expected_res", [(500, 8), (1500, 7), (5000, 6)])
def test__meters_to_resolution(h3_tess, input_meters, expected_res):
    assert h3_tess._meters_to_resolution(input_meters) == expected_res


def test__isinstance_geodataframe_or_geoseries(h3_tess):
    assert h3_tess._isinstance_geodataframe_or_geoseries(bbox) == True

def test__str_to_geometry(h3_tess):
    assert isinstance(h3_tess._str_to_geometry("Milan, Italy", 1), gpd.GeoDataFrame)


def test__find_first_polygon_expected_length(h3_tess):
    base_shapes = bbox.append({"geometry": Point(9, 45)},ignore_index=True)
    first_polygon = h3_tess._find_first_polygon(base_shapes)
    assert isinstance(first_polygon.values.tolist()[0][0], Polygon)


def test__find_first_polygon_expected_type(h3_tess):
    base_shapes = bbox.append({"geometry": Point(9, 45)},ignore_index=True)
    first_polygon = h3_tess._find_first_polygon(base_shapes)
    assert isinstance(first_polygon.values.tolist()[0][0], Polygon)


def test__isinstance_poly_or_multipolygon(h3_tess):
    assert h3_tess._isinstance_poly_or_multipolygon(Polygon( [[1,0], [1,1], [0,1], [0,0]])) == True


def test__merge_all_polygons(h3_tess):
    assert h3_tess._merge_all_polygons(bbox).shape[0] == 1


def test__get_resolution(h3_tess):
    assert h3_tess._get_resolution(base_shape=bbox, meters=5000) == 6
    assert h3_tess._get_resolution(base_shape=bbox, meters=50000) == 3


def test__suggest_minimum_resolution_which_still_fits(h3_tess):
    with pytest.warns(UserWarning) as user_warnings:
        h3_tess._suggest_minimum_resolution_which_still_fits(4)


def test__handle_polyfill():
    assert False


def test__extract_geometry():
    assert False


def test__get_hexagons():
    assert False


def test__create_hexagon_polygons():
    assert False


def test__add_tile_id():
    assert False


def test__meters_to_resolution():
    assert False


def test__meters_to_kilometers():
    assert False


def test__load_h3_utils():
    assert False


def test__find_min_resolution():
    assert False


def test__squared_meters_to_squared_kilometers():
    assert False



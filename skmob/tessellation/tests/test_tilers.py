import geopandas as gpd
import pytest
import shapely
from pandas import DataFrame
from shapely.geometry import Point, Polygon

from skmob.tessellation import tilers

poly = [
    [
        [116.1440758191, 39.8846396072],
        [116.3449987678, 39.8846396072],
        [116.3449987678, 40.0430521004],
        [116.1440758191, 40.0430521004],
        [116.1440758191, 39.8846396072],
    ]
]
geom = [shapely.geometry.Polygon(p) for p in poly]
bbox = gpd.GeoDataFrame(geometry=geom, crs="EPSG:4326")


@pytest.fixture()
def h3_tess():
    return tilers.H3TessellationTiler()


@pytest.mark.parametrize("tiler_type", ["squared", "h3_tessellation"])
@pytest.mark.parametrize("base_shape", ["Beijing, China", bbox])
@pytest.mark.parametrize("meters", [15000])
def test_tiler_get(tiler_type, base_shape, meters):
    tessellation = tilers.tiler.get(tiler_type, base_shape=base_shape, meters=meters)
    assert isinstance(tessellation, gpd.GeoDataFrame)


def test__isinstance_geodataframe_or_geoseries(h3_tess):
    if h3_tess._isinstance_geodataframe_or_geoseries(bbox):
        return True


def test__str_to_geometry(h3_tess):
    assert isinstance(h3_tess._str_to_geometry("Milan, Italy", 1), gpd.GeoDataFrame)


def test__find_first_polygon_expected_length(h3_tess):
    base_shapes = bbox.append({"geometry": Point(9, 45)}, ignore_index=True)
    first_polygon = h3_tess._find_first_polygon(base_shapes)
    assert isinstance(first_polygon.values.tolist()[0][0], Polygon)


def test__find_first_polygon_expected_type(h3_tess):
    base_shapes = bbox.append({"geometry": Point(9, 45)}, ignore_index=True)
    first_polygon = h3_tess._find_first_polygon(base_shapes)
    assert isinstance(first_polygon.values.tolist()[0][0], Polygon)


def test__isinstance_poly_or_multipolygon(h3_tess):
    candidate_polygon = Polygon([[1, 0], [1, 1], [0, 1], [0, 0]])
    if h3_tess._isinstance_poly_or_multipolygon(candidate_polygon):
        return True


def test__merge_all_polygons(h3_tess):
    assert h3_tess._merge_all_polygons(bbox).shape[0] == 1


@pytest.mark.parametrize("input_meters, expected_resolution", [(5000, 6), (50000, 3)])
def test__get_resolution(h3_tess, input_meters, expected_resolution):
    assert (
        h3_tess._get_resolution(base_shape=bbox, meters=input_meters)
        == expected_resolution
    )


def test__suggest_minimum_resolution_which_still_fits(h3_tess):
    with pytest.warns(UserWarning) as user_warnings:  # noqa: F841
        h3_tess._suggest_minimum_resolution_which_still_fits(4)


def test__handle_polyfill(h3_tess):
    hexagons = h3_tess._handle_polyfill(bbox, 5)
    assert hexagons[0] == 599852472416075775


def test__extract_geometry(h3_tess):
    extracted_geometry = h3_tess._extract_geometry(bbox)
    assert extracted_geometry["type"] == "Polygon"


def test__get_hexagons(h3_tess):
    hexagon = h3_tess._get_hexagons(Polygon(geom[0]), 5)
    assert hexagon == 599852472416075775


def test__create_hexagon_polygons(h3_tess):
    hexagon_polygons = h3_tess._create_hexagon_polygons([599852472416075775])
    assert isinstance(hexagon_polygons, DataFrame)


def test__add_tile_id(h3_tess):

    hexagon_polygons_with_id = h3_tess._add_tile_id(bbox)

    assert "tile_ID" in hexagon_polygons_with_id.columns


@pytest.mark.parametrize(
    "input_meters, expected_resolution", [(50, 10), (500, 8), (5000, 6)]
)
def test__meters_to_resolution(h3_tess, input_meters, expected_resolution):
    assert h3_tess._meters_to_resolution(input_meters) == expected_resolution


@pytest.mark.parametrize("input_meters, expected_kilometers", [(1000, 1), (10000, 10)])
def test__meters_to_kilometers(h3_tess, input_meters, expected_kilometers):
    assert h3_tess._meters_to_kilometers(input_meters) == expected_kilometers


def test__load_h3_utils(h3_tess):
    assert len(h3_tess._load_h3_utils("average_hexagon_area")) == 16
    assert len(h3_tess._load_h3_utils("average_hexagon_edge_length")) == 16


def test__find_min_resolution(h3_tess):
    assert h3_tess._find_min_resolution(bbox) == 15


def test__squared_meters_to_squared_kilometers(h3_tess):
    assert h3_tess._squared_meters_to_squared_kilometers(bbox) == 3.18287052446638e-08

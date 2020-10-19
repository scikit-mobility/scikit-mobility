import pytest
import numpy as np
from ...utils import constants
import geopandas as gpd
from ...core.flowdataframe import FlowDataFrame
import folium


# tessellation
tess_features = {'type': 'FeatureCollection',
                 'features': [{'id': '0',
                               'type': 'Feature',
                               'properties': {'tile_ID': '0'},
                               'geometry': {'type': 'Polygon',
                                            'coordinates': [[[116.14407581909998, 39.8846396072],
                                                             [116.14407581909998, 39.98795822127371],
                                                             [116.27882311171793, 39.98795822127371],
                                                             [116.27882311171793, 39.8846396072],
                                                             [116.14407581909998, 39.8846396072]]]}},
                              {'id': '1',
                               'type': 'Feature',
                               'properties': {'tile_ID': '1'},
                               'geometry': {'type': 'Polygon',
                                            'coordinates': [[[116.14407581909998, 39.98795822127371],
                                                             [116.14407581909998, 40.091120806035285],
                                                             [116.27882311171793, 40.091120806035285],
                                                             [116.27882311171793, 39.98795822127371],
                                                             [116.14407581909998, 39.98795822127371]]]}},
                              {'id': '2',
                               'type': 'Feature',
                               'properties': {'tile_ID': '2'},
                               'geometry': {'type': 'Polygon',
                                            'coordinates': [[[116.27882311171793, 39.8846396072],
                                                             [116.27882311171793, 39.98795822127371],
                                                             [116.41357040433583, 39.98795822127371],
                                                             [116.41357040433583, 39.8846396072],
                                                             [116.27882311171793, 39.8846396072]]]}},
                              {'id': '3',
                               'type': 'Feature',
                               'properties': {'tile_ID': '3'},
                               'geometry': {'type': 'Polygon',
                                            'coordinates': [[[116.27882311171793, 39.98795822127371],
                                                             [116.27882311171793, 40.091120806035285],
                                                             [116.41357040433583, 40.091120806035285],
                                                             [116.41357040433583, 39.98795822127371],
                                                             [116.27882311171793, 39.98795822127371]]]}}]}
tessellation = gpd.GeoDataFrame.from_features(tess_features, crs={"init": "epsg:4326"})

# flowdataframe
flow_matrix = np.array([[0., 0., 0., 0.],
                        [0., 0., 0., 4.],
                        [0., 0., 3., 1.],
                        [0., 2., 0., 0.]])
flows = {'origin': {0: '2', 1: '2', 2: '3', 3: '1'},
              'destination': {0: '2', 1: '3', 2: '1', 3: '3'},
              'flow': {0: 3, 1: 1, 2: 2, 3: 4}}
fdf = FlowDataFrame(flows, tessellation=tessellation)


class TestFlowDataFrame:

    # @pytest.mark.parametrize('fdf', [fdf])
    # @pytest.mark.parametrize('expected_result', [True])
    def test_is_flowdataframe(self, expected_result=True):
        assert(fdf._is_flowdataframe() == expected_result)

    def test_get_flow(self):
        for i, f in flows['flow'].items():
            assert f == fdf.get_flow(flows['origin'][i], flows['destination'][i])

    # @pytest.mark.parametrize('fdf', [fdf])
    def test_get_geometry(self):
        assert np.all([[g == fdf.get_geometry(t)] for i, (g, t) in fdf.tessellation.iterrows()])

    # @pytest.mark.parametrize('fdf', [fdf])
    def test_to_matrix(self):
        assert np.all(fdf.to_matrix() == flow_matrix)

    # def test_settings_from(self):
    #     assert 0 == 0

    def test_from_file(self):
        tess = gpd.GeoDataFrame.from_file(constants.NY_COUNTIES_2011).rename(columns={'tile_id': constants.TILE_ID})
        fdf0 = FlowDataFrame.from_file(constants.NY_FLOWS_2011, tessellation=tess)
        assert fdf0._is_flowdataframe()

    # @pytest.mark.parametrize('fdf', [fdf])
    @pytest.mark.parametrize('min_flow', [0, 2])
    @pytest.mark.parametrize('flow_popup', [False, True])
    def test_plot_flows(self, min_flow, flow_popup):
        map_f = fdf.plot_flows(min_flow=min_flow, flow_popup=flow_popup)
        assert isinstance(map_f, folium.folium.Map)

    def test_plot_gdf(self):
        map_f = fdf.plot_tessellation()
        assert isinstance(map_f, folium.folium.Map)

import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
from operator import itemgetter
from ...utils import constants
from ...utils.constants import UID, DATETIME, LATITUDE, LONGITUDE, GEOLIFE_SAMPLE
from ...core.trajectorydataframe import TrajDataFrame
from ...core.flowdataframe import FlowDataFrame
from ...preprocessing import detection, clustering
import shapely
import folium
import matplotlib
import pytest

EXPECTED_NUM_OF_COLUMNS_IN_TDF = 4


class TestTrajectoryDataFrame:

    def setup_method(self):
        self.default_data_list = [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'],
                                  [1, 39.984198, 116.319322, '2008-10-23 13:53:06'],
                                  [1, 39.984224, 116.319402, '2008-10-23 13:53:11'],
                                  [1, 39.984211, 116.319389, '2008-10-23 13:53:16']]
        self.default_data_df = pd.DataFrame(self.default_data_list, columns=['user', 'latitude', 'lng', 'hour'])
        self.default_data_dict = self.default_data_df.to_dict(orient='list')

        # instantiate a TrajDataFrame

        lats_lngs = np.array([[39.978253, 116.327275],
                              [40.013819, 116.306532],
                              [39.878987, 116.126686],
                              [40.013819, 116.306532],
                              [39.979580, 116.313649],
                              [39.978696, 116.326220],
                              [39.981537, 116.310790],
                              [39.978161, 116.327242],
                              [39.900000, 116.000000]])
        traj = pd.DataFrame(lats_lngs, columns=[constants.LATITUDE, constants.LONGITUDE])
        traj[constants.DATETIME] = pd.to_datetime([
            '20130101 8:34:04', '20130101 10:34:08', '20130105 10:34:08',
            '20130110 12:34:15', '20130101 1:34:28', '20130101 3:34:54',
            '20130101 4:34:55', '20130105 5:29:12', '20130115 00:29:12'])
        traj[constants.UID] = [1 for _ in range(5)] + [2 for _ in range(3)] + [3]
        self.tdf0 = TrajDataFrame(traj)
        self.stdf = detection.stops(self.tdf0)
        self.cstdf = clustering.cluster(self.stdf)

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
        self.tessellation = gpd.GeoDataFrame.from_features(tess_features, crs={"init": "epsg:4326"})

    def perform_default_asserts(self, tdf):
        assert tdf._is_trajdataframe()
        assert tdf.shape == (4, EXPECTED_NUM_OF_COLUMNS_IN_TDF)
        assert tdf[UID][0] == 1
        assert tdf[DATETIME][0] == datetime(2008, 10, 23, 13, 53, 5)
        assert tdf[LATITUDE][0] == 39.984094
        assert tdf[LONGITUDE][3] == 116.319389

    def test_tdf_from_list(self):
        tdf = TrajDataFrame(self.default_data_list, latitude=1, longitude=2, datetime=3, user_id=0)
        self.perform_default_asserts(tdf)
        print(tdf.head())  # raised TypeError: 'BlockManager' object is not iterable

    def test_tdf_from_df(self):
        tdf = TrajDataFrame(self.default_data_df, latitude='latitude', datetime='hour', user_id='user')
        self.perform_default_asserts(tdf)

    def test_tdf_from_dict(self):
        tdf = TrajDataFrame(self.default_data_dict, latitude='latitude', datetime='hour', user_id='user')
        self.perform_default_asserts(tdf)

    def test_tdf_from_csv_file(self):
        tdf = TrajDataFrame.from_file(GEOLIFE_SAMPLE, sep=',')
        assert tdf._is_trajdataframe()
        assert tdf.shape == (217653, EXPECTED_NUM_OF_COLUMNS_IN_TDF)
        assert list(tdf[UID].unique()) == [1, 5]

    def test_timezone_conversion(self):
        tdf = TrajDataFrame(self.default_data_df, latitude='latitude', datetime='hour', user_id='user')
        tdf.timezone_conversion(from_timezone='Europe/London', to_timezone='Europe/Berlin')
        assert tdf[DATETIME][0] == pd.Timestamp('2008-10-23 14:53:05')

    def test_slicing_a_tdf_returns_a_tdf(self):
        tdf = TrajDataFrame(self.default_data_df, latitude='latitude', datetime='hour', user_id='user')
        assert isinstance(tdf[tdf[UID] == 1][:1], TrajDataFrame)

    def test_sort_by_uid_and_datetime(self):
        # shuffle the TrajDataFrame rows
        tdf1 = self.tdf0.sample(frac=1)
        tdf = tdf1.sort_by_uid_and_datetime()
        assert isinstance(tdf, TrajDataFrame)
        assert np.all(tdf[[UID, DATETIME]].values == sorted(tdf1[[UID, DATETIME]].values, key=itemgetter(0, 1)))

    def test_plot_trajectory(self):
        map_f = self.tdf0.plot_trajectory()
        assert isinstance(map_f, folium.folium.Map)

    def test_plot_stops(self):
        map_f = self.stdf.plot_stops()
        assert isinstance(map_f, folium.folium.Map)

    def test_plot_diary(self):
        ax = self.cstdf.plot_diary(self.tdf0[UID].iloc[0])
        assert isinstance(ax, matplotlib.axes._subplots.Subplot)

    @pytest.mark.parametrize('self_loops', [True, False])
    def test_to_flowdataframe(self, self_loops):

        expected_flows = {'origin': {0: '2', 1: '2'},
                         'destination': {0: '2', 1: '3'},
                         'flow': {0: 3, 1: 1}}
        expected_fdf = FlowDataFrame(expected_flows, tessellation=self.tessellation)
        if not self_loops:
            expected_fdf.drop(0, inplace=True)
        fdf = self.tdf0.to_flowdataframe(self.tessellation, self_loops=self_loops)

        assert isinstance(fdf, FlowDataFrame)
        pd.testing.assert_frame_equal(expected_fdf, fdf)

    def test_to_geodataframe(self):
        assert isinstance(self.tdf0.to_geodataframe(), gpd.GeoDataFrame)

    @pytest.mark.parametrize('remove_na', [True, False])
    def test_mapping(self, remove_na):
        mtdf = self.tdf0.mapping(self.tessellation, remove_na=remove_na)

        def _point_in_poly(x, tess):
            point = shapely.geometry.Point([x[constants.LONGITUDE], x[constants.LATITUDE]])
            try:
                poly = tess[tess[constants.TILE_ID] == x[constants.TILE_ID]][['geometry']].values[0, 0]
                return poly.contains(point)
            except IndexError:
                poly = shapely.ops.unary_union(self.tessellation.geometry.values)
                return not poly.contains(point)

        assert np.all(mtdf.apply(lambda x: _point_in_poly(x, self.tessellation), axis=1).values)

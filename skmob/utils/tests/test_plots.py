import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import pytest
from ...core.trajectorydataframe import TrajDataFrame, FlowDataFrame
from .. import plot
from ...utils import constants
from ...preprocessing import detection, clustering
import folium
import matplotlib

lat = constants.LATITUDE
lng = constants.LONGITUDE
dt = constants.DATETIME
uid = constants.UID
tid = constants.TID
atol = 1e-12


def all_equal(a, b):
    return np.allclose(a, b, rtol=0., atol=atol)


lats_lngs = np.array([ [39.97, 116.32],
                       [39.90, 116.51],
                       [39.60, 116.60],
                       [40.01, 115.90],
                       [39.96, 115.85],
                       [39.70, 115.80],
                       [39.5999, 116.5999],
                       [38.60, 115.56],
                       [38.98, 114.51],
                       [40.19, 114.32],
                       [40.97, 113.82]])

traj = pd.DataFrame(lats_lngs, columns=[lat, lng])

traj[dt] = pd.to_datetime([
        '2013/01/01 8:00:00', '2013/01/01 8:05:00',
        '2013/01/01 8:10:00', '2013/01/01 9:00:00',
        '2013/01/01 9:01:00', '2013/01/02 9:55:00',
        '2013/01/02 9:57:00', '2013/01/02 10:40:00',
        '2013/01/01 1:00:00', '2013/01/01 1:00:10', '2013/01/01 2:00:00'])

traj[uid] = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
traj[tid] = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

tdf_test = TrajDataFrame(traj)

all_users = [1, 2, 3]


points = [shapely.geometry.Point(ll) for ll in [[39.97, 116.32], [39.90, 116.51]]]

lines = [shapely.geometry.LineString(ll) for ll in [[[116.32, 39.97], [116.51, 39.90]],
                                                     [[116.22, 39.87], [116.41, 39.80]]]]

polygons = [shapely.geometry.Polygon(ll) for ll in [[[116.32, 39.97], [116.51, 39.90], [116.51, 39.97]],
                                                   [[116.22, 39.87], [116.41, 39.80], [116.22, 39.80]]]]


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
  [7.526, 45.184]]]#,
 # [[7.526, 45.216],
 #  [7.526, 45.247],
 #  [7.571, 45.247],
 #  [7.571, 45.216],
 #  [7.526, 45.216]]]

geom = [shapely.geometry.Polygon(p) for p in tess_polygons]
tessellation = gpd.GeoDataFrame(geometry=geom, crs="EPSG:4326")
tessellation = tessellation.reset_index().rename(columns={"index": constants.TILE_ID})


# flows

flow_list = [[1, 0, 1],
             [5, 0, 2],
             [3, 1, 0],
             [2, 1, 2],
             [8, 2, 0],
             [9, 2, 1],]

df = pd.DataFrame(flow_list, columns=[constants.FLOW, constants.ORIGIN, constants.DESTINATION])
fdf = FlowDataFrame(df, tessellation=tessellation)


# plot_trajectory

@pytest.mark.parametrize('tdf', [tdf_test])
@pytest.mark.parametrize('marker', [True, False])
def test_plot_trajectory(tdf, marker):
    map_f = plot.plot_trajectory(tdf, start_end_markers=marker)
    assert isinstance(map_f, folium.folium.Map)


@pytest.mark.parametrize('tdf', [tdf_test])
@pytest.mark.parametrize('marker', [True, False])
def test_plot_trajectory_tdf(tdf, marker):
    map_f = tdf.plot_trajectory(start_end_markers=marker)
    assert isinstance(map_f, folium.folium.Map)


# plot_stops

@pytest.mark.parametrize('tdf', [tdf_test])
def test_plot_stops(tdf):
    map_f = plot.plot_trajectory(tdf)
    # map_f = plot.plot_stops(tdf, map_f=map_f)

    stdf = detection.stops(tdf)
    map_f = plot.plot_stops(stdf, map_f=map_f)

    assert isinstance(map_f, folium.folium.Map)


@pytest.mark.parametrize('tdf', [tdf_test])
def test_plot_stops_tdf(tdf):
    map_f = tdf.plot_trajectory()
    # map_f = tdf.plot_stops(map_f=map_f)

    stdf = detection.stops(tdf)
    map_f = stdf.plot_stops(map_f=map_f)

    assert isinstance(map_f, folium.folium.Map)


# plot_diary

@pytest.mark.parametrize('tdf', [tdf_test])
@pytest.mark.parametrize('user', [1, 2])
@pytest.mark.parametrize('start_datetime', [None, '2013/01/01 00:00:00'])
def test_plot_diary(tdf, user, start_datetime):
    stdf = detection.stops(tdf)
    cstdf = clustering.cluster(stdf)
    ax = plot.plot_diary(cstdf, user, start_datetime=start_datetime)

    assert isinstance(ax, matplotlib.axes._subplots.Subplot)


@pytest.mark.parametrize('tdf', [tdf_test])
@pytest.mark.parametrize('user', [1, 2])
@pytest.mark.parametrize('start_datetime', [None, '2013/01/01 00:00:00'])
def test_plot_diary(tdf, user, start_datetime):
    stdf = detection.stops(tdf)
    cstdf = clustering.cluster(stdf)
    ax = cstdf.plot_diary(user, start_datetime=start_datetime)

    assert isinstance(ax, matplotlib.axes._subplots.Subplot)


# plot_flows

@pytest.mark.parametrize('fdf', [fdf])
@pytest.mark.parametrize('min_flow', [0, 2])
@pytest.mark.parametrize('flow_popup', [False, True])
def test_plot_flows(fdf, min_flow, flow_popup):
    map_f = plot.plot_flows(fdf, min_flow=min_flow, flow_popup=flow_popup)
    assert isinstance(map_f, folium.folium.Map)

@pytest.mark.parametrize('fdf', [fdf])
@pytest.mark.parametrize('min_flow', [0, 2])
@pytest.mark.parametrize('flow_popup', [False, True])
def test_plot_flows_fdf(fdf, min_flow, flow_popup):
    map_f = fdf.plot_flows(min_flow=min_flow, flow_popup=flow_popup)
    assert isinstance(map_f, folium.folium.Map)


# plot_gdf

@pytest.mark.parametrize('geom', [points, lines, polygons])
def test_plot_gdf(geom):
    gdf = gpd.GeoDataFrame(geom, columns=['geometry'])
    map_f = plot.plot_gdf(gdf)
    assert isinstance(map_f, folium.folium.Map)



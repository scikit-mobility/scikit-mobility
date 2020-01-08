# import pandas as pd
# import numpy as np
# import osmnx as ox
# from ..utils import constants, gislib
#
# distance = gislib.getDistance
#
#
# def route(tdf, G=None, index_origin=0, index_destin=-1):
#     """Routing.
#
#     Create a route on Google Maps between two locations.
#
#     Parameters
#     ----------
#     tdf : TrajDataFrame
#         the TrajDataFrame with the locations
#
#     Examples
#     --------
#
#     G, shortest_route = route(tdf)
#
#     m = ox.plot_route_folium(G, shortest_route, route_color='green')
#
#     origin_coords = tuple(tdf1.iloc[0][['lat', 'lng']].values)
#     destin_coords = tuple(tdf1.iloc[-1][['lat', 'lng']].values)
#
#     folium.Marker(location=origin_coords,
#               icon=folium.Icon(color='red')).add_to(m)
#     folium.Marker(location=destin_coords,
#               icon=folium.Icon(color='blue')).add_to(m)
#
#     tdf1.plot_trajectory(map_f=m)
#     """
#     if index_destin == -1:
#         index_destin_last = None
#     else:
#         index_destin_last = index_destin
#     tdf1 = tdf[index_origin: index_destin_last]
#
#     origin_coords = tuple(tdf1.iloc[index_origin][[constants.LATITUDE, constants.LONGITUDE]].values)
#     destin_coords = tuple(tdf1.iloc[index_destin][[constants.LATITUDE, constants.LONGITUDE]].values)
#
#     if G is None:
#         mid_point = tuple(np.mean(np.array([origin_coords, destin_coords]), axis=0))
#         # all distances from mid_point
#         all_dists = pd.DataFrame(tdf1[[constants.LATITUDE, constants.LONGITUDE]]).apply(
#             lambda x: distance(mid_point, tuple(x.values)).m, axis=1).values
#         max_dist = 1.1 * max(all_dists)
#         G = ox.graph_from_point(mid_point, distance=max_dist)
#
#     # nodes, _ = ox.graph_to_gdfs(G)
#
#     # closest points to origin and destination on graph
#     closest_o_i = ox.utils.get_nearest_node(G, origin_coords)
#     closest_d_i = ox.utils.get_nearest_node(G, destin_coords)
#
#     # find shortest path
#     shortest_route = ox.nx.shortest_path(G, closest_o_i, closest_d_i, weight='length')
#
#     return G, shortest_route

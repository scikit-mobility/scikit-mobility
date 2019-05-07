from ..utils import constants, utils
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from geojson import LineString
import geopandas as gpd
import json


COLOR = {
    0: '#FF0000',  # Red
    1: '#008000',  # Green
    2: '#000080',  # Navy
    3: '#800000',  # Maroon
    4: '#FFD700',  # Gold
    5: '#00FF00',  # Lime
    6: '#800080',  # Purple
    7: '#00FFFF',  # Aqua
    8: '#DC143C',  # Crimson
    9: '#0000FF',  # Blue
    10: '#F08080',  # LightCoral
    11: '#FF00FF',  # Fuchsia
    12: '#FF8C00',  # DarkOrange
    13: '#6A5ACD',  # SlateBlue
    14: '#8B4513',  # SaddleBrown
    15: '#1E90FF',  # DodgerBlue
    16: '#FFFF00',  # Yellow
    17: '#808080',  # Gray
    18: '#008080',  # Teal
    19: '#9370DB',  # MediumPurple
    20: '#2F4F4F'  # DarkSlateGray
}


def get_color(k=-1, color_dict=COLOR):
    """
    Return a color (random if "k" is negative)
    """
    if k < 0:
        return np.random.choice(list(color_dict.values()))  # color_dict[random.randint(0,20)]
    else:
        return color_dict[k % 21]


def random_hex():
    r = lambda: np.random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())


def plot_trajectory(tdf, map_f=None, max_users=10, max_points=1000, imin=0, imax=-1,
                    tiles='cartodbpositron', zoom=12, hex_color=-1, weight=2, opacity=0.75):
    # group by user and keep only the first `max_users`
    nu = 0
    for user, df in tdf.groupby(constants.UID):
        if nu >= max_users:
            break
        nu += 1

        traj = df[[constants.LONGITUDE, constants.LATITUDE]].values[imin:imax]

        if max_points == None:
            di = 1
        else:
            di = max(1, len(traj) // max_points)
        traj = traj[::di]

        if nu == 1 and map_f == None:
            # initialise map
            center = list(np.median(traj, axis=0)[::-1])
            map_f = folium.Map(location=center, zoom_start=zoom, tiles=tiles)

        line = LineString(traj.tolist())

        if hex_color == -1:
            color = get_color(hex_color)
        else:
            color = hex_color
        ss = {
            "type": "Feature",
            "geometry": line,
            "properties": {"style":
                {
                    "color": color,
                    "weight": weight,
                    "opacity": opacity
                }
            }
        }
        folium.GeoJson(ss).add_to(map_f)

    return map_f


def plot_stops(stdf, map_f=None, max_users=10, tiles='cartodbpositron', zoom=12,
               hex_color=-1, opacity=0.3, popup=True):
    if map_f == None:
        # initialise map
        lo_la = stdf[['lng', 'lat']].values
        center = list(np.median(lo_la, axis=0)[::-1])
        map_f = folium.Map(location=center, zoom_start=zoom, tiles=tiles)

    # group by user and keep only the first `max_users`
    nu = 0
    for user, df in stdf.groupby(constants.UID):
        if nu >= max_users:
            break
        nu += 1

        if hex_color == -1:
            color = get_color(hex_color)
        else:
            color = hex_color

        for idx, row in df.iterrows():

            la = row[constants.LATITUDE]
            lo = row[constants.LONGITUDE]
            t0 = row[constants.DATETIME]
            t1 = row[constants.LEAVING_DATETIME]
            u = row[constants.UID]
            try:
                ncluster = row[constants.CLUSTER]
                cl = '<BR>Cluster: {}'.format(ncluster)
                color = get_color(ncluster)
            except (KeyError, NameError):
                cl = ''

            fpoly = folium.RegularPolygonMarker([la, lo],
                                        radius=12,
                                        color=color,
                                        fill_color=color,
                                        fill_opacity=opacity
                                        )
            if popup:
                popup = folium.Popup('User: {}<BR>Coord: <a href="https://www.google.co.uk/maps/place/{},{}" target="_blank">{}, {}</a><BR>Arr: {}<BR>Dep: {}{}' \
                    .format(u, la, lo, np.round(la, 4), np.round(lo, 4),
                            pd.datetime.strftime(t0, '%Y/%m/%d %H:%M'),
                            pd.datetime.strftime(t1, '%Y/%m/%d %H:%M'), cl), max_width=300)
                fpoly = fpoly.add_child(popup)

            fpoly.add_to(map_f)

    return map_f


def plot_diary(cstdf, user, start_datetime=None, end_datetime=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 2))

    df = cstdf[cstdf[constants.UID] == user]

    # TODO: add warning if days between start_datetime and end_datetime do not overlap with cstdf
    if start_datetime is None:
        start_datetime = df[constants.DATETIME].min()
    if end_datetime is None:
        end_datetime = df[constants.LEAVING_DATETIME].max()

    for idx, row in df.iterrows():

        t0 = row[constants.DATETIME]
        t1 = row[constants.LEAVING_DATETIME]
        cl = row[constants.CLUSTER]

        color = get_color(cl)
        if start_datetime <= t0 <= end_datetime:
            ax.axvspan(t0, t1, lw=0.0, alpha=0.75, color=color)

    # Backout times
    # TO DO

    plt.xlim(start_datetime, end_datetime)
    # plt.legend(loc='lower right', frameon=False)
    # plt.legend(ncol=15 ,bbox_to_anchor=(1., -0.2), frameon=0)
    ax.set_title('user %s' % user)

    return ax



flow_style_function = lambda color, weight, weight_factor, flow_exp: \
    (lambda feature: dict(color=color, weight=weight_factor * weight ** flow_exp, opacity=0.5)) #, dashArray='5, 5'))


def plot_flows(fdf, map_f=None, min_flow=0, tiles='Stamen Toner', zoom=6, flow_color='red', flow_weight=5,
               num_od_popup=5, flow_exp=0.5,
               style_function=flow_style_function, flow_popup=False, tile_popup=True, radius_origin_point=5,
               color_origin_point='#3186cc'):

    if map_f is None:
        # initialise map
        lon, lat = np.mean(np.array(list(fdf.tessellation.geometry.apply(utils.get_geom_centroid).values)), axis=0)
        map_f = folium.Map(location=[lat,lon], tiles=tiles, zoom_start=zoom)

    mean_flows = fdf[constants.FLOW].mean()

    O_groups = fdf.groupby(by=constants.ORIGIN)
    for O, OD in O_groups:

        geom = fdf.get_geometry(O)
        lonO, latO = utils.get_geom_centroid(geom)

        for D, T in OD[[constants.DESTINATION, constants.FLOW]].values:
            if O == D:
                continue
            if T < min_flow:
                continue

            geom = fdf.get_geometry(D)
            lonD, latD = utils.get_geom_centroid(geom)

            gjc = LineString([(lonO,latO), (lonD,latD)])

            fgeojson = folium.GeoJson(gjc,
                                      name='geojson',
                                      style_function = style_function(flow_color, T / mean_flows, flow_weight, flow_exp)
                                      )
            if flow_popup:
                popup = folium.Popup('flow from %s to %s: %s'%(O, D, int(T)), max_width=300)
                fgeojson = fgeojson.add_child(popup)

            fgeojson.add_to(map_f)

    if radius_origin_point > 0:
        for O, OD in O_groups:

            name = 'origin: %s' % O.replace('\'', '_')
            T_D = [[T, D] for D, T in OD[[constants.DESTINATION, constants.FLOW]].values]
            trips_info = '<br/>'.join(["flow to %s: %s" %
                                       (dd.replace('\'', '_'), int(tt)) \
                                       for tt, dd in sorted(T_D, reverse=True)[:num_od_popup]])

            geom = fdf.get_geometry(O)
            lonO, latO = utils.get_geom_centroid(geom)
            fmarker = folium.CircleMarker([latO, lonO],
                                          radius=radius_origin_point,
                                          weight=2,
                                          color=color_origin_point,
                                          fill=True, fill_color=color_origin_point
                                          )
            if tile_popup:
                popup = folium.Popup(name+'<br/>'+trips_info, max_width=300)
                fmarker = fmarker.add_child(popup)
            fmarker.add_to(map_f)

    return map_f



default_style_func_args = {'weight': 1, 'color': 'random', 'opacity': 0.5, 'fillColor': 'red', 'fillOpacity': 0.25}

geojson_style_function = lambda weight, color, opacity, fillColor, fillOpacity: \
    (lambda feature: dict(weight=weight, color=color, opacity=opacity, fillColor=fillColor, fillOpacity=fillOpacity))


def add_to_map(gway, g, map_osm, style_func_args, popup_features=[]):
    weight, color, opacity, fillColor, fillOpacity = [
        style_func_args[k] if k in style_func_args else default_style_func_args[k]
        for k in ['weight', 'color', 'opacity', 'fillColor', 'fillOpacity']]

    if type(gway) == shapely.geometry.multipolygon.MultiPolygon:

        # Multipolygon
        for gg in gway:
            if color == 'random':
                color = random_hex()
                fillColor = color

            vertices = list(zip(*gg.exterior.xy))
            gj = folium.GeoJson({"type": "Polygon", "coordinates": [vertices]},
                                style_function=geojson_style_function(weight=weight, color=color, opacity=opacity,
                                                                      fillColor=fillColor, fillOpacity=fillOpacity))

    elif type(gway) == shapely.geometry.polygon.Polygon:

        # Polygon
        if color == 'random':
            color = random_hex()
            fillColor = color

        vertices = list(zip(*gway.exterior.xy))
        gj = folium.GeoJson({"type": "Polygon", "coordinates": [vertices]},
                            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity,
                                                                  fillColor=fillColor, fillOpacity=fillOpacity))

    elif type(gway) == shapely.geometry.multilinestring.MultiLineString:

        # MultiLine
        for gg in gway:
            if color == 'random':
                color = random_hex()
                fillColor = color

            vertices = list(zip(*gg.xy))
            gj = folium.GeoJson({"type": "LineString", "coordinates": vertices},
                                style_function=geojson_style_function(weight=weight, color=color, opacity=opacity,
                                                                      fillColor=fillColor, fillOpacity=fillOpacity))

    elif type(gway) == shapely.geometry.linestring.LineString:

        # LineString
        if color == 'random':
            color = random_hex()
            fillColor = color
        vertices = list(zip(*gway.xy))
        gj = folium.GeoJson({"type": "LineString", "coordinates": vertices},
                            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity,
                                                                  fillColor=fillColor, fillOpacity=fillOpacity))

    else:

        # Point
        if color == 'random':
            color = random_hex()
            fillColor = color

        point = list(zip(*gway.xy))[0]
        #         gj = folium.CircleMarker(
        gj = folium.Circle(
            location=point[::-1],
            radius=5,
            color=color,  # '#3186cc',
            fill=True,
            fill_color=fillColor
        )

    popup = []
    for pf in popup_features:
        try:
            popup += ['%s: %s' % (pf, g[pf])]
        except KeyError:
            pass

    try:
        popup = '<br>'.join(popup)
        popup += json.dumps(g.tags)
        popup = popup.replace("""'""", """_""")
    except AttributeError:
        pass
    if len(popup) > 0:
        gj.add_child(folium.Popup(popup, max_width=300))

    gj.add_to(map_osm)

    return map_osm


def plot_gdf(gdf, map_osm=None, maxitems=-1, style_func_args={}, popup_features=[],
            tiles='Stamen Toner', zoom=6, geom_col='geometry'):

    if map_osm is None:
        # initialise map
        lon, lat = np.mean(np.array(list(gdf[geom_col].apply(utils.get_geom_centroid).values)), axis=0)
        map_osm = folium.Map(location=[lat, lon], tiles=tiles, zoom_start=zoom)

    count = 0
    for k in gdf.index:
        g = gdf.loc[k]

        if type(g[geom_col]) == gpd.geoseries.GeoSeries:
            for i in range(len(g[geom_col])):
                map_osm = add_to_map(g[geom_col].iloc[i], g.iloc[i], map_osm,
                                     popup_features=popup_features,
                                     style_func_args=style_func_args)
        else:
            map_osm = add_to_map(g[geom_col], g, map_osm,
                                 popup_features=popup_features,
                                 style_func_args=style_func_args)

        count += 1
        if count == maxitems:
            break

    return map_osm

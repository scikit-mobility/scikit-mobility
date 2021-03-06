from ..utils import constants, utils
import folium
from folium.plugins import HeatMap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from geojson import LineString
import geopandas as gpd
import json
import warnings


# COLOR = {
#     0: '#FF0000',  # Red
#     1: '#008000',  # Green
#     2: '#000080',  # Navy
#     3: '#800000',  # Maroon
#     4: '#FFD700',  # Gold
#     5: '#00FF00',  # Lime
#     6: '#800080',  # Purple
#     7: '#00FFFF',  # Aqua
#     8: '#DC143C',  # Crimson
#     9: '#0000FF',  # Blue
#     10: '#F08080',  # LightCoral
#     11: '#FF00FF',  # Fuchsia
#     12: '#FF8C00',  # DarkOrange
#     13: '#6A5ACD',  # SlateBlue
#     14: '#8B4513',  # SaddleBrown
#     15: '#1E90FF',  # DodgerBlue
#     16: '#FFFF00',  # Yellow
#     17: '#808080',  # Gray
#     18: '#008080',  # Teal
#     19: '#9370DB',  # MediumPurple
#     20: '#2F4F4F'  # DarkSlateGray
# }


# PALETTE FOR COLOR BLINDNESS
# from <http://mkweb.bcgsc.ca/colorblind/palettes.mhtml>
COLOR = {
    0:  '#6A0213',
    1:  '#008607',
    2:  '#F60239',
    3:  '#00E307',
    4:  '#FFDC3D',
    5:  '#003C86',
    6:  '#9400E6',
    7:  '#009FFA',
    8:  '#FF71FD',
    9:  '#7CFFFA',
    10: '#68023F',
    11: '#008169',
    12: '#EF0096',
    13: '#00DCB5',
    14: '#FFCFE2'
}


def get_color(k=-2, color_dict=COLOR):
    """
    Return a color (gray if k == -1, random if k < -1)
    """
    if k < -1:
        return np.random.choice(list(color_dict.values()))  # color_dict[random.randint(0,20)]
    elif k == -1:
        return '#808080'  # Gray
    else:
        return color_dict[k % len(color_dict)]


def random_hex():
    r = lambda: np.random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())


traj_style_function = lambda weight, color, opacity, dashArray: \
    (lambda feature: dict(color=color, weight=weight, opacity=opacity, dashArray=dashArray))


def plot_trajectory(tdf, map_f=None, max_users=10, max_points=1000, style_function=traj_style_function,
                    tiles='cartodbpositron', zoom=12, hex_color=None, weight=2, opacity=0.75, dashArray='0, 0',
                    start_end_markers=True, control_scale=True):


    """
    :param tdf: TrajDataFrame
         TrajDataFrame to be plotted.

    :param map_f: folium.Map
        `folium.Map` object where the trajectory will be plotted. If `None`, a new map will be created.

    :param max_users: int
        maximum number of users whose trajectories should be plotted.

    :param max_points: int
        maximum number of points per user to plot.
        If necessary, a user's trajectory will be down-sampled to have at most `max_points` points.

    :param style_function: lambda function
        function specifying the style (weight, color, opacity) of the GeoJson object.

    :param tiles: str
        folium's `tiles` parameter.

    :param zoom: int
        initial zoom.

    :param hex_color: str
        hex color of the trajectory line. If `None` a random color will be generated for each trajectory.

    :param weight: float
        thickness of the trajectory line.

    :param opacity: float
        opacity (alpha level) of the trajectory line.

    :param dashArray: str
        style of the trajectory line: '0, 0' for a solid trajectory line, '5, 5' for a dashed line
        (where dashArray='size of segment, size of spacing').

    :param start_end_markers: bool
        add markers on the start and end points of the trajectory.

    :param control_scale: bool
        if `True`, add scale information in the bottom left corner of the visualization. The default is `True`.

    Returns
    -------
        `folium.Map` object with the plotted trajectories.

    """
    warnings.warn("Only the trajectories of the first 10 users will be plotted. Use the argument `max_users` to specify the desired number of users, or filter the TrajDataFrame.")

    # group by user and keep only the first `max_users`
    nu = 0

    try:
        # column 'uid' is present in the TrajDataFrame
        groups = tdf.groupby(constants.UID)
    except KeyError:
        # column 'uid' is not present
        groups = [[None, tdf]]

    warned = False
    for user, df in groups:

        if nu >= max_users:
            break
        nu += 1

        traj = df[[constants.LONGITUDE, constants.LATITUDE]]

        if max_points is None:
            di = 1
        else:
            if not warned: 
                warnings.warn("If necessary, trajectories will be down-sampled to have at most `max_points` points. To avoid this, sepecify `max_points=None`.")
                warned = True
            di = max(1, len(traj) // max_points)
        traj = traj[::di]

        if nu == 1 and map_f is None:
            # initialise map
            center = list(np.median(traj, axis=0)[::-1])
            map_f = folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=control_scale)

        trajlist = traj.values.tolist()
        line = LineString(trajlist)

        if hex_color is None:
            color = get_color(-2)
        else:
            color = hex_color

        tgeojson = folium.GeoJson(line,
                                  name='tgeojson',
                                  style_function=style_function(weight, color, opacity, dashArray)
                                  )
        tgeojson.add_to(map_f)

        if start_end_markers:

            dtime, la, lo = df.loc[df['datetime'].idxmin()]\
                [[constants.DATETIME, constants.LATITUDE, constants.LONGITUDE]].values
            dtime = dtime.strftime('%Y/%m/%d %H:%M')
            mker = folium.Marker(trajlist[0][::-1], icon=folium.Icon(color='green'))
            popup = folium.Popup('<i>Start</i><BR>{}<BR>Coord: <a href="https://www.google.co.uk/maps/place/{},{}" target="_blank">{}, {}</a>'.\
                          format(dtime, la, lo, np.round(la, 4), np.round(lo, 4)), max_width=300)
            mker = mker.add_child(popup)
            mker.add_to(map_f)

            dtime, la, lo = df.loc[df['datetime'].idxmax()]\
                [[constants.DATETIME, constants.LATITUDE, constants.LONGITUDE]].values
            dtime = dtime.strftime('%Y/%m/%d %H:%M')
            mker = folium.Marker(trajlist[-1][::-1], icon=folium.Icon(color='red'))
            popup = folium.Popup('<i>End</i><BR>{}<BR>Coord: <a href="https://www.google.co.uk/maps/place/{},{}" target="_blank">{}, {}</a>'.\
                          format(dtime, la, lo, np.round(la, 4), np.round(lo, 4)), max_width=300)
            mker = mker.add_child(popup)
            mker.add_to(map_f)

    return map_f

def plot_points_heatmap(tdf, map_f=None, max_points=1000, 
                        tiles='cartodbpositron', zoom=2,
                       min_opacity=0.5, radius=25, blur=15,
                       gradient=None):
    """
    Plot the points in a trajectories on a Folium map.

    Parameters
    ----------
    map_f : folium.Map, optional
        a `folium.Map` object where the trajectory will be plotted. If `None`, a new map will be created. The default is `None`.

    max_points : int, optional
        maximum number of points per individual to plot. The default is `1000`. If necessary, an individual's trajectory will be down-sampled to have at most `max_points` points.

    tiles : str, optional
        folium's `tiles` parameter. The default is 'cartodbpositron'.

    zoom : int, optional
        the initial zoom on the map. The default is `2`.

    min_opacity : float, optional
        the minimum opacity (alpha level) the heat will start at. The default is `0.5`.

    radius : int, optional
        radius of each "point" of the heatmap. The default is `25`.
    
    blur : int, optional
        amount of blur. The default is blur 15.
        
    gradient : dict, optional 
        color gradient configuration, e.g. {0.4: ‘blue’, 0.65: ‘lime’, 1: ‘red’}. The default is `None`.
    
    Returns
    -------
    folium.Map
        a `folium.Map` object with the plotted trajectories.

    Examples
    --------
    >>> import skmob
    >>> import pandas as pd
    >>> # read the trajectory data (GeoLife, Beijing, China)
    >>> url = 'https://raw.githubusercontent.com/scikit-mobility/scikit-mobility/master/tutorial/data/geolife_sample.txt.gz'
    >>> df = pd.read_csv(url, sep=',', compression='gzip')
    >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
    >>> print(tdf.head())
             lat         lng            datetime  uid
    0  39.984094  116.319236 2008-10-23 05:53:05    1
    1  39.984198  116.319322 2008-10-23 05:53:06    1
    2  39.984224  116.319402 2008-10-23 05:53:11    1
    3  39.984211  116.319389 2008-10-23 05:53:16    1
    4  39.984217  116.319422 2008-10-23 05:53:21    1
    >>> m = tdf.plot_points_heatmap(zoom=12, opacity=0.9, tiles='Stamen Toner')
    >>> m
    """   
    if max_points is None:
        di = 1
    else:
        di = max(1, len(tdf) // max_points)
    traj = tdf[::di]
    traj = traj[[constants.LATITUDE, constants.LONGITUDE]]

    if map_f is None:
        center = list(np.median(traj[[constants.LONGITUDE, constants.LATITUDE]], axis=0)[::-1])
        map_f = folium.Map(zoom_start=zoom, tiles=tiles, control_scale=True, location=center)
    HeatMap(traj.values, 
            min_opacity=min_opacity, radius=radius,
           blur=blur, gradient=gradient).add_to(map_f)
    
    return map_f

def plot_stops(stdf, map_f=None, max_users=10, tiles='cartodbpositron', zoom=12, hex_color=None, opacity=0.3,
               radius=12, number_of_sides=4, popup=True, control_scale=True):

    """
    :param stdf: TrajDataFrame
         Requires a TrajDataFrame with stops or clusters, output of `preprocessing.detection.stops`
         or `preprocessing.clustering.cluster`. The column `constants.LEAVING_DATETIME` must be present.

    :param map_f: folium.Map
        `folium.Map` object where the stops will be plotted. If `None`, a new map will be created.

    :param max_users: int
        maximum number of users whose stops should be plotted.

    :param tiles: str
        folium's `tiles` parameter.

    :param zoom: int
        initial zoom.

    :param hex_color: str
        hex color of the stop markers. If `None` a random color will be generated for each user.

    :param opacity: float
        opacity (alpha level) of the stop makers.

    :param radius: float
        size of the markers.

    :param number_of_sides: int
        number of sides of the markers.

    :param popup: bool
        if `True`, when clicking on a marker a popup window displaying information on the stop will appear.

    :param control_scale: bool
        if `True`, add scale information in the bottom left corner of the visualization. The default is `True`.

    Returns
    -------
        `folium.Map` object with the plotted stops.

    """
    warnings.warn("Only the stops of the first 10 users will be plotted. Use the argument `max_users` to specify the desired number of users, or filter the TrajDataFrame.")

    if map_f is None:
        # initialise map
        lo_la = stdf[['lng', 'lat']].values
        center = list(np.median(lo_la, axis=0)[::-1])
        map_f = folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=control_scale)

    # group by user and keep only the first `max_users`
    nu = 0

    try:
        # column 'uid' is present in the TrajDataFrame
        groups = stdf.groupby(constants.UID)
    except KeyError:
        # column 'uid' is not present
        groups = [[None, stdf]]

    for user, df in groups:
        if nu >= max_users:
            break
        nu += 1

        if hex_color is None:
            color = get_color(-2)
        else:
            color = hex_color

        for idx, row in df.iterrows():

            la = row[constants.LATITUDE]
            lo = row[constants.LONGITUDE]
            t0 = row[constants.DATETIME]
            try:
                t1 = row[constants.LEAVING_DATETIME]
                _number_of_sides = number_of_sides
                marker_radius = radius
            except KeyError:
                t1 = t0
                _number_of_sides = number_of_sides
                marker_radius = radius // 2
            u = user
            try:
                ncluster = row[constants.CLUSTER]
                cl = '<BR>Cluster: {}'.format(ncluster)
                color = get_color(ncluster)
            except (KeyError, NameError):
                cl = ''

            fpoly = folium.RegularPolygonMarker([la, lo],
                                        radius=marker_radius,
                                        color=color,
                                        fill_color=color,
                                        fill_opacity=opacity,
                                        number_of_sides=_number_of_sides
                                        )
            if popup:
                popup = folium.Popup('User: {}<BR>Coord: <a href="https://www.google.co.uk/maps/place/{},{}" target="_blank">{}, {}</a><BR>Arr: {}<BR>Dep: {}{}' \
                    .format(u, la, lo, np.round(la, 4), np.round(lo, 4),
                            t0.strftime('%Y/%m/%d %H:%M'),
                            t1.strftime('%Y/%m/%d %H:%M'), cl), max_width=300)
                fpoly = fpoly.add_child(popup)

            fpoly.add_to(map_f)

    return map_f


def plot_diary(cstdf, user, start_datetime=None, end_datetime=None, ax=None, legend=False):
    """
        Plot a mobility diary of an individual in a TrajDataFrame. It requires a TrajDataFrame with clusters, output of `preprocessing.clustering.cluster`. The column `constants.CLUSTER` must be present.

        Parameters
        ----------
        user : str or int
            user identifier whose diary should be plotted.

        start_datetime : datetime.datetime, optional
            only stops made after this date will be plotted. If `None` the datetime of the oldest stop will be selected. The default is `None`.

        end_datetime : datetime.datetime, optional
            only stops made before this date will be plotted. If `None` the datetime of the newest stop will be selected. The default is `None`.

        ax : matplotlib.axes, optional
            axes where the diary will be plotted. If `None` a new ax is created. The default is `None`.

        legend : bool, optional
            If `True`, legend with cluster IDs is shown. The default is `False`.

        Returns
        -------
        matplotlib.axes
            the `matplotlib.axes` object of the plotted diary.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 2))

    if user is None:
        df = cstdf
    else:
        df = cstdf[cstdf[constants.UID] == user]

    if len(df) == 0:
        raise KeyError("""User id is not in the input TrajDataFrame.""")

    # TODO: add warning if days between start_datetime and end_datetime do not overlap with cstdf
    if start_datetime is None:
        start_datetime = df[constants.DATETIME].min()
    elif type(start_datetime) is str:
        start_datetime = pd.to_datetime(start_datetime)
    if end_datetime is None:
        end_datetime = df[constants.LEAVING_DATETIME].max()
    elif type(end_datetime) is str:
        end_datetime = pd.to_datetime(end_datetime)

    current_labels = []

    for idx, row in df.iterrows():

        t0 = row[constants.DATETIME]
        t1 = row[constants.LEAVING_DATETIME]
        cl = row[constants.CLUSTER]

        color = get_color(cl)
        if start_datetime <= t0 <= end_datetime:
            if cl in current_labels:
                ax.axvspan(t0.to_pydatetime(), t1.to_pydatetime(), lw=0.0, alpha=0.75, color=color)
            else:
                current_labels += [cl]
                ax.axvspan(t0.to_pydatetime(), t1.to_pydatetime(), lw=0.0, alpha=0.75, color=color, label=cl)

    plt.xlim(start_datetime, end_datetime)

    if legend:
        handles, labels_str = ax.get_legend_handles_labels()
        labels = list(map(int, labels_str))
        # sort them by labels
        import operator
        hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
        handles2, labels2 = zip(*hl)

        ax.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1., -0.2), frameon=0)

    ax.set_title('user %s' % user)

    return ax



flow_style_function = lambda weight, color, opacity, weight_factor, flow_exp: \
    (lambda feature: dict(color=color, weight=weight_factor * weight ** flow_exp, opacity=opacity)) #, dashArray='5, 5'))


def plot_flows(fdf, map_f=None, min_flow=0, tiles='cartodbpositron', zoom=6, flow_color='red', opacity=0.5,
               flow_weight=5, flow_exp=0.5, style_function=flow_style_function,
               flow_popup=False, num_od_popup=5, tile_popup=True, radius_origin_point=5,
               color_origin_point='#3186cc', control_scale=True):
    """
    :param fdf: FlowDataFrame
        `FlowDataFrame` to visualize.

    :param map_f: folium.Map
        `folium.Map` object where the flows will be plotted. If `None`, a new map will be created.

    :param min_flow: float
        only flows larger than `min_flow` will be plotted.

    :param tiles: str
        folium's `tiles` parameter.

    :param zoom: int
        initial zoom.

    :param flow_color: str
        color of the flow edges

    :param opacity: float
        opacity (alpha level) of the flow edges.

    :param flow_weight: float
        weight factor used in the function to compute the thickness of the flow edges.

    :param flow_exp: float
        weight exponent used in the function to compute the thickness of the flow edges.

    :param style_function: lambda function
        GeoJson style function.

    :param flow_popup: bool
        if `True`, when clicking on a flow edge a popup window displaying information on the flow will appear.

    :param num_od_popup: int
        number of origin-destination pairs to show in the popup window of each origin location.

    :param tile_popup: bool
        if `True`, when clicking on a location marker a popup window displaying information on the flows
        departing from that location will appear.

    :param radius_origin_point: float
        size of the location markers.

    :param color_origin_point: str
        color of the location markers.

    :param control_scale: bool
        if `True`, add scale information in the bottom left corner of the visualization. The default is `True`.

    Returns
    -------
        `folium.Map` object with the plotted flows.

    """
    if map_f is None:
        # initialise map
        lon, lat = np.mean(np.array(list(fdf.tessellation.geometry.apply(utils.get_geom_centroid).values)), axis=0)
        map_f = folium.Map(location=[lat,lon], tiles=tiles, zoom_start=zoom, control_scale=control_scale)

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
                                      style_function = style_function(T / mean_flows, flow_color, opacity,
                                                                      flow_weight, flow_exp)
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


default_style_func_args = {'weight': 1, 'color': 'random', 'opacity': 0.5,
                           'fillColor': 'random', 'fillOpacity': 0.25, 'radius': 5}

geojson_style_function = lambda weight, color, opacity, fillColor, fillOpacity: \
    (lambda feature: dict(weight=weight, color=color, opacity=opacity, fillColor=fillColor,
                          fillOpacity=fillOpacity))


def manage_colors(color, fillColor):
    if color == 'random':
        if fillColor == 'random':
            color = random_hex()
            fillColor = color
        else:
            color = random_hex()
    elif fillColor == 'random':
        fillColor = random_hex()
    return color, fillColor


def add_to_map(gway, g, map_f, style_func_args, popup_features=[]):

    styles = []
    for k in ['weight', 'color', 'opacity', 'fillColor', 'fillOpacity', 'radius']:
        if k in style_func_args:
            if callable(style_func_args[k]):
                styles += [style_func_args[k](g)]
            else:
                styles += [style_func_args[k]]
        else:
            styles += [default_style_func_args[k]]
    weight, color, opacity, fillColor, fillOpacity, radius = styles

    color, fillColor = manage_colors(color, fillColor)

    if type(gway) == shapely.geometry.multipolygon.MultiPolygon:

        # Multipolygon
        vertices = [list(zip(*p.exterior.xy)) for p in gway]
        gj = folium.GeoJson({"type": "MultiPolygon", "coordinates": [vertices]},
                            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity,
                                                                  fillColor=fillColor, fillOpacity=fillOpacity))

    elif type(gway) == shapely.geometry.polygon.Polygon:

        # Polygon
        vertices = list(zip(*gway.exterior.xy))
        gj = folium.GeoJson({"type": "Polygon", "coordinates": [vertices]},
                            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity,
                                                                  fillColor=fillColor, fillOpacity=fillOpacity))

    elif type(gway) == shapely.geometry.multilinestring.MultiLineString:

        # MultiLine
        vertices = [list(zip(*l.xy)) for l in gway]
        gj = folium.GeoJson({"type": "MultiLineString", "coordinates": vertices},
                            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity,
                                                                  fillColor=fillColor, fillOpacity=fillOpacity))

    elif type(gway) == shapely.geometry.linestring.LineString:

        # LineString
        vertices = list(zip(*gway.xy))
        gj = folium.GeoJson({"type": "LineString", "coordinates": vertices},
                            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity,
                                                                  fillColor=fillColor, fillOpacity=fillOpacity))

    else:

        # Point
        point = list(zip(*gway.xy))[0]
        #         gj = folium.CircleMarker(
        gj = folium.Circle(
            location=point[::-1],
            radius=radius,
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

    gj.add_to(map_f)

    return map_f


def plot_gdf(gdf, map_f=None, maxitems=-1, style_func_args={}, popup_features=[],
             tiles='cartodbpositron', zoom=6, geom_col='geometry', control_scale=True):

    """
    :param gdf: GeoDataFrame
        GeoDataFrame to visualize.

    :param map_f: folium.Map
        `folium.Map` object where the GeoDataFrame `gdf` will be plotted. If `None`, a new map will be created.

    :param maxitems: int
        maximum number of tiles to plot. If `-1`, all tiles will be plotted.

    :param style_func_args: dict
        dictionary to pass the following style parameters (keys) to the GeoJson style function of the polygons:
        'weight', 'color', 'opacity', 'fillColor', 'fillOpacity', 'radius'

    :param popup_features: list
        when clicking on a tile polygon, a popup window displaying the information in the
        columns of `gdf` listed in `popup_features` will appear.

    :param tiles: str
        folium's `tiles` parameter.

    :param zoom: int
        initial zoom.

    :param geom_col: str
         name of the geometry column of `gdf`.

    :param control_scale: bool
        if `True`, add scale information in the bottom left corner of the visualization. The default is `True`.

    Returns
    -------
        `folium.Map` object with the plotted GeoDataFrame.

    """
    if map_f is None:
        # initialise map
        lon, lat = np.mean(np.array(list(gdf[geom_col].apply(utils.get_geom_centroid).values)), axis=0)
        map_f = folium.Map(location=[lat, lon], tiles=tiles, zoom_start=zoom, control_scale=control_scale)



    count = 0
    for k in gdf.index:
        g = gdf.loc[k]

        if type(g[geom_col]) == gpd.geoseries.GeoSeries:
            for i in range(len(g[geom_col])):
                map_f = add_to_map(g[geom_col].iloc[i], g.iloc[i], map_f,
                                     popup_features=popup_features,
                                     style_func_args=style_func_args)
        else:
            map_f = add_to_map(g[geom_col], g, map_f,
                                 popup_features=popup_features,
                                 style_func_args=style_func_args)

        count += 1
        if count == maxitems:
            break

    return map_f

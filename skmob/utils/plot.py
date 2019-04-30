from ..utils import constants
import folium
from geojson import LineString
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_trajectory(tdf, map_f=None, max_users=10, max_points=1000, imin=0, imax=-1,
                    tiles='OpenStreetMap', zoom=12, hex_color=-1, weight=2, opacity=0.75):
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


def plot_stops(stdf, map_f=None, max_users=10, tiles='OpenStreetMap', zoom=12,
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

            if popup:
                popup_str = 'User: {}<BR>Coord: <a href="https://www.google.co.uk/maps/place/{},{}" target="_blank">{}, {}</a><BR>Arr: {}<BR>Dep: {}{}' \
                    .format(u, la, lo, np.round(la, 4), np.round(lo, 4),
                            pd.datetime.strftime(t0, '%Y/%m/%d %H:%M'),
                            pd.datetime.strftime(t1, '%Y/%m/%d %H:%M'), cl)
                folium.RegularPolygonMarker([la, lo],
                                            radius=12,
                                            popup=popup_str,
                                            color=color,
                                            fill_color=color,
                                            fill_opacity=opacity
                                            ).add_to(map_f)
            else:
                folium.RegularPolygonMarker([la, lo],
                                            radius=12,
                                            color=color,
                                            fill_color=color,
                                            fill_opacity=opacity
                                            ).add_to(map_f)

    return map_f


def plot_diary(cstdf, user, start_datetime=None, end_datetime=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 2))

    df = cstdf[cstdf[constants.UID] == user]

    if start_datetime is None:
        start_datetime = df.datetime.min()
    if end_datetime is None:
        end_datetime = df.leaving_datetime.max()

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

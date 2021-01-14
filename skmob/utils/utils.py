from ..utils import constants, gislib
import pandas as pd
import geopandas as gpd
import shapely
import os
import errno
import requests
import numpy as np

distance = gislib.getDistance


def diff_seconds(t_0, t_1):
    return (t_1 - t_0).total_seconds()


def is_multi_user(data):
    if constants.UID in data.columns:
        return True

    return False


def is_multi_trajectory(data):
    if constants.TID in data.columns:
        return True

    return False


def to_matrix(data, columns=None):
    if columns is None:
        columns = [constants.LATITUDE, constants.LONGITUDE, constants.DATETIME]
        columns = columns + list(set(data.columns) - set(columns))

    return data[columns].values


def get_columns(data):
    columns = [constants.LATITUDE, constants.LONGITUDE, constants.DATETIME]
    columns = columns + list(set(data.columns) - set(columns))

    return columns


def to_dataframe(data, columns):
    # Reorder columns to maintain the original order
    df = pd.DataFrame(data, columns=columns)

    return df


def assign_crs(shape, crs):
    if crs is not None:
        return shape.to_crs(crs)

    return shape


def to_geodataframe(df, keep=False, latitude=constants.LATITUDE, longitude=constants.LONGITUDE,
                    crs=constants.DEFAULT_CRS):
    geometry = [shapely.geometry.Point(xy) for xy in zip(df[longitude], df[latitude])]

    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    df.drop('geometry', inplace=True, axis=1)

    if keep is False:
        return gdf.drop([longitude, latitude], axis=1)

    return gdf


def silentremove(filename):
    try:

        os.remove(filename)

    except OSError as e:  # this would be "except OSError, e:" before Python 2.6

        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def setattrpandas(df, name, value):
    df._metadata.append(name)
    df.name = value

    return df


def group_df_by_time(tdf, freq_str='1D', offset_value=0, offset_unit='hours', add_starting_location=False,
                     dtformat='%Y-%m-%d %H:%M:%S'):
    """
    Split a `TrajDataFrame` into subtrajectories of fixed temporal length (`freq_str`).

    :param tdf: TrajDataFrame
        `TrajDataFrame` to split.

    :param freq_str: str
        `freq` parameter of `pd.date_range`. default: '1D' (one day).

    :param offset_value: float
        value of the offset time used to shift the start time of each subtrajectories.

    :param offset_unit: str
        time unit of the offset time used to shift the start time of each subtrajectories.

    :param add_starting_location: bool
        if `True`, the last point of the previous subtrajectory will be appended
         at the beginning of the next subtrajectory.

    :param dtformat: str
        datetime format.

    :return: list containing the subtrajectories

    """
    df = tdf.sort_values([constants.DATETIME])

    offset = pd.Timedelta(offset_value, offset_unit)  # datetime.timedelta(seconds=offset_secs)
    t_init = pd.to_datetime(df[constants.DATETIME].min().date())
    t_end = pd.to_datetime(df[constants.DATETIME].max().date()) + pd.Timedelta(days=1)  # datetime.timedelta(days=1)

    rg = pd.date_range((t_init + offset).strftime(dtformat), end=(t_end + offset).strftime(dtformat), freq=freq_str)

    groups = []
    for st, en in list(zip(rg, rg[1:])):
        dfslice = df.loc[(df[constants.DATETIME] >= st.strftime(dtformat)) &
                         (df[constants.DATETIME] < en.strftime(dtformat))]
        if len(dfslice) > 0:
            #             groups[st.date()] = [dfslice]
            groups += [dfslice.reset_index(drop=True)]

    if add_starting_location:
        for i in range(1, len(groups)):
            groups[i] = pd.concat([groups[i - 1][-1:], groups[i]]).reset_index(drop=True)

    return groups


def frequency_vector(trajectory):
    freq = trajectory.groupby([constants.UID, constants.LATITUDE, constants.LONGITUDE]).size().reset_index(
        name=constants.FREQUENCY)
    return freq.sort_values(by=[constants.UID, constants.FREQUENCY])


def probability_vector(trajectory):
    freq = trajectory.groupby([constants.UID, constants.LATITUDE, constants.LONGITUDE]).size().reset_index(
        name=constants.FREQUENCY)
    prob = pd.merge(freq, trajectory.groupby(constants.UID).size().reset_index(name=constants.TOTAL_FREQ),
                    left_on=constants.UID, right_on=constants.UID)
    prob[constants.PROBABILITY] = prob[constants.FREQUENCY] / prob[constants.TOTAL_FREQ]
    return prob[[constants.UID, constants.LATITUDE, constants.LONGITUDE, constants.PROBABILITY]].sort_values(
        by=[constants.UID, constants.PROBABILITY])


def date_time_precision(dt, precision):
    result = ""
    if precision == "Year" or precision == "year":
        result += str(dt.year)
    elif precision == "Month" or precision == "month":
        result += str(dt.year) + str(dt.month)
    elif precision == "Day" or precision == "day":
        result += str(dt.year) + str(dt.month) + str(dt.day)
    elif precision == "Hour" or precision == "hour":
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month)
    elif precision == "Minute" or precision == "minute":
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month) + str(dt.minute)
    elif precision == "Second" or precision == "second":
        result += str(dt.year) + str(dt.month) + str(dt.day) + str(dt.month) + str(dt.minute) + str(dt.second)
    return result


def bbox_from_points(points, crs=None):
    try:
        coords = points.total_bounds
    except AttributeError:
        coords = points

    base = shapely.geometry.box(coords[0], coords[1], coords[2], coords[3], ccw=True)
    base = gpd.GeoDataFrame(geometry=[base], crs=constants.DEFAULT_CRS)

    if crs is None:
        return base

    return base.to_crs(crs)


def bbox_from_area(area, bbox_side_len=500, crs=None):
    centroid = area.iloc[0].geometry.centroid

    # get North-East corner
    ne = [float(coord) + (bbox_side_len / 2) for coord in centroid]

    # get South-West corner
    sw = [float(coord) - (bbox_side_len / 2) for coord in centroid]

    # build bbox from NE,SW corners
    bbox = shapely.geometry.box(sw[0], sw[1], ne[0], ne[1], ccw=True)

    base = gpd.GeoDataFrame(geometry=[bbox], crs=constants.DEFAULT_CRS)

    if crs is None:
        return base

    return base.to_crs(crs)


def bbox_from_name(query, which_osm_result=0, crs=None):
    """
    Create a GeoDataFrame from a single place name query.
    (adapted from https://github.com/gboeing/osmnx)

    Parameters
    ----------
    query : string or dict
        query string or structured query dict to geocode/download

    which_osm_result : int
        number of result to return (`which_osm_result=-1` to return all results)

    Returns
    -------
    GeoDataFrame

    Example
    -------

    gdf = gdf_from_string("Florence, Italy")

    """
    nominatim_url = "https://nominatim.openstreetmap.org/search.php?" + \
                    "q=%s&polygon_geojson=1&format=json" % query

    response = requests.get(nominatim_url)
    data = response.json()

    if len(data) > 0:

        features = []
        for result in data:
            # extract data elements from the JSON response
            bbox_south, bbox_north, bbox_west, bbox_east = [float(x) for x in result['boundingbox']]

            coords = result['geojson']['coordinates']
            try:
                # it is a MultiPolygon
                geometry = shapely.geometry.MultiPolygon( \
                    [shapely.geometry.Polygon(c[0], [inner for inner in c[1:]]) for c in coords])
            except TypeError:
                try:
                    # it is a Polygon
                    geometry = shapely.geometry.MultiPolygon( \
                        [shapely.geometry.Polygon(coords[0], [inner for inner in coords[1:]])])
                except TypeError:
                    # it is something else
                    geometry = result['geojson']

            place = result['display_name']
            features += [{'type': 'Feature',
                          'geometry': geometry,
                          'properties': {'place_name': place,
                                         'bbox_north': bbox_north,
                                         'bbox_south': bbox_south,
                                         'bbox_east': bbox_east,
                                         'bbox_west': bbox_west}}]

        gdf = gpd.GeoDataFrame.from_features(features)

        if crs is None:
            gdf.crs = constants.DEFAULT_CRS
        else:
            gdf = gdf.to_crs(crs)

        if which_osm_result >= 0:
            gdf = gdf.loc[[which_osm_result]]

    else:
        gdf = gpd.GeoDataFrame()

    return gdf


def nearest(origin, tessellation, col):
    """
    Brute force approach to find, for each point in a geodataframe, the nearest point into another geodataframe. It
    returns a Pandas Series containing the value in col for each matching row.
    :param origin: GeoDataFrame
    :param tessellation: GeoDataFrame
    :param col: column containing the value to return from the tessellation
    :return: Series
    """

    def _nearest(df, points):

        near = float("+inf")
        point = None

        main = (df['geometry'].y, df['geometry'].x)

        for index, row in points.iterrows():

            p = row['geometry']
            d = distance(main, (p.y, p.x))

            if d < near:
                near = d
                point = index

        return point

    return tessellation.iloc[origin.apply(_nearest, args=(tessellation,), axis=1)][col]


def get_geom_centroid(geom, return_lat_lng=False):
    """
    Compute the centroid of a Polygon or Multipolygon.

    :param geom: shapely Polygon or Multipolygon
        geometry, Polygon or Multipolygon, whose centroid will be computed.

    :param return_lat_lng: bool
        if `True`, the first coordinate in the returned list is the centroid's latitude, otherwise it is the longitude.

    :return: list
        coordinates of `geom`'s centroid.

    """
    lng, lat = map(lambda x: x.pop(), geom.centroid.xy)
    if return_lat_lng:
        return [lat, lng]
    else:
        return [lng, lat]


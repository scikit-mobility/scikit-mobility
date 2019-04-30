from ..utils import constants
import pandas as pd
import geopandas as gpd
import shapely
import os
import errno
from shapely.ops import nearest_points
import osmnx
from ..core.trajectorydataframe import TrajDataFrame

LATITUDE = constants.LATITUDE
LONGITUDE = constants.LONGITUDE
DATETIME = constants.DATETIME
UID = constants.UID
FREQUENCY = "freq"
PROBABILITY = "prob"
TOTAL_FREQ = "T_freq"
COUNT = "count"
TEMP = "tmp"
PROPORTION = "prop"
PRECISION_LEVELS = ["Year", "Month", "Day", "Hour", "Minute", "Second", "year", "month", "day", "hour", "minute",
                    "second"]
PRIVACY_RISK = "risk"
INSTANCE = "instance"
REIDENTIFICATION_PROBABILITY = "reid_prob"


def diff_seconds(t_0, t_1):
    return (t_1-t_0).total_seconds()


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
    # Reoder colmuns to maintain the original order
    df = pd.DataFrame(data, columns=columns)

    return df


def nparray_to_trajdataframe(trajectory_array, columns, parameters={}):
    df = pd.DataFrame(trajectory_array, columns=columns)
    tdf = TrajDataFrame(df, parameters=parameters)
    return tdf


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


def group_df_by_time(df, freq_str='1D', offset_value=0, offset_unit='hours', add_starting_location=False,
                     dtformat='%Y-%m-%d %H:%M:%S'):
    """
    freq_str  :  `freq` parameter of `pd.date_range`. default: '1D'
    offset_value  :
    offset_unit  :
    """
    df = df.sort_values([constants.DATETIME])

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
    freq = trajectory.groupby([UID, LATITUDE, LONGITUDE]).size().reset_index(name=FREQUENCY)
    return freq.sort_values(by=[UID, FREQUENCY])


def probability_vector(trajectory):
    freq = trajectory.groupby([UID, LATITUDE, LONGITUDE]).size().reset_index(name=FREQUENCY)
    prob = pd.merge(freq, trajectory.groupby(UID).size().reset_index(name=TOTAL_FREQ), left_on=UID, right_on=UID)
    prob[PROBABILITY] = prob[FREQUENCY] / prob[TOTAL_FREQ]

    return prob[UID, LATITUDE, LONGITUDE, PROBABILITY].sort_values(by=[UID, PROBABILITY])


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

    coords = points.total_bounds

    base = shapely.geometry.box(coords[0], coords[1], coords[2], coords[3], ccw=True)
    base = gpd.GeoDataFrame(geometry=[base], crs=constants.DEFAULT_CRS)

    if crs is None:
        return base

    return base.to_crs(crs)


def bbox_from_area(area, bbox_side_len=500, crs=None):

    centroid = area.iloc[0].geometry.centroid

    # get North-East corner
    ne = [float(coord)+(bbox_side_len/2) for coord in centroid]

    # get South-West corner
    sw = [float(coord)-(bbox_side_len/2) for coord in centroid]

    # build bbox from NE,SW corners
    bbox = shapely.geometry.box(sw[0], sw[1], ne[0], ne[1], ccw=True)

    base = gpd.GeoDataFrame(geometry=[bbox], crs=constants.DEFAULT_CRS)

    if crs is None:
        return base

    return base.to_crs(crs)


def bbox_from_name(area_name, crs=None):

    # Get the shape by using osmnx, it returns the shape in DEFAULT_CRS
    boundary = osmnx.gdf_from_place(area_name)

    if isinstance(boundary.loc[0]['geometry'], shapely.geometry.Point):

        boundary = osmnx.gdf_from_place(area_name, which_result=2)

    if crs is None:
        return boundary

    return boundary.to_crs(crs)


def nearest(row, geom_union, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):

    """Find the nearest point and return the corresponding value from specified column."""
    # Find the geometry that is closest
    nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]

    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df2[nearest][src_column].get_values()[0]
    return value

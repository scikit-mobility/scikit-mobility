from ..utils import gislib, utils, constants
from ..core.trajectorydataframe import *
import numpy as np
import inspect

def stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2,  leaving_time=True):
    """
    Detect a stop when the user spends at least `minutes_for_a_stop` minutes
    within a distance (`stop_radius_factor` * `spatial_radius`) km
    from a given trajectory point.
    The stop's coordinates are the median latitude and longitude values of the points found
    within the specified distance.

    :param tdf: TrajDataFrame
        the input trajectory

    :param stop_radius_factor: float (default )
        radius of the ball enclosing all trajectory points within the stop location

    :param minutes_for_a_stop: float (default 20.0)
        minimum stop duration (in minutes)

    :param spatial_radius: float (default None)
        if `None` use the spatial_radius specified in the TrajDataFrame properties
        (assigned by a `preprocessing.compression` function)

    :param leaving_time: bool (default True)
        if `True` a new column 'leaving_datetime' is added with the departure time from the stop location

    :return: TrajDataFrame
        a TrajDataFrame with the coordinates (latitude, longitude) of the stop locations


    References:
        .. [hariharan2004project] Hariharan, Ramaswamy, and Kentaro Toyama. "Project Lachesis: parsing and modeling location histories." In International Conference on Geographic Information Science, pp. 106-124. Springer, Berlin, Heidelberg, 2004.
        .. [zheng2015trajectory] Zheng, Yu. "Trajectory data mining: an overview." ACM Transactions on Intelligent Systems and Technology (TIST) 6, no. 3 (2015): 29.
    """
    # Sort
    tdf = tdf.sort_by_uid_and_datetime()

    # Save function arguments and values in a dictionary
    frame = inspect.currentframe()
    args, _, _, arg_values = inspect.getargvalues(frame)
    arguments = dict([('function', stops.__name__)]+[(i, arg_values[i]) for i in args[1:]])

    groupby = []

    if utils.is_multi_user(tdf):
        groupby.append(constants.UID)
    if utils.is_multi_trajectory(tdf):
        groupby.append(constants.TID)

    # Use the spatial_radius in the tdf parameters, if present, otherwise use the default argument.
    try:
        spatial_radius_km = tdf.parameters[constants.COMPRESSION_PARAMS]['spatial_radius_km']
    except KeyError:
        pass
    stop_radius = spatial_radius_km * stop_radius_factor

    if len(groupby) > 0:
        # Apply simplify trajectory to each group of points
        stdf = tdf.groupby(groupby, group_keys=False, as_index=False).apply(_stops_trajectory, stop_radius=stop_radius,
                                                             minutes_for_a_stop=minutes_for_a_stop,
                                                             leaving_time=leaving_time).reset_index(drop=True)
    else:
        stdf = _stops_trajectory(tdf, stop_radius=stop_radius, minutes_for_a_stop=minutes_for_a_stop,
                                                            leaving_time=leaving_time).reset_index(drop=True)

    stdf.parameters = tdf.parameters
    stdf.set_parameter(constants.DETECTION_PARAMS, arguments)
    return stdf


def _stops_trajectory(tdf, stop_radius, minutes_for_a_stop, leaving_time):

    # From dataframe convert to numpy matrix
    lat_lng_dtime_other = list(utils.to_matrix(tdf))
    columns_order = list(tdf.columns)

    stops, leaving_times = _stops_array(lat_lng_dtime_other, stop_radius, minutes_for_a_stop, leaving_time)

    #print(utils.get_columns(data))
    # stops = utils.to_dataframe(stops, utils.get_columns(data))
    stops = nparray_to_trajdataframe(stops, utils.get_columns(tdf), {})

    # Put back to the original order
    stops = stops[columns_order]

    if leaving_time:
        stops.loc[:, 'leaving_datetime'] = leaving_times

    return stops


def _stops_array(lat_lng_dtime_other, stop_radius, minutes_for_a_stop, leaving_time):
    """
    Create a stop if the user spend at least `minutes_for_a_stop` minutes
    within a distance `stop_radius` from a given point.
    """
    # Define the distance function to use
    measure_distance = gislib.getDistance

    stops = []
    leaving_times = []

    lat_0, lon_0, t_0 = lat_lng_dtime_other[0][:3]
    sum_lat, sum_lon = [lat_0], [lon_0]

    count = 1
    lendata = len(lat_lng_dtime_other) - 1

    for i in range(lendata):

        lat, lon, t = lat_lng_dtime_other[i+1][:3]
        Dt = utils.diff_seconds(t_0,t) / 60.
        Dr = measure_distance([lat_0,lon_0],[lat, lon])

        if Dr > stop_radius:
            if Dt > minutes_for_a_stop:
                extra_cols = list(lat_lng_dtime_other[i][3:])
                estimated_final_t = t
                if leaving_time:
                    leaving_times.append(estimated_final_t)
                stops += [[np.median(sum_lat), np.median(sum_lon), t_0] + extra_cols]

                count = 0
                lat_0, lon_0, t_0 = lat, lon, t
                sum_lat, sum_lon = [], []
            else:
                # Not a stop
                count = 0
                lat_0, lon_0, t_0 = lat, lon, t
                sum_lat, sum_lon = [], []

        count += 1
        sum_lat += [lat]
        sum_lon += [lon]

    return stops, leaving_times

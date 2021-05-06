from ..utils import gislib, utils, constants
from ..core.trajectorydataframe import *
import numpy as np
import inspect


def stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2, leaving_time=True, no_data_for_minutes=1e12, min_speed_kmh=None):
    """Stops detection.
    
    Detect the stops for each individual in a TrajDataFrame. A stop is detected when the individual spends at least `minutes_for_a_stop` minutes within a distance `stop_radius_factor * spatial_radius` km from a given trajectory point. The stop's coordinates are the median latitude and longitude values of the points found within the specified distance [RT2004]_ [Z2015]_.
    
    Parameters
    ----------
    tdf : TrajDataFrame
        the input trajectories of the individuals.

    stop_radius_factor : float, optional
        if argument `spatial_radius_km` is `None`, the spatial_radius used is the value specified in the TrajDataFrame properties ("spatial_radius_km" assigned by a `preprocessing.compression` function) multiplied by this argument, `stop_radius_factor`. The default is `0.5`.

    minutes_for_a_stop : float, optional
        the minimum stop duration, in minutes. The default is `20.0`.

    spatial_radius_km : float or None, optional
        the radius of the ball enclosing all trajectory points within the stop location. The default is `0.2`.

    leaving_time : boolean, optional
        if `True`, a new column 'leaving_datetime' is added with the departure time from the stop location. The default is `True`.

    no_data_for_minutes : float, optional
        if the number of minutes between two consecutive points is larger than `no_data_for_minutes`,
        then this is interpreted as missing data and does not count as a stop. The default is `1e12`.

    min_speed_kmh : float or None, optional
        if not `None`, remove the points at the end of a stop if their speed is larger than `min_speed_kmh` km/h. The default is `None`.
    
    Returns
    -------
    TrajDataFrame
        a TrajDataFrame with the coordinates (latitude, longitude) of the stop locations.

    Examples
    --------
    >>> import skmob
    >>> import pandas as pd
    >>> from skmob.preprocessing import detection
    >>> # read the trajectory data (GeoLife)
    >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
    >>> df = pd.read_csv(url, sep=',', compression='gzip')
    >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
    >>> print(tdf.head())
             lat         lng            datetime  uid
    0  39.984094  116.319236 2008-10-23 05:53:05    1
    1  39.984198  116.319322 2008-10-23 05:53:06    1
    2  39.984224  116.319402 2008-10-23 05:53:11    1
    3  39.984211  116.319389 2008-10-23 05:53:16    1
    4  39.984217  116.319422 2008-10-23 05:53:21    1
    >>> stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2, leaving_time=True)
    >>> print(stdf.head())
             lat         lng            datetime  uid    leaving_datetime
    0  39.978030  116.327481 2008-10-23 06:01:37    1 2008-10-23 10:32:53
    1  40.013820  116.306532 2008-10-23 11:10:19    1 2008-10-23 23:45:27
    2  39.978419  116.326870 2008-10-24 00:21:52    1 2008-10-24 01:47:30
    3  39.981166  116.308475 2008-10-24 02:02:31    1 2008-10-24 02:30:29
    4  39.981431  116.309902 2008-10-24 02:30:29    1 2008-10-24 03:16:35
    >>> print(stdf.parameters)
    {'detect': {'function': 'stops', 'stop_radius_factor': 0.5, 'minutes_for_a_stop': 20.0, 'spatial_radius_km': 0.2, 'leaving_time': True, 'no_data_for_minutes': 1000000000000.0, 'min_speed_kmh': None}}
    >>> print('Points of the original trajectory:\\t%s'%len(tdf))
    >>> print('Points of stops:\\t\\t\\t%s'%len(stdf))
    Points of the original trajectory:	217653
    Points of stops:			391
    
    References
    ----------
    .. [RT2004] Ramaswamy, H. & Toyama, K. (2004) Project Lachesis: parsing and modeling location histories. In International Conference on Geographic Information Science, 106-124, http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf
    .. [Z2015] Zheng, Y. (2015) Trajectory data mining: an overview. ACM Transactions on Intelligent Systems and Technology 6(3), https://dl.acm.org/citation.cfm?id=2743025
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
        stop_radius = tdf.parameters[constants.COMPRESSION_PARAMS]['spatial_radius_km'] * stop_radius_factor
    except (KeyError, TypeError):
        pass
    if spatial_radius_km is not None:
        stop_radius = spatial_radius_km

    if len(groupby) > 0:
        # Apply simplify trajectory to each group of points
        stdf = tdf.groupby(groupby, group_keys=False, as_index=False).apply(_stops_trajectory, stop_radius=stop_radius,
                           minutes_for_a_stop=minutes_for_a_stop, leaving_time=leaving_time,
                           no_data_for_minutes=no_data_for_minutes, min_speed_kmh=min_speed_kmh).reset_index(drop=True)
    else:
        stdf = _stops_trajectory(tdf, stop_radius=stop_radius, minutes_for_a_stop=minutes_for_a_stop,
                            leaving_time=leaving_time, no_data_for_minutes=no_data_for_minutes,
                            min_speed_kmh=min_speed_kmh).reset_index(drop=True)

    # TODO: remove the following line when issue #71 (Preserve the TrajDataFrame index during preprocessing operations) is solved.
    stdf.reset_index(inplace=True, drop=True)

    stdf.parameters = tdf.parameters
    stdf.set_parameter(constants.DETECTION_PARAMS, arguments)
    return stdf


def _stops_trajectory(tdf, stop_radius, minutes_for_a_stop, leaving_time, no_data_for_minutes, min_speed_kmh):

    # From dataframe convert to numpy matrix
    lat_lng_dtime_other = list(utils.to_matrix(tdf))
    columns_order = list(tdf.columns)

    stops, leaving_times = _stops_array(lat_lng_dtime_other, stop_radius,
                                        minutes_for_a_stop, leaving_time, no_data_for_minutes, min_speed_kmh)

    #print(utils.get_columns(data))
    # stops = utils.to_dataframe(stops, utils.get_columns(data))
    stops = nparray_to_trajdataframe(stops, utils.get_columns(tdf), {})

    # Put back to the original order
    stops = stops[columns_order]

    if leaving_time:
        stops.loc[:, constants.LEAVING_DATETIME] = pd.to_datetime(leaving_times)

    return stops


def _stops_array(lat_lng_dtime_other, stop_radius, minutes_for_a_stop, leaving_time, no_data_for_minutes, min_speed_kmh):
    """
    Create a stop if the user spend at least `minutes_for_a_stop` minutes
    within a distance `stop_radius` from a given point.
    """
    # Define the distance function to use
    measure_distance = gislib.getDistance

    stops = []
    leaving_times = []

    lat_0, lon_0, t_0 = lat_lng_dtime_other[0][:3]
    sum_lat, sum_lon, sum_t = [lat_0], [lon_0], [t_0]
    speeds_kmh = []

    count = 1
    lendata = len(lat_lng_dtime_other) - 1

    for i in range(lendata):

        lat, lon, t = lat_lng_dtime_other[i+1][:3]

        if utils.diff_seconds(lat_lng_dtime_other[i][2], t) / 60. > no_data_for_minutes:
            # No data for more than `no_data_for_minutes` minutes: Not a stop
            count = 0
            lat_0, lon_0, t_0 = lat, lon, t
            sum_lat, sum_lon, sum_t = [], [], []
            speeds_kmh = []

        Dt = utils.diff_seconds(t_0, t) / 60.
        Dr = measure_distance([lat_0, lon_0], [lat, lon])
        try:
            speeds_kmh += [Dr / Dt * 60.]
        except ZeroDivisionError:
            speeds_kmh += [0.]

        if Dr > stop_radius:
            if Dt > minutes_for_a_stop:
                extra_cols = list(lat_lng_dtime_other[i][3:])

                # estimate the leaving time
                if min_speed_kmh is None:
                    estimated_final_t = t
                else:
                    j = 1
                    for j in range(1, len(speeds_kmh)):
                        if speeds_kmh[-j] < min_speed_kmh:
                            break
                    if j == 1:
                        estimated_final_t = t
                    else:
                        estimated_final_t = sum_t[-j + 1]
                        sum_lat = sum_lat[:-j]
                        sum_lon = sum_lon[:-j]

                if len(sum_lat) > 0 and utils.diff_seconds(t_0, estimated_final_t) / 60. > minutes_for_a_stop:
                    if leaving_time:
                        leaving_times.append(estimated_final_t)

                    stops += [[np.median(sum_lat), np.median(sum_lon), t_0] + extra_cols]

                count = 0
                lat_0, lon_0, t_0 = lat, lon, t
                sum_lat, sum_lon, sum_t = [], [], []
                speeds_kmh = []
            else:
                # Not a stop
                count = 0
                lat_0, lon_0, t_0 = lat, lon, t
                sum_lat, sum_lon, sum_t = [], [], []
                speeds_kmh = []

        count += 1
        sum_lat += [lat]
        sum_lon += [lon]
        sum_t += [t]

    return stops, leaving_times

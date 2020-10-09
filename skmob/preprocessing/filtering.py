from ..utils import gislib, utils, constants
from ..core.trajectorydataframe import *
import numpy as np
import inspect

def filter(tdf, max_speed_kmh=500., include_loops=False, speed_kmh=5., max_loop=6, ratio_max=0.25):
    """Trajectory filtering.
    
    For each individual in a TrajDataFrame, filter out the trajectory points that are considered noise or outliers [Z2015]_.
    
    Parameters
    ----------
    tdf : TrajDataFrame
        the trajectories of the individuals.

    max_speed_kmh : float, optional
        delete a trajectory point if the speed (in km/h) from the previous point is higher than `max_speed_kmh`. The default is `500.0`.

    include_loops: boolean, optional
        If `True`, trajectory points belonging to short and fast "loops" are removed. Specifically, points are removed if within the next `max_loop` points the individual has come back to a distance (`ratio_max` * the maximum distance reached), AND the average speed (in km/h) is higher than `speed`. The default is `False`.
    
    speed : float, optional 
        the default is 5km/h (walking speed).

    max_loop : int, optional
        the default is `6`.

    ratio_max : float, optional
        the default is `0.25`.
    
    Returns
    -------
    TrajDataFrame
        the TrajDataFrame without the trajectory points that have been filtered out.
    
    Warnings
    --------
    if `include_loops` is `True`, the filter is very slow. Use only if raw data is really noisy.
    
    Examples
    --------
    >>> import skmob
    >>> import pandas as pd
    >>> from skmob.preprocessing import filtering
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
    >>> # filter out all points with a speed (in km/h) from the previous point higher than 500 km/h
    >>> ftdf = filtering.filter(tdf, max_speed_kmh=500.)
    >>> print(ftdf.parameters)
    {'filter': {'function': 'filter', 'max_speed_kmh': 500.0, 'include_loops': False, 'speed_kmh': 5.0, 'max_loop': 6, 'ratio_max': 0.25}}
    >>> n_deleted_points = len(tdf) - len(ftdf) # number of deleted points
    >>> print(n_deleted_points)
    54
    
    References
    ----------
    .. [Z2015] Zheng, Y. (2015) Trajectory data mining: an overview. ACM Transactions on Intelligent Systems and Technology 6(3), https://dl.acm.org/citation.cfm?id=2743025
    """
    # Sort
    tdf = tdf.sort_by_uid_and_datetime()

    # Save function arguments and values in a dictionary
    frame = inspect.currentframe()
    args, _, _, arg_values = inspect.getargvalues(frame)
    arguments = dict([('function', filter.__name__)]+[(i, arg_values[i]) for i in args[1:]])

    groupby = []

    if utils.is_multi_user(tdf):
        groupby.append(constants.UID)
    if utils.is_multi_trajectory(tdf):
        groupby.append(constants.TID)

    if len(groupby) > 0:
        # Apply simplify trajectory to each group of points
        ftdf = tdf.groupby(groupby, group_keys=False).apply(_filter_trajectory, max_speed=max_speed_kmh,
                                                            include_loops=include_loops, speed=speed_kmh,
                                                             max_loop=max_loop, ratio_max=ratio_max)
    else:
        ftdf = _filter_trajectory(tdf, speed=speed_kmh, max_speed=max_speed_kmh,
                        max_loop=max_loop, ratio_max=ratio_max,
                        include_loops=include_loops)

    # TODO: remove the following line when issue #71 (Preserve the TrajDataFrame index during preprocessing operations) is solved.
    ftdf.reset_index(inplace=True, drop=True)

    ftdf.parameters = tdf.parameters
    ftdf.set_parameter(constants.FILTERING_PARAMS, arguments)
    return ftdf


def _filter_trajectory(tdf, max_speed, include_loops, speed, max_loop, ratio_max):
    # From dataframe convert to numpy matrix
    lat_lng_dtime_other = list(utils.to_matrix(tdf))
    columns_order = list(tdf.columns)

    trajectory = _filter_array(lat_lng_dtime_other, max_speed, include_loops, speed, max_loop, ratio_max)

    filtered_traj = nparray_to_trajdataframe(trajectory, utils.get_columns(tdf), {})
    # Put back to the original order
    filtered_traj = filtered_traj[columns_order]

    return filtered_traj


def _filter_array(lat_lng_dtime_other, max_speed, include_loops, speed, max_loop, ratio_max):
    """
    TODO: add a filter based on the acceleration

     Delete points from raw trajectory `data` if:

    1. The speed from previous point is > `max_speed` km/h

    2. Within the next `max_loop` points the user has come back
            of `ratio_max`% of the maximum distance reached, AND s/he travelled
            at a speed > `speed` km/h
     """
    distfunc = gislib.getDistance
    lX = len(lat_lng_dtime_other)
    i = 0

    while i < lX - 2:

        try:
            dt = utils.diff_seconds(lat_lng_dtime_other[i][2], lat_lng_dtime_other[i + 1][2])

        except IndexError:
            pass
        try:
            if distfunc(lat_lng_dtime_other[i][:2], lat_lng_dtime_other[i + 1][:2]) / dt * 3600. > max_speed:
                del lat_lng_dtime_other[i + 1]
                lX = len(lat_lng_dtime_other)
                continue

        except ZeroDivisionError:
            del lat_lng_dtime_other[i + 1]
            lX = len(lat_lng_dtime_other)
            continue

        if include_loops:
            ahead = min(max_loop, lX - i - 1)
            DrDt = np.array([[distfunc(lat_lng_dtime_other[i][:2], lat_lng_dtime_other[i + j][:2]),
                              utils.diff_seconds(lat_lng_dtime_other[i][2], lat_lng_dtime_other[i + j][2])] for j in range(1, ahead)])

            imax = np.argmax(DrDt, axis=0)[0]
            inside = imax + np.where(DrDt[imax:, 0] < DrDt[imax, 0] * ratio_max)[0]
            try:
                imin = inside[0]
            except IndexError:
                i += 1
                continue
            Dr, Dt = sum(DrDt[:imin, :])

            if Dr / Dt * 3600. > speed:
                del lat_lng_dtime_other[i + 1 + imax]
                lX = len(lat_lng_dtime_other)
            else:
                i += 1
        else:
            i += 1

    return lat_lng_dtime_other
from ..utils import gislib, utils, constants
import numpy as np
import inspect

def filter(tdf, max_speed=500., include_loops=False, speed=5., max_loop=6, ratio_max=0.25):
    """
    Filter out trajectory points that are considered noise or outliers.

    :param tdf: TrajDataFrame
        the raw trajectory

    :param max_speed: float (default 500.0)
        delete trajectory point if the speed from the previous point is higher than `max_speed`

    :param include_loops: bool (default False)
        optional: this filter is very slow. Use only if raw data is really noisy.
        If `True`, trajectory points belonging to short and fast "loops" are removed.
        Remove points if within the next `max_loop` points the user has come back to a distance
        (`ratio_max` * the maximum distance reached), AND the average speed is higher than `speed` km/h.

    :param speed: float (default 5 km/h, walking speed)

    :param max_loop: int (default 6)

    :param ratio_max: float (default 0.25)

    :return: TrajDataFrame
        the TrajDataFrame without the trajectory points that have been filtered out.

    References:
        .. [zheng2015trajectory] Zheng, Yu. "Trajectory data mining: an overview." ACM Transactions on Intelligent Systems and Technology (TIST) 6, no. 3 (2015): 29.
    """
    # Sort
    tdf.sort_by_uid_and_datetime()

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
        ftdf = tdf.groupby(groupby, group_keys=False).apply(_filter_trajectory, max_speed=max_speed,
                                                            include_loops=include_loops, speed=speed,
                                                             max_loop=max_loop, ratio_max=ratio_max)

    else:
        ftdf = _filter_trajectory(tdf, speed=speed, max_speed=max_speed,
                        max_loop=max_loop, ratio_max=ratio_max,
                        include_loops=include_loops)

    ftdf.parameters = tdf.parameters
    ftdf.set_parameter(constants.FILTERING_PARAMS, arguments)
    return ftdf


def _filter_trajectory(tdf, max_speed, include_loops, speed, max_loop, ratio_max):
    # From dataframe convert to numpy matrix
    lat_lng_dtime_other = list(utils.to_matrix(tdf))
    columns_order = list(tdf.columns)

    trajectory = _filter_array(lat_lng_dtime_other, max_speed, include_loops, speed, max_loop, ratio_max)

    filtered_traj = utils.nparray_to_trajdataframe(trajectory, utils.get_columns(tdf), {})
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
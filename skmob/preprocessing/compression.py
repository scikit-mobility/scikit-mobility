from ..utils import gislib, utils, constants
from ..core.trajectorydataframe import *
import numpy as np
import inspect

def compress(tdf, spatial_radius_km=0.2):
    """Trajectory compression.
    
    Reduce the number of points in a trajectory for each individual in a TrajDataFrame. All points within a radius of `spatial_radius_km` kilometers from a given initial point are compressed into a single point that has the median coordinates of all points and the time of the initial point [Z2015]_.
    
    Parameters
    ----------
    tdf : TrajDataFrame
        the input trajectories of the individuals.

    spatial_radius_km : float, optional
        the minimum distance (in km) between consecutive points of the compressed trajectory. The default is `0.2`.
    
    Returns
    -------
    TrajDataFrame
        the compressed TrajDataFrame.
    
    Examples
    --------
    >>> import skmob
    >>> import pandas as pd
    >>> from skmob.preprocessing import compression
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
    >>> # compress the trajectory using a spatial radius of 0.2 km
    >>> ctdf = compression.compress(tdf, spatial_radius_km=0.2)
    >>> print('Points of the original trajectory:\\t%s'%len(tdf))
    >>> print('Points of the compressed trajectory:\\t%s'%len(ctdf))
    Points of the original trajectory:	217653
    Points of the compressed trajectory:	6281
    
    References
    ----------
    .. [Z2015] Zheng, Y. (2015) Trajectory data mining: an overview. ACM Transactions on Intelligent Systems and Technology 6(3), https://dl.acm.org/citation.cfm?id=2743025
    """
    # Sort
    tdf = tdf.sort_by_uid_and_datetime()

    # Save function arguments and values in a dictionary
    frame = inspect.currentframe()
    args, _, _, arg_values = inspect.getargvalues(frame)
    arguments = dict([('function', compress.__name__)]+[(i, arg_values[i]) for i in args[1:]])

    groupby = []

    if utils.is_multi_user(tdf):
        groupby.append(constants.UID)
    if utils.is_multi_trajectory(tdf):
        groupby.append(constants.TID)

    if len(groupby) > 0:
        # Apply simplify trajectory to each group of points
        ctdf = tdf.groupby(groupby, group_keys=False).apply(_compress_trajectory, spatial_radius=spatial_radius_km)
    else:
        ctdf = _compress_trajectory(tdf, spatial_radius=spatial_radius_km)

    # TODO: remove the following line when issue #71 (Preserve the TrajDataFrame index during preprocessing operations) is solved.
    ctdf.reset_index(inplace=True, drop=True)

    ctdf.parameters = tdf.parameters
    ctdf.set_parameter(constants.COMPRESSION_PARAMS, arguments)
    return ctdf


def _compress_trajectory(tdf, spatial_radius):
    # From dataframe convert to numpy matrix
    lat_lng_dtime_other = utils.to_matrix(tdf)
    columns_order = list(tdf.columns)

    compressed_traj = _compress_array(lat_lng_dtime_other, spatial_radius)

    compressed_traj = nparray_to_trajdataframe(compressed_traj, utils.get_columns(tdf), {})
    # Put back to the original order
    compressed_traj = compressed_traj[columns_order]

    return compressed_traj


def _compress_array(lat_lng_dtime_other, spatial_radius):
    if len(lat_lng_dtime_other) < 2:
        return lat_lng_dtime_other

    # Define the distance function to use
    measure_distance = gislib.getDistance

    compressed_traj = []
    lat_0, lon_0 = lat_lng_dtime_other[0][:2]

    sum_lat, sum_lon = [lat_0], [lon_0]
    t_0 = lat_lng_dtime_other[0][2]
    i_0 = 0
    count = 1
    lendata = len(lat_lng_dtime_other) - 1

    for i in range(lendata):
        lat,lon,t = lat_lng_dtime_other[i+1][:3]

        Dr = measure_distance([lat_0,lon_0],[lat, lon])

        if Dr > spatial_radius:

            extra_cols = list(lat_lng_dtime_other[i_0][3:])
            compressed_traj += [[np.median(sum_lat), np.median(sum_lon), t_0] + extra_cols]

            t_0 = t
            count = 0
            lat_0, lon_0 = lat, lon
            i_0 = i + 1
            sum_lat, sum_lon = [], []

        count += 1
        sum_lat += [lat]
        sum_lon += [lon]

        if i == lendata - 1:
            extra_cols = list(lat_lng_dtime_other[i_0][3:])
            compressed_traj += [[np.median(sum_lat), np.median(sum_lon), t_0] + extra_cols]

    return compressed_traj

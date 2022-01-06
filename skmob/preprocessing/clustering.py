from ..utils import utils, constants
from ..core.trajectorydataframe import *
from sklearn.cluster import DBSCAN
import numpy as np
import inspect

kms_per_radian = 6371.0088   # Caution: this is only true at the Equator!
                             # This may cause problems at high latitudes.


def cluster(tdf, cluster_radius_km=0.1, min_samples=1):
    """Clustering of locations.
    
    Cluster the stops of each individual in a TrajDataFrame. The stops correspond to visits to the same location at different times, based on spatial proximity [RT2004]_. The clustering algorithm used is DBSCAN (by sklearn [DBSCAN]_).
    
    Parameters
    ----------
    tdf : TrajDataFrame
        the input TrajDataFrame that should contain the stops, i.e., the output of a `preprocessing.detection` function.

    cluster_radius_km : float, optional
        the parameter `eps` of the function sklearn.cluster.DBSCAN, in kilometers. The default is `0.1`.
        
    min_samples : int, optional
        the parameter `min_samples` of the function sklearn.cluster.DBSCAN indicating the minimum number of stops to form a cluster. The default is `1`.
    
    Returns
    -------
    TrajDataFrame
        a TrajDataFrame with the additional column 'cluster' containing the cluster labels. The stops that belong to the same cluster have the same label. The labels are integers corresponding to the ranks of clusters according to the frequency of visitation (the most visited cluster has label 0, the second most visited has label 1, etc.).
    
    Examples
    --------
    >>> import skmob
    >>> import pandas as pd
    >>> from skmob.preprocessing import detection, clustering
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
    >>> # detect the stops first
    >>> stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2, leaving_time=True)
    >>> # cluster the stops
    >>> cstdf = clustering.cluster(stdf, cluster_radius_km=0.1, min_samples=1)
    >>> print(cstdf.head())
             lat         lng            datetime  uid    leaving_datetime  cluster
    0  39.978030  116.327481 2008-10-23 06:01:37    1 2008-10-23 10:32:53        0
    1  40.013820  116.306532 2008-10-23 11:10:19    1 2008-10-23 23:45:27        1
    2  39.978419  116.326870 2008-10-24 00:21:52    1 2008-10-24 01:47:30        0
    3  39.981166  116.308475 2008-10-24 02:02:31    1 2008-10-24 02:30:29       42
    4  39.981431  116.309902 2008-10-24 02:30:29    1 2008-10-24 03:16:35       41    
    >>> print(cstdf.parameters)
    {'detect': {'function': 'stops', 'stop_radius_factor': 0.5, 'minutes_for_a_stop': 20.0, 'spatial_radius_km': 0.2, 'leaving_time': True, 'no_data_for_minutes': 1000000000000.0, 'min_speed_kmh': None}, 'cluster': {'function': 'cluster', 'cluster_radius_km': 0.1, 'min_samples': 1}}

    References
    ----------
    .. [DBSCAN] DBSCAN implementation, scikit-learn, https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    .. [RT2004] Ramaswamy, H. & Toyama, K. (2004) Project Lachesis: parsing and modeling location histories. In International Conference on Geographic Information Science, 106-124, http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf
    """
    # Sort
    tdf = tdf.sort_by_uid_and_datetime()

    # Save function arguments and values in a dictionary
    frame = inspect.currentframe()
    args, _, _, arg_values = inspect.getargvalues(frame)
    arguments = dict([('function', cluster.__name__)]+[(i, arg_values[i]) for i in args[1:]])

    groupby = []

    if utils.is_multi_user(tdf):
        groupby.append(constants.UID)
    # if utils.is_multi_trajectory(data):
    #     groupby.append(constants.TID)

    stops_df = tdf
    # stops_df = detection.stops(data, stop_radius_factor=0.5, \
    #         minutes_for_a_stop=20.0, spatial_radius=0.2, leaving_time=True)

    if len(groupby) > 0:
        # Apply cluster stops to each group of points
        ctdf = stops_df.groupby(groupby, group_keys=False, as_index=False).apply(_cluster_trajectory,
                                cluster_radius_km=cluster_radius_km, min_samples=min_samples).reset_index(drop=True)
    else:
        ctdf = _cluster_trajectory(stops_df, cluster_radius_km=cluster_radius_km, min_samples=min_samples).reset_index(drop=True)

    # TODO: remove the following line when issue #71 (Preserve the TrajDataFrame index during preprocessing operations) is solved.
    ctdf.reset_index(inplace=True, drop=True)

    ctdf.parameters = tdf.parameters
    ctdf.set_parameter(constants.CLUSTERING_PARAMS, arguments)
    return ctdf


def _cluster_trajectory(tdf, cluster_radius_km, min_samples):
    # From dataframe convert to numpy matrix
    lat_lng_dtime_other = list(utils.to_matrix(tdf))
    columns_order = list(tdf.columns)

    l2x, cluster_IDs = _cluster_array(lat_lng_dtime_other, cluster_radius_km, min_samples)

    clusters_df = nparray_to_trajdataframe(lat_lng_dtime_other, utils.get_columns(tdf), {})
    # Put back to the original order
    clusters_df = clusters_df[columns_order]

    clusters_df.loc[:, 'cluster'] = cluster_IDs

    return clusters_df


def group_by_label(X, labels):
    """
     return a dictionary 'l2x' in which the elements 'x' of list 'X'
     are grouped according to 'labels'
    """
    l2x = dict([(l ,[]) for l in set(labels)])
    for x ,l in list(zip(X ,labels)):
        l2x[l] += [x]
    return l2x


def _cluster_array(lat_lng_dtime_other, cluster_radius_km, min_samples, verbose=False):

    X = np.array([[point[0], point[1]] for point in lat_lng_dtime_other])

    # Compute DBSCAN
    eps_rad = cluster_radius_km / kms_per_radian

    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    clus = db.fit(np.radians(X))
    # core_samples = clus.core_sample_indices_
    labels = clus.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    if verbose:
        print('Estimated number of clusters: %d' % n_clusters_)
    l02x = group_by_label(X, labels)

    # Map cluster index to most frequent location: label2fml
    c2mfl = dict([(c[1] ,i) for i ,c in \
                  enumerate(sorted([[len(v) ,l] for l ,v in l02x.items() if l> -0.5], reverse=True))])
    l2x = dict([(c2mfl[k], v) for k, v in l02x.items() if k > -0.5])
    try:
        l2x[-1] = l02x[-1.]
    except KeyError:
        pass

    return l2x, [c2mfl[k] for k in labels if k > -0.5]


def split_trajectories_in_tdf(tdf, stop_tdf):
    """Cluster the points of a TrajDataFrame into trajectories by using stop locations.
    Parameters
    ----------
    tdf : TrajDataFrame
        original trajectories
    stop_tdf : TrajDataFrame
        the output of skmob.preprocessing.detection.stops, containing the stop locations of the users in the tdf
    Returns
    -------
    TrajDataFrame
        the TrajDataFrame with a new column 'tid' collecting the unique identifier of the trajectory to which the point
        belongs.

    Examples
    --------
    >>> import skmob
    >>> import pandas as pd
    >>> from skmob.preprocessing import detection, clustering
    >>> # read the trajectory data (GeoLife)
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
    >>> # detect the stops first
    >>> stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2, leaving_time=True)
    >>> # cluster the trajectories based on the stops
    >>> tdf_splitted = split_trajectories_in_tdf(tdf, stdf)
    >>> print(tdf_splitted.head())
              lat         lng            datetime  uid  tid
    0   39.984094  116.319236 2008-10-23 05:53:05    1    1
    1   39.984198  116.319322 2008-10-23 05:53:06    1    1
    2   39.984224  116.319402 2008-10-23 05:53:11    1    1
    3   39.984211  116.319389 2008-10-23 05:53:16    1    1
    4   39.984217  116.319422 2008-10-23 05:53:21    1    1
    """

    tdf_with_tid = tdf.groupby('uid').apply(_split_trajectories, stop_tdf)
    return tdf_with_tid.reset_index(drop=True)


def _split_trajectories(tdf, stop_tdf):

    c_uid = tdf.uid[:1].item()
    stop_tdf_current_user = stop_tdf[stop_tdf.uid == c_uid]

    if stop_tdf_current_user.empty:
        return
    else:
        first_traj = [tdf[tdf.datetime <= stop_tdf_current_user.datetime[:1].item()]]
        last_traj = [tdf[tdf.datetime >= stop_tdf_current_user.leaving_datetime[-1:].item()]]
        all_other_traj = [tdf[(tdf.datetime >= start_traj_time) & (tdf.datetime <= end_traj_time)] for end_traj_time, start_traj_time in zip(stop_tdf_current_user['datetime'][1:], stop_tdf_current_user['leaving_datetime'][:-1])]
        all_traj = first_traj + all_other_traj + last_traj
        tdf_with_tid = pd.concat(all_traj)
        list_tids = [list(np.repeat(i, len(df))) for i, df in zip(range(1,len(all_traj)+1), all_traj)]
        list_tids_ravel = [item for sublist in list_tids for item in sublist]
        tdf_with_tid['tid'] = list_tids_ravel
        return tdf_with_tid


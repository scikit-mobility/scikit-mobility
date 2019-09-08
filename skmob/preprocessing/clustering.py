from ..utils import utils, constants
from ..core.trajectorydataframe import *
from sklearn.cluster import DBSCAN
import numpy as np
import inspect

kms_per_radian = 6371.0088   # Caution: this is only true at the Equator!
                             # This may cause problems at high latitudes.


def cluster(tdf, cluster_radius_km=0.1, min_samples=1):
    """
    Cluster stops corresponding to visits to the same location at different times, based on spatial proximity.
    The clustering algorithm used is DBSCAN (by sklearn).

    :param tdf: TrajDataFrame
        the input TrajDataFrame that should contain the stops, i.e. the output of a `preprocessing.detection` function

    :param cluster_radius_km: float (default 0.1)
        the parameter `eps` of the function sklearn.cluster.DBSCAN in kilometers

    :param min_samples: int (default 1)
        the parameter `min_samples` of the function sklearn.cluster.DBSCAN (minimum size of a cluster)

    :return: TrajDataFrame
        a TrajDataFrame with the additional column 'cluster' containing the cluster labels.
        Stops belonging to the same cluster have the same label.
        Labels are integers corresponding to the ranks of clusters according to the frequency of visitation
        (the most visited cluster has label 0, the second most visited has label 1, ...)

    References:
        .. [hariharan2004project] Hariharan, Ramaswamy, and Kentaro Toyama. "Project Lachesis: parsing and modeling location histories." In International Conference on Geographic Information Science, pp. 106-124. Springer, Berlin, Heidelberg, 2004.
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

    return l2x, [c2mfl[k] for k in labels]

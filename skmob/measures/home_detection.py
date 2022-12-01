from math import nan
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import pdist, squareform
from skmob.utils.gislib import getDistanceByHaversine
from tqdm import tqdm
from ..utils import constants

tqdm.pandas()

# wrapper function to use the haversine formula as a metric for numpy's pdist
def _haversine_metric(point_1, point_2):
    loc1 = (point_1[0], point_1[1])
    loc2 = (point_2[0], point_2[1])
    return getDistanceByHaversine(loc1, loc2)

# create a DataFrame containing the distances of all locations
def _build_distance_matrix(df):
    distances = pdist(df.values, metric=_haversine_metric)
    points = [f'{df[constants.LATITUDE][i]} {df[constants.LONGITUDE][i]}' for i in range(0, len(df))]
    return pd.DataFrame(squareform(distances), columns=points, index=points)


def _between_days(traj, start_day, end_day):
    dayOfWeek = traj[constants.DATETIME].dt.dayofweek
    return traj[(dayOfWeek >= start_day) & (dayOfWeek <= end_day)]


def _uid_apply(function, traj, show_progress, **args):
    # create a copy of the TrajDataFrame
    traj_copy = traj.copy()

    # if 'uid' column in not present in the TrajDataFrame, apply the specified function
    if constants.UID not in traj_copy.columns:
        return pd.DataFrame([function(traj_copy, **args)], columns=[constants.LATITUDE, constants.LONGITUDE])
    
    # show a progress bar, based on a flag, and perform a groupby 'uid' column and apply the specified function
    if show_progress:
        df = traj_copy.groupby(constants.UID).progress_apply(lambda x: function(x, **args))
    else:
        df = traj_copy.groupby(constants.UID).apply(lambda x: function(x, **args))

    return pd.DataFrame(df.to_list(), index=df.index).reset_index().rename(columns={0: constants.LATITUDE, 1: constants.LONGITUDE})


def _radius_activity(traj, radius):
    # build the distance matrix, containing the distances in kilometers
    dmat = _build_distance_matrix(traj)

    traj.columns = [constants.LATITUDE, constants.LONGITUDE, constants.ACTIVITY] 

    # generate the (latitude, longitude) index for the new DataFrame
    index_points = [f'{traj[constants.LATITUDE][i]} {traj[constants.LONGITUDE][i]}' for i in range(0, len(traj))]

    # create and initialize a new 'radius_activity'
    traj = pd.DataFrame(traj[constants.ACTIVITY])
    traj.index = index_points
    traj[constants.RADIUS_ACTIVITY] = traj[constants.ACTIVITY]

    # iterate through the matrix and calculate the 'radius_activity' values
    for i, column in enumerate(dmat):
        for j, distance in enumerate(dmat[column].values):
            # since the matrix is symmetric, iterating on the upper matrix is enough for the computation 
            if i == j:
                break
            
            col_index = dmat.columns[i]
            row_index = dmat.index[j]

            # if the calculated distance is less than the specified radius, update the 'radius_activity' values
            if distance <= radius:
                col_activity = traj.at[col_index, constants.ACTIVITY]
                row_activity = traj.at[row_index, constants.ACTIVITY]

                traj.at[row_index, constants.RADIUS_ACTIVITY] += col_activity
                traj.at[col_index, constants.RADIUS_ACTIVITY] += row_activity

    # return the (latitude, longitude) tuple having the higher 'radius_activity' values
    lat, lng = traj.sort_values(by=constants.RADIUS_ACTIVITY, ascending=False).index[0].split(' ')   
    return (float(lat), float(lng))


def _sjoin_activity(traj, radius):
    # generate the index for the new GeoDataFrame
    index = list(range(0,len(traj.index)))

    # create a new GeoDataFrame, filling the 'geometry' attribute with the (longitude, latitude) TrajDataFrame points
    # transform the geometries into an 'universal' crs, in order to treat the points in a two-dimensional space.
    towers = gpd.GeoDataFrame(traj, index=index, geometry=gpd.points_from_xy(traj[constants.LONGITUDE], traj[constants.LATITUDE]),crs='EPSG:4326').to_crs(constants.UNIVERSAL_CRS)
    towers['index'] = index
    towers = towers.set_index('index')

    # create a circular polygons, with a specified radius in kilometers, around the points arranged in the
    # previous GeoDataFrame
    circles = towers[['geometry']].copy()
    circles['geometry'] = circles.buffer(radius * 1000)    

    # perform a spatial join between all the polygons and the original points
    # for each circle, 'radius_activity' values are calculated through the sum of the individual activities
    points_within = towers.sjoin(circles, predicate='within').groupby('index_right')[[constants.ACTIVITY]].sum()
    points_within.index.rename('index', inplace=True)
    points_within.rename(columns={constants.ACTIVITY: constants.RADIUS_ACTIVITY}, inplace=True)
    towers = towers.join(points_within)
    towers = towers[[constants.LATITUDE, constants.LONGITUDE, constants.RADIUS_ACTIVITY]]    

    # sort and return the (latitude, longitude) tuple having the higher 'radius_activity'
    df = towers.rename(columns={0: constants.LATITUDE, 1: constants.LONGITUDE, 2: constants.RADIUS_ACTIVITY}).sort_values(by=constants.RADIUS_ACTIVITY, ascending=False).reset_index(drop=True)

    lat = df.at[0, constants.LATITUDE]
    lng = df.at[0, constants.LONGITUDE]

    return (lat, lng)


def _home_location_ma(traj, week_period, radius, mode): 
    # filter specified week period from the TrajDataFrame
    if week_period == constants.WEEK_DAYS:
        traj = _between_days(traj, constants.MONDAY, constants.FRIDAY)
    elif week_period == constants.WEEKEND_DAYS:
        traj = _between_days(traj, constants.SATURDAY, constants.SUNDAY)
    elif week_period != None:
        raise ValueError("Invalid week_period argument")

    # if the specified TrajDataFrame is empty, return (NaN, Nan)
    if traj.empty:
        return (nan, nan)

    # create a new column containing the home location activity
    traj[constants.ACTIVITY] = 0

    # calculate the number of occurences for (latitude, longitude) and sort the resulting DataFrame in descending order
    traj = traj[[constants.LATITUDE, constants.LONGITUDE, constants.ACTIVITY]].groupby([constants.LATITUDE, constants.LONGITUDE]).count().sort_values(by=constants.ACTIVITY, ascending=False).reset_index()
 
    # if a radius argument was specified, apply the related radius_activity algorithm
    if isinstance(radius, (int, float)):
        if mode == 'sjoin':
            lat, lng = _sjoin_activity(traj, radius)
        elif mode == 'distance_matrix':
            lat, lng = _radius_activity(traj, radius)
        else:
            raise ValueError("Invalid mode argument")
    # if no radius was specified, select and return the first row (latitude, longitude) from the processed DataFrame       
    elif(radius == None):
        lat = traj.at[0, constants.LATITUDE]
        lng = traj.at[0, constants.LONGITUDE]
    else:
        raise ValueError("Invalid radius argument")
    return (lat, lng)
    

def _home_location_tc(traj, week_period, radius, mode, start_time, end_time):
    # set the datetime as index, apply the time filters and perform the mostAmount algorithm on the new DataFrame
    traj = traj.set_index(pd.DatetimeIndex(traj.datetime)).between_time(start_time, end_time).reset_index(drop=True)
    return _home_location_ma(traj, week_period, radius, mode)


def _calculate_dd_activity(traj):
    # return a tuple, containing the number of distinct days and the total number
    distinct_days = traj[constants.DATETIME].dt.date.unique()
    return (len(distinct_days), len(traj))


def _home_location_dd(traj):
    traj = traj.groupby([constants.LATITUDE, constants.LONGITUDE]).apply(lambda x: _calculate_dd_activity(x)).reset_index().rename(columns={0: constants.ACTIVITY})
    traj = traj.sort_values(by=constants.ACTIVITY, ascending=False).reset_index()

    lat = traj.at[0, constants.LATITUDE]
    lng = traj.at[0, constants.LONGITUDE]

    return (lat, lng)


def _home_location_in(traj, threshold):
    # create and calculate a new 'time_gap' column, containing the difference, in minutes, between sequential rows in the DataFrame
    traj[constants.TIME_GAP] = traj.sort_values(constants.DATETIME, ascending=True)[constants.DATETIME].diff(1).dt.total_seconds().div(60).shift(-1)
    
    # filter the rows that are exceeding the specified threshold on the new DataFrame and apply the mostAmount algorithm
    traj = traj[traj[constants.TIME_GAP] > threshold]
    return _home_location_ma(traj, None, None, None)




def home_location_ma(traj, week_period=None, radius=None, mode='sjoin', show_progress=True):
    """Home location - Most Amount criterion.
    
    Compute the home location of a set of individuals in a TrajDataFrame. 
    The home location :math:` h(u)_{\hbox{\tiny{MA}}}` of an individual :math:`u` is defined as the location :math:`u` visits the most during the specified week period [PFSCB2021]_ [CBTDHVSB2012]_ [PSO2012]_: 
    
    .. math:: 
        h(u)_{\hbox{\tiny{MA}}} = arg\max_{i} |\{r_i | t(r_i) \in WP \}|
    
    where :math:`r_i` is a location visited by :math:`u`, :math:`t(r_i)` is the time when :math:`u` visited :math:`r_i`, and :math:`WP` is the week period taken into consideration.

    If a :math:`radius` is also specified, a spatial perimeter will be implemented around each location, which aggregates all activities within and computes the :math:`radius activity`.  
    In this case, the home location will be calculated based on the highest :math:`radius activity`. 

    The :math:`radius activity` can be be computed using a spatial join method to check which points are within around the specified radius or a .


    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    week_period : str, optional
        the week period to consider. It can be 'WK' (week days), 'WE' (weekend days) or None (every day of the week). The default is None.

    radius : int, optional
        if specified, a spatial perimeter of a certain radius will be implemented around each location, the radius is expressed in kilometers. The default is None.
    
    mode : str, optional
        the method used to compute the :math:`radius activity`. It can be 'sjoin' (spatial join) or 'distance_matrix' . The default is 'sjoin'.

    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    ----------
    pandas DataFrame
        the home location, as a :math:`(latitude, longitude)` pair, of the individuals.

    Examples
    --------
    >>> import skmob
    >>> from skmob.measures.home_detection import home_location_ma
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, 
                 names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user')
    >>> hl = home_location_ma(tdf)
    >>> print(hl.head())
       uid        lat         lng
    0    0  39.762146 -104.982480
    1    1  37.630490 -122.411084
    2    2  39.739154 -104.984703
    3    3  37.748170 -122.459192
    4    4  60.166970   24.944602
    
    References
    ----------
    .. [PFSCB2021] Pappalardo, Luca & Ferres, Leo & Sacasa, Manuel & Cattuto, Ciro & Bravo, Loreto. (2021). Evaluation of home detection algorithms on mobile phone data using individual-level ground truth. EPJ Data Science. 10. 29. 10.1140/epjds/s13688-021-00284-9. 
    .. [CBTDHVSB2012] Csáji, B. C., Browet, A., Traag, V. A., Delvenne, J.-C., Huens, E., Van Dooren, P., Smoreda, Z. & Blondel, V. D. (2012) Exploring the Mobility of Mobile Phone Users. Physica A: Statistical Mechanics and its Applications 392(6), 1459-1473, https://www.sciencedirect.com/science/article/pii/S0378437112010059
    .. [PSO2012] Phithakkitnukoon, S., Smoreda, Z. & Olivier, P. (2012) Socio-geography of human mobility: A study using longitudinal mobile phone data. PLOS ONE 7(6): e39253. https://doi.org/10.1371/journal.pone.0039253
    """
    return _uid_apply(_home_location_ma, traj, show_progress, week_period=week_period, radius=radius, mode=mode)


def home_location_tc(traj, week_period=None, start_time='22:00', end_time='07:00', radius=None, mode='sjoin', show_progress=True):
    """Home location - Time Constraint criterion.
    
    Compute the home location of a set of individuals in a TrajDataFrame. 
    The home location :math:`h(u)_{\hbox{\tiny{TC}}}` of an individual :math:`u` is defined as the location :math:`u` visits the most during nighttime [PFSCB2021]_ [CBTDHVSB2012]_ [PSO2012]_: 
    
    .. math:: 
        h(u)_{\hbox{\tiny{TC}}} = arg\max_{i} |\{r_i | t(r_i) \in WP \cap [t_{startTime},\;t_{endTime}] \}|
    
    where :math:`r_i` is a location visited by :math:`u`, :math:`t(r_i)` is the time when :math:`u` visited :math:`r_i`, and :math:`WP` is the week period and :math:`t_{startTime}` and :math:`t_{endTime}` indicates the times when nighttime starts and ends, respectively.

    If a :math:`radius` is also specified, a spatial perimeter will be implemented around each location, which aggregates all activities within and computes the :math:`radius activity`.  
    In this case, the home location will be calculated based on the highest :math:`radius activity`. 

    The :math:`radius activity` can be be computed using a spatial join method to check which points are within around the specified radius or a .


    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    week_period : str, optional
        the week period to consider. It can be 'WK' (week days), 'WE' (weekend days) or None (every day of the week). The default is None.

    start_time : str, optional
        the starting time of the night (format HH:MM). The default is '22:00'.
        
    end_time : str, optional
        the ending time for the night (format HH:MM). The default is '07:00'.

    radius : int, optional
        if specified, a spatial perimeter of a certain radius will be implemented around each location, the radius is expressed in kilometers. The default is None.
    
    mode : str, optional
        the method used to compute the :math:`radius activity`. It can be 'sjoin' (spatial join) or 'distance_matrix' . The default is 'sjoin'.

    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    ----------
    pandas DataFrame
        the home location, as a :math:`(latitude, longitude)` pair, of the individuals.

    Examples
    --------
    >>> import skmob
    >>> from skmob.measures.home_detection import home_location_tc
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, 
                 names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user')
    >>> hl = home_location_tc(tdf, start_time='19:00', end_time='07:00')
    >>> print(hl.head())
       uid        lat         lng
    0    0  39.762146 -104.982480
    1    1  37.630490 -122.411084
    2    2  39.739154 -104.984703
    3    3  37.748170 -122.459192
    4    4  60.180171   24.949728

    References
    ----------
    .. [PFSCB2021] Pappalardo, Luca & Ferres, Leo & Sacasa, Manuel & Cattuto, Ciro & Bravo, Loreto. (2021). Evaluation of home detection algorithms on mobile phone data using individual-level ground truth. EPJ Data Science. 10. 29. 10.1140/epjds/s13688-021-00284-9. 
    .. [CBTDHVSB2012] Csáji, B. C., Browet, A., Traag, V. A., Delvenne, J.-C., Huens, E., Van Dooren, P., Smoreda, Z. & Blondel, V. D. (2012) Exploring the Mobility of Mobile Phone Users. Physica A: Statistical Mechanics and its Applications 392(6), 1459-1473, https://www.sciencedirect.com/science/article/pii/S0378437112010059
    .. [PSO2012] Phithakkitnukoon, S., Smoreda, Z. & Olivier, P. (2012) Socio-geography of human mobility: A study using longitudinal mobile phone data. PLOS ONE 7(6): e39253. https://doi.org/10.1371/journal.pone.0039253
    """
    return _uid_apply(_home_location_tc, traj, show_progress, week_period=week_period, radius=radius, mode=mode, start_time=start_time, end_time=end_time)


def home_location_dd(traj, show_progress=True):
    """Home location - Distinct Days criterion.
    
    Compute the home location of a set of individuals in a TrajDataFrame. 
    The home location is the location visited in the most distinct days [PFSCB2021]_ [CBTDHVSB2012]_ [PSO2012]_: 

    In case of a tie between the most distinct days, the location with the most visits in general will be selected.


    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    ----------
    pandas DataFrame
        the home location, as a :math:`(latitude, longitude)` pair, of the individuals.

    Examples
    --------
    >>> import skmob
    >>> from skmob.measures.home_detection import home_location_dd
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, 
                 names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user')
    >>> hl = home_location_dd(tdf)
    >>> print(hl.head())
        uid        lat         lng
    0    0  39.891077 -105.068532
    1    1  37.580304 -122.343679
    2    2  39.739154 -104.984703
    3    3  37.748170 -122.459192
    4    4  60.166970   24.944602

    References
    ----------
    .. [PFSCB2021] Pappalardo, Luca & Ferres, Leo & Sacasa, Manuel & Cattuto, Ciro & Bravo, Loreto. (2021). Evaluation of home detection algorithms on mobile phone data using individual-level ground truth. EPJ Data Science. 10. 29. 10.1140/epjds/s13688-021-00284-9. 
    .. [CBTDHVSB2012] Csáji, B. C., Browet, A., Traag, V. A., Delvenne, J.-C., Huens, E., Van Dooren, P., Smoreda, Z. & Blondel, V. D. (2012) Exploring the Mobility of Mobile Phone Users. Physica A: Statistical Mechanics and its Applications 392(6), 1459-1473, https://www.sciencedirect.com/science/article/pii/S0378437112010059
    .. [PSO2012] Phithakkitnukoon, S., Smoreda, Z. & Olivier, P. (2012) Socio-geography of human mobility: A study using longitudinal mobile phone data. PLOS ONE 7(6): e39253. https://doi.org/10.1371/journal.pone.0039253
    """
    return _uid_apply(_home_location_dd, traj, show_progress)
 

def home_location_in(traj, threshold=300, show_progress=True):
    """Home location - Inactivity criterion.
    
    Compute the home location of a set of individuals in a TrajDataFrame. 
    The home location is the most visited location preceding an inactivity period of time [OBB2020]_ [PFSCB2021]_ [CBTDHVSB2012]_ [PSO2012]_: 
    
    .. math:: 
        h(u)_{\hbox{\tiny{IN}}} = arg\max_{i} |\{r_i | t(r_i) - t(r_{i+1}) | > s \}|
    
    where :math:`r_i` is a location visited by :math:`u`, :math:`t(r_i)` is the time when :math:`u` visited, :math:`t(r_{i+1})` is the following time and :math:`s` is the threshold.


    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    threshold : integer, optional
        the threshold between two consecutive times, expressed in minutes.

    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    ----------
    pandas DataFrame
        the home location, as a :math:`(latitude, longitude)` pair, of the individuals.

    Examples
    --------
    >>> import skmob
    >>> from skmob.measures.home_detection import home_location_in
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, 
                 names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user')
    >>> hl = home_location_in(tdf, start_time='19:00', end_time='07:00')
    >>> print(hl.head())
       uid        lat         lng
    0    0  39.891586 -105.068463
    1    1  37.580304 -122.343679
    2    2  39.739154 -104.984703
    3    3  37.748170 -122.459192
    4    4  60.166970   24.944602

    References
    ----------
    .. [OBB2020] Oosterlinck, Dieter & Baecke, Philippe & Benoit, Dries. (2020). Home location prediction with telecom data: Benchmarking heuristics with a predictive modelling approach.. Expert Systems with Applications. 170. 114507. 10.1016/j.eswa.2020.114507. 
    .. [PFSCB2021] Pappalardo, Luca & Ferres, Leo & Sacasa, Manuel & Cattuto, Ciro & Bravo, Loreto. (2021). Evaluation of home detection algorithms on mobile phone data using individual-level ground truth. EPJ Data Science. 10. 29. 10.1140/epjds/s13688-021-00284-9. 
    .. [CBTDHVSB2012] Csáji, B. C., Browet, A., Traag, V. A., Delvenne, J.-C., Huens, E., Van Dooren, P., Smoreda, Z. & Blondel, V. D. (2012) Exploring the Mobility of Mobile Phone Users. Physica A: Statistical Mechanics and its Applications 392(6), 1459-1473, https://www.sciencedirect.com/science/article/pii/S0378437112010059
    .. [PSO2012] Phithakkitnukoon, S., Smoreda, Z. & Olivier, P. (2012) Socio-geography of human mobility: A study using longitudinal mobile phone data. PLOS ONE 7(6): e39253. https://doi.org/10.1371/journal.pone.0039253
    """
    return _uid_apply(_home_location_in, traj, show_progress, threshold=threshold)
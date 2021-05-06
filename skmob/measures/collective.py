import numpy as np
from scipy import stats
import pandas as pd
from datetime import timedelta
from collections import defaultdict
from skmob.measures.individual import home_location
from ..utils import constants
import sys
from tqdm import tqdm
tqdm.pandas()
from skmob.utils.gislib import getDistanceByHaversine

def _random_location_entropy_individual(traj):
    """
    Compute the random location entropy of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectory of the individual.
    
    Returns
    -------
    float
        the random location entropy of the individual.
    """
    n_distinct_users = len(traj.groupby(constants.UID))
    entropy = np.log2(n_distinct_users)
    return entropy

def random_location_entropy(traj, show_progress=True):
    """Random location entropy.
    
    Compute the random location entropy of the locations in a TrajDataFrame. The random location entropy of a location :math:`j` captures the degree of predictability of :math:`j` if each individual visits it with equal probability, and it is defined as:
    
    .. math::
        LE_{rand}(j) = log_2(N_j)
        
    where :math:`N_j` is the number of distinct individuals that visited location :math:`j`.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    show_progress : boolean, optional
        if True, show a progress bar. The default is True.
    
    Returns
    -------
    pandas DataFrame
        the random location entropy of the locations.
        
    Example
    -------
    >>> import skmob
    >>> from skmob.measures.collective import random_location_entropy
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, 
                 names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user')
    >>> rle_df = random_location_entropy(tdf, show_progress=True).sort_values(by='random_location_entropy', ascending=False)
    >>> print(rle_df.head())
                 lat         lng  random_location_entropy
    10286  39.739154 -104.984703                 6.129283
    49      0.000000    0.000000                 5.643856
    5991   37.774929 -122.419415                 5.523562
    12504  39.878664 -104.682105                 5.491853
    5377   37.615223 -122.389979                 5.247928   
    
    See Also
    --------
    uncorrelated_location_entropy
    """
    # if 'uid' column in not present in the TrajDataFrame
    if constants.UID not in traj.columns:
        all_locations = traj[[constants.LATITUDE, constants.LONGITUDE]].drop_duplicates([constants.LATITUDE, constants.LONGITUDE])
        all_locations['random_location_entropy'] = 0.0
        return all_locations
    
    if show_progress:
        df = pd.DataFrame(traj.groupby([constants.LATITUDE, constants.LONGITUDE]).progress_apply(lambda x: _random_location_entropy_individual(x)))
    else:
        df = pd.DataFrame(traj.groupby([constants.LATITUDE, constants.LONGITUDE]).apply(lambda x: _random_location_entropy_individual(x)))
    column_name = sys._getframe().f_code.co_name
    return df.reset_index().rename(columns={0: column_name})

def _uncorrelated_location_entropy_individual(traj, normalize=True):
    n = len(traj)
    probs = [1.0 * len(group) / n for group in traj.groupby(by=constants.UID).groups.values()]
    entropy = stats.entropy(probs)
    if normalize:
        n_unique_users = len(traj[constants.UID].unique())
        if n_unique_users > 1:
            entropy /= np.log2(n_unique_users)
        else:  # to avoid NaN
            entropy = 0.0
    return entropy


def uncorrelated_location_entropy(traj, normalize=False, show_progress=True):
    """Temporal-uncorrelated entropy.
    
    Compute the temporal-uncorrelated location entropy of the locations in a TrajDataFrame. The temporal-uncorrelated location entropy :math:`LE_{unc}(j)` of a location :math:`j` is the historical probability that :math:`j` is visited by an individual :math:`u`. Formally, it is defined as [CML2011]_:
    
    .. math::
        LE_{unc}(j) = -\sum_{i=j}^{N_j} p_jlog_2(p_j)
        
    where :math:`N_j` is the number of distinct individuals that visited :math:`j` and :math:`p_j` is the historical probability that a visit to location :math:`j` is by individual :math:`u`. 
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    normalize : boolean, optional
        if True, normalize the location entropy by dividing by :math:`log2(N_j)`, where :math:`N_j` is the number of
        distinct individuals that visited location :math:`j`. The default is False.
    
    show_progress : boolean
        if True, show a progress bar. The default is True.
    
    Returns
    -------
    pandas DataFrame
        the temporal-uncorrelated location entropies of the locations.

    Examples
    --------
    >>> import skmob
    >>> from skmob.measures.collective import uncorrelated_location_entropy
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, 
                 names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user')
    >>> ule_df = uncorrelated_location_entropy(tdf, show_progress=True).sort_values(by='uncorrelated_location_entropy', ascending=False)
    >>> print(ule_df.head())
                 lat         lng  uncorrelated_location_entropy
    12504  39.878664 -104.682105                       3.415713
    5377   37.615223 -122.389979                       3.176950
    10286  39.739154 -104.984703                       3.118656
    12435  39.861656 -104.673177                       2.918413
    12361  39.848233 -104.675031                       2.899175
    dtype: float64
    
    References
    ----------
    .. [CML2011] Cho, E., Myers, S. A. & Leskovec, J. (2011) Friendship and mobility: user movement in location-based social networks. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining, 1082-1090, https://dl.acm.org/citation.cfm?id=2020579
        
    See Also
    --------
    random_location_entropy
    """
    # if 'uid' column in not present in the TrajDataFrame
    if constants.UID not in traj.columns:
        all_locations = traj[[constants.LATITUDE, constants.LONGITUDE]].drop_duplicates([constants.LATITUDE, constants.LONGITUDE])
        all_locations['uncorrelated_location_entropy'] = 0.0
        return all_locations
    
    if show_progress:
        df = pd.DataFrame(traj.groupby([constants.LATITUDE, constants.LONGITUDE]).progress_apply(lambda x: _uncorrelated_location_entropy_individual(x, normalize=normalize)))
    else:
        df = pd.DataFrame(traj.groupby([constants.LATITUDE, constants.LONGITUDE]).apply(lambda x: _uncorrelated_location_entropy_individual(x, normalize=normalize)))
    column_name = sys._getframe().f_code.co_name
    if normalize:
        column_name = 'norm_%s' % sys._getframe().f_code.co_name
    return df.reset_index().rename(columns={0: column_name})

def _square_displacement(traj, delta_t):
    """
    Compute the square displacement after time `delta_t` of a single individual given their TrajDataFrame.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectory of the individual.
        
    delta_t : datetime.timedelta
        the time from the reference position
        
    Returns
    -------
    float
        the square displacement of the individual.
    """
    r0 = traj.iloc[0]
    t = r0[constants.DATETIME] + delta_t
    rt = traj[traj[constants.DATETIME] <= t].iloc[-1]
    square_displacement = getDistanceByHaversine((r0.lat, r0.lng), (rt.lat, rt.lng)) ** 2
    return square_displacement


def mean_square_displacement(traj, days=0, hours=1, minutes=0, show_progress=True):
    """Mean Square Displacement.
    
    Compute the mean square displacement across the individuals in a TrajDataFrame. The mean squared displacement is a measure of the deviation of the position of an object with respect to a reference position over time [BHG2006]_ [SKWB2010]_. It is defined as:
    
    .. math::
        MSD = \\langle |r(t) - r(0)| \\rangle = \\frac{1}{N} \sum_{i = 1}^N |r^{(i)}(t) - r^{(i)}(0)|^2

    where :math:`N` is the number of individuals to be averaged, vector :math:`x^{(i)}(0)` is the reference position of the :math:`i`-th individual, and vector :math:`x^{(i)}(t)` is the position of the :math:`i`-th individual at time :math:`t` [FS2002]_.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    days : int, optional
        the days since the starting time. The default is 0.
    
    hours : int, optional
        the hours since the days since the starting time. The default is 1.
    
    minutes : int, optional
        the minutes since the hours since the days since the starting time. The default is 0.

    show_progress : boolean, optional
        if True, show a progress bar. The default is True. 
    
    Returns
    -------
    float
        the mean square displacement.

    Warning
    -------
    The input TrajDataFrame must be sorted in ascending order by `datetime`.

    Examples
    --------
    >>> import skmob
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user').sort_values(by='datetime')
    >>> msd = mean_square_displacement(tdf, days=0, hours=1, minutes=0)
    >>> print(msd)
    534672.3361996822

    References
    ----------
    .. [FS2002] Frenkel, D. & Smit, B. (2002) Understanding molecular simulation: From algorithms to applications. Academic Press, 196 (2nd Ed.), https://www.sciencedirect.com/book/9780122673511/understanding-molecular-simulation.
    .. [BHG2006] Brockmann, D., Hufnagel, L. & Geisel, T. (2006) The scaling laws of human travel. Nature 439, 462-465, https://www.nature.com/articles/nature04292
    .. [SKWB2010] Song, C., Koren, T., Wang, P. & Barabasi, A.L. (2010) Modelling the scaling properties of human mobility. Nature Physics 6, 818-823, https://www.nature.com/articles/nphys1760
    """
    delta_t = timedelta(days=days, hours=hours, minutes=minutes)
    
    # if 'uid' column in not present in the TrajDataFrame
    if constants.UID not in traj.columns:
        return _square_displacement(traj, delta_t)
    
    if show_progress:
        return traj.groupby(constants.UID).progress_apply(lambda x: _square_displacement(x, delta_t)).mean()
    else:
        return traj.groupby(constants.UID).apply(lambda x: _square_displacement(x, delta_t)).mean()


def visits_per_location(traj):
    """Visits per location.
    
    Compute the number of visits to each location in a TrajDataFrame [PF2018]_.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    Returns
    -------
    pandas DataFrame
        the number of visits per location.

    Examples
    --------
    >>> import skmob
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user').sort_values(by='datetime')
    >>> vl_df = visits_per_location(df)
    >>> print(vl_df.head())
             lat         lng  n_visits
    0  39.739154 -104.984703      3392
    1  37.580304 -122.343679      2248
    2  39.099275  -76.848306      1715
    3  39.762146 -104.982480      1442
    4  40.014986 -105.270546      1310

    References
    ----------
    .. [PF2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4
        
    See also
    --------
    homes_per_location
    """
    return traj.groupby([constants.LATITUDE,
                          constants.LONGITUDE]).count().sort_values(by=constants.DATETIME,
                                                                    ascending=False)[[constants.DATETIME]].reset_index().rename({constants.DATETIME: 'n_visits'}, axis=1)


def homes_per_location(traj, start_night='22:00', end_night='07:00'):
    """Homes per location.
    
    Compute the number of home locations in each location. The number of home locations in a location :math:`j` is computed as [PRS2016]_:
    
    .. math:: 
        N_{homes}(j) = |\{h_u | h_u = j, u \in U \}|

    where :math:`h_u` indicates the home location of an individual :math:`u` and :math:`U` is the set of individuals.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    start_night : str, optional
        the starting time of the night (format HH:MM). The default is '22:00'.
        
    end_night : str, optional
        the ending time for the night (format HH:MM). The default is '07:00'.
    
    Returns
    -------
    pandas DataFrame
        the number of homes per location.

    Examples
    --------
    >>> import skmob
    >>> from skmob.measures.collective import homes_per_location
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user').sort_values(by='datetime')
    >>> hl_df = homes_per_location(tdf).sort_values(by='n_homes', ascending=False)
    >>> print(hl_df.head())
             lat         lng  n_homes
    0  39.739154 -104.984703       15
    1  37.584103 -122.366083        6
    2  40.014986 -105.270546        5
    3  37.580304 -122.343679        5
    4  37.774929 -122.419415        4

    References
    ----------
    .. [PRS2016] Pappalardo, L., Rinzivillo, S. & Simini, F. (2016) Human Mobility Modelling: exploration and preferential return meet the gravity model. Procedia Computer Science 83, 934-939, http://dx.doi.org/10.1016/j.procs.2016.04.188
    """
    # if column 'uid' is not present in the TrajDataFrame
    uid_flag = False
    if constants.UID not in traj.columns:
        traj['uid'] = 0
        uid_flag = True
    df = home_location(traj,
                         start_night=start_night,
                         end_night=end_night).groupby([constants.LATITUDE,
                                                       constants.LONGITUDE]).count().sort_values(constants.UID,
                                                                                                 ascending=False).reset_index().rename(
        columns={constants.UID: 'n_homes'})
    if uid_flag:
        traj.drop('uid', axis=1, inplace=True)
    return df


def visits_per_time_unit(traj, time_unit='1h'):
    """Visits per time unit.
    
    Compute the number of data points per time unit in the TrajDataFrame [PRS2016]_.
    
    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.
    
    time_unit : str, optional
        the time unit to use for grouping the time slots. The default '1h', which creates slots of 1 hour. For full specification of available time units, see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    
    Returns
    -------
    pandas Series
        the number of visits per time unit.

    Examples
    --------
    >>> import skmob
    >>> from skmob.measures.collective import visits_per_time_unit
    >>> url = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"
    >>> df = pd.read_csv(url, sep='\\t', header=0, nrows=100000, names=['user', 'check-in_time', 'latitude', 'longitude', 'location id'])
    >>> tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime='check-in_time', user_id='user').sort_values(by='datetime')
    >>> vtu_df = visits_per_time_unit(df)
    >>> print(vtu_df.head())
                               n_visits
    datetime                           
    2008-03-22 05:00:00+00:00         2
    2008-03-22 06:00:00+00:00         2
    2008-03-22 07:00:00+00:00         0
    2008-03-22 08:00:00+00:00         0
    2008-03-22 09:00:00+00:00         0
    """
    return pd.DataFrame(traj[constants.DATETIME]).set_index(traj[constants.DATETIME]).groupby(pd.Grouper(freq=time_unit)).count().rename(columns={constants.DATETIME: 'n_visits'})

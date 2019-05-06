from math import sqrt, sin, cos, pi, asin, pow, ceil, log
import numpy as np
from scipy import stats
import pandas as pd
from collections import defaultdict
import sys
from tqdm import tqdm
from skmob.utils.gislib import getDistanceByHaversine
tqdm.pandas()
from ..utils import constants

def _radius_of_gyration_individual(traj):
    """
    Compute the radius of gyration of an individual given their TrajDataFrame

    :param traj: the trajectory of the individual
    :type traj: TrajDataFrame

    :return: the radius of gyration of the individual
    :rtype: float
    """
    lats_lngs = traj[[constants.LATITUDE, constants.LONGITUDE]].values
    center_of_mass = np.mean(lats_lngs, axis=0)
    rg = np.sqrt(np.mean([getDistanceByHaversine((lat, lng), center_of_mass) ** 2.0 for lat, lng in lats_lngs]))
    return rg


def radius_of_gyration(traj, show_progress=True):
    """
    Compute the radii of gyration (in kilometers) of a set of individuals given a TrajDataFrame.
    The radius of gyration :math:`r_g(u)` of an individual :math:`u` indicates the characteristic distance travelled by
    :math:`u` during a time period.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean

    :return: the radii of gyration of the individuals
    :rtype: DataFrame

    Examples:
        Computing the radius of gyration of each individual from a DataFrame of trajectories
    >>> import skmob
    >>> from skmob import TrajDataFrame
    >>> from skmob.measures.individual import radius_of_gyration
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> radius_of_gyration(traj).head()
       uid  radius_of_gyration
    0    1           20.815129
    1    2           15.689436
    2    3           14.918760
    3    4           14.706123
    4    5           23.990570

    References:
        .. [gonzalez2008understanding] Gonzalez, Marta C., Hidalgo, Cesar A. and Barabasi, Albert-Laszlo. "Understanding individual human mobility patterns." Nature 453, no. 7196 (2008): 779--782.
        .. [pappalardo2013understanding] Pappalardo, L., Rinzivillo, S., Qu, Z., Pedreschi, D., Giannotti, F. "Understanding the patterns of car travel." European Physics Journal Special Topics 215, no. 61 (2013).
        .. [zhao2008empirical] M. Zhao, L. Mason, W. Wang, "Empirical study on human mobility for mobile wireless networks", in:  Military Communications Conference, 2008. MILCOM 2008. IEEE, IEEE, 2008, pp. 1–7.
        .. [song2010modelling] Song, Chaoming, Koren, Tal, Wang, Pu and Barabasi, Albert-Laszlo. "Modelling the scaling properties of human mobility." Nature Physics 6 , no. 10 (2010): 818--823.

    .. seealso:: :func:`k_radius_of_gyration`
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _radius_of_gyration_individual(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _radius_of_gyration_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name}) 


def _k_radius_of_gyration_individual(traj, k=2):
    """
    Compute the k-radius of gyration of a single individual given their TrajDataFrame

    :param traj: the trajectories of the individual
    :type traj: TrajDataFrame

    :param k: the number of most frequent locations to consider
    :type k: int

    :return: float
    """
    traj['visits'] = traj.groupby([constants.LATITUDE, constants.LONGITUDE]).transform('count')[constants.DATETIME]
    top_k_locations = traj.drop_duplicates(subset=[constants.LATITUDE, constants.LONGITUDE]).sort_values(by=['visits', constants.DATETIME],
                                                                              ascending=[False, True])[:k]
    visits = top_k_locations['visits'].values
    total_visits = sum(visits)
    lats_lngs = top_k_locations[[constants.LATITUDE, constants.LONGITUDE]].values

    center_of_mass = visits.dot(lats_lngs) / total_visits
    krg = np.sqrt(sum([visits[i] * (getDistanceByHaversine((lat, lng), center_of_mass) ** 2.0)
                       for i, (lat, lng) in enumerate(lats_lngs)]) / total_visits)
    return krg


def k_radius_of_gyration(traj, k=2, show_progress=True):
    """
    Compute the k-radius of gyration (in kilometers) of a set of individuals given a TrajDataFrame.
    The k-radius of gyration :math:`r_g^{(k)}(u)` indicates the characteristic distance travelled by an individual
    between their $k$ most frequent locations.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame

    :param int k: the number of most frequent locations to consider, default=2, range=[2, +inf]
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the k-radii of gyration of the individuals
    :rtype: pandas DataFrame

    Examples:
        Computing the k-radius of gyration of each individual from a DataFrame of trajectories
    >>> import skmob
    >>> from skmob import TrajDataFrame
    >>> from skmob.measures.individual import radius_of_gyration
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> k_radius_of_gyration(tdf).head()
       uid  k_radius_of_gyration
    0    1              1.798615
    1    2              2.449305
    2    3              1.223604
    3    4              6.034151
    4    5             20.678760

    .. seealso:: :func:`radius_of_gyration`

    References:
        .. [pappalardo2015returners] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F., Barabasi, A.-L. "Returners and Explorers dichotomy in human mobility.", Nature Communications, 6:8166, doi: 10.1038/ncomms9166 (2015).
        .. [barbosa2016returners] Barbosa H.S., de Lima Neto F.B., Evsukoff A., Menezes R. "Returners and Explorers Dichotomy in Web Browsing Behavior - A Human Mobility Approach.". Complex Networks VII. Studies in Computational Intelligence, vol 644. (2016).
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _k_radius_of_gyration_individual(x, k))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _k_radius_of_gyration_individual(x, k))
    return pd.DataFrame(df).reset_index().rename(columns={0: '%s%s' % (k, sys._getframe().f_code.co_name)})


def _random_entropy_individual(traj):
    """
    Compute the random entropy of a single individual given their TrajDataFrame

    :param traj: the trajectories of the individual
    :type traj: TrajDataFrame

    :return: float
    """
    n_distinct_locs = len(traj.groupby([constants.LATITUDE, constants.LONGITUDE]))
    entropy = np.log2(n_distinct_locs)
    return entropy


def random_entropy(traj, show_progress=True):
    """
    Compute the random entropy of a set of individuals given a TrajDataFrame.
    The random entropy :math:`E_{rand}(u)` (in base 2) of an individual :math:`u` captures the degree of predictability of :math:`u`'s whereabouts if each location is visited with equal probability.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the random entropies of the individuals
    :rtype: pandas DataFrame

    Examples:
        Computing the random entropy of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import random_entropy
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> random_entropy(tdf).head()
       uid  random_entropy
    0    1        6.658211
    1    2        6.942515
    2    3        6.491853
    3    4        6.228819
    4    5        6.727920

    .. seealso:: :func:`uncorrelated_entropy`, :func:`real_entropy`

    References:
        .. [eagle2009eigenbehaviors] Eagle, Nathan and Pentland, Alex Sandy. "Eigenbehaviors: identifying structure in routine." Behavioral Ecology and Sociobiology 63 , no. 7 (2009): 1057--1066.
        .. [song2010limits] Song, Chaoming, Qu, Zehui, Blumm, Nicholas and Barabási, Albert-László. "Limits of Predictability in Human Mobility." Science 327 , no. 5968 (2010): 1018-1021.
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _random_entropy_individual(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _random_entropy_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})


def _uncorrelated_entropy_individual(traj, normalize=False):
    """
    Compute the uncorrelated entropy of a single individual given their TrajDataFrame

    :param traj: the trajectories of the individual
    :type traj: TrajDataFrame

    :param normalize: if True normalize the entropies

    :return: float
    """
    n = len(traj)
    probs = [1.0 * len(group) / n for group in traj.groupby(by=[constants.LATITUDE, constants.LONGITUDE]).groups.values()]
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = len(np.unique(traj[[constants.LATITUDE, constants.LONGITUDE]].values, axis=0))
        if n_vals > 1:
            entropy /= np.log2(n_vals)
        else:  # to avoid NaN
            entropy = 0.0
    return entropy


def uncorrelated_entropy(traj, normalize=False, show_progress=True):
    """
    Compute the temporal-uncorrelated entropy of a set of individuals given a TrajDataFrame. The temporal-uncorrelated entropy :math:`E_{unc}(u)` (in base 2) of an individual :math:`u` is the historical probability that a location :math:`j` was visited by :math:`u` and characterizes the heterogeneity of :math:`u`'s visitation patterns.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame

    :param boolean normalize: if True normalize the entropy by dividing by log2(N), where N is the number of
        distinct locations visited by the individual

    :param show_progress: if True show a progress bar
    :type show_progress: boolean

    :return: the temporal-uncorrelated entropies of the individuals
    :rtype: pandas DataFrame

    Examples:
        Computing the temporal uncorrelated entropy of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import uncorrelated_entropy
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> uncorrelated_entropy(tdf, normalize=True).head()
       uid  uncorrelated_entropy
    0    1              5.492801
    1    2              5.764952
    2    3              4.628958
    3    4              5.112809
    4    5              5.696118

    .. seealso:: :func:`random_entropy`, :func:`real_entropy`

    References:
        .. [eagle2009eigenbehaviors] Eagle, Nathan and Pentland, Alex Sandy. "Eigenbehaviors: identifying structure in routine." Behavioral Ecology and Sociobiology 63 , no. 7 (2009): 1057--1066.
        .. [song2010limits] Song, Chaoming, Qu, Zehui, Blumm, Nicholas and Barabási, Albert-László. "Limits of Predictability in Human Mobility." Science 327 , no. 5968 (2010): 1018-1021.
        .. [pappalardo2016analytical] Pappalardo, Luca, et al. "An analytical framework to nowcast well-being using mobile phone data." International Journal of Data Science and Analytics 2, no. 75 (2016)
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _uncorrelated_entropy_individual(x, normalize=normalize))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _uncorrelated_entropy_individual(x, normalize=normalize))
    column_name = sys._getframe().f_code.co_name
    if normalize:
        column_name = 'norm_%s' % sys._getframe().f_code.co_name
    return pd.DataFrame(df).reset_index().rename(columns={0: column_name})


def _stringify(seq):
    return '|'.join(['_'.join(list(map(str, r))) for r in seq])


def _true_entropy(sequence):
    n = len(sequence)

    # these are the first and last elements
    sum_lambda = 1. + 2.

    for i in range(1, n - 1):
        str_seq = _stringify(sequence[:i])
        j = 1
        str_sub_seq = _stringify(sequence[i:i + j])
        while str_sub_seq in str_seq:
            j += 1
            str_sub_seq = _stringify(sequence[i:i + j])
            if i + j == n:
                # EOF character
                j += 1
                break
        sum_lambda += j

    return 1. / sum_lambda * n * np.log2(n)


def _real_entropy_individual(traj):
    """
    Compute the real entropy of a single individual given their TrajDataFrame

    :param traj: the trajectories of the individual
    :type traj: TrajDataFrame

    :return: float
    """
    time_series = tuple(map(tuple, traj[[constants.LATITUDE, constants.LONGITUDE]].values))
    entropy = _true_entropy(time_series)
    return entropy


def real_entropy(traj, show_progress=True):
    """
    Compute the real entropy of a set of individuals given a TrajDataFrame. The real entropy :math:`E(u)` of an individual :math:`u` depends not only on the frequency of visitation, but also the order in which the nodes were visited and the time spent at each location, thus capturing the full spatiotemporal order present in an :math:`u`'s mobility patterns.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the real entropies of the individuals
    :rtype: pandas DataFrame
    
    Examples:
        Computing the real entropy of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import real_entropy
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> real_entropy(tdf).head()
       uid  real_entropy
    0    1      4.416683
    1    2      4.643135
    2    3      3.759747
    3    4      4.255130
    4    5      4.601280

    .. seealso:: :func:`random_entropy`, :func:`uncorrelated_entropy`

    References:
        .. [song2010limits] Song, Chaoming, Qu, Zehui, Blumm, Nicholas and Barabási, Albert-László. "Limits of Predictability in Human Mobility." Science 327 , no. 5968 (2010): 1018-1021.
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _real_entropy_individual(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _real_entropy_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})


def _jump_lengths_individual(traj):
    """
    Compute the jump lengths (in kilometers) of a single individual from their TrajDataFrame

    :param traj: the TrajDataFrame of a single individual
    :return: the list of distances (in kilometers) traveled by the individual
    :rtype: list
    """
    if len(traj) == 1:  # if there is just one point, no distance can be computed
        return []
    lats_lngs = traj.sort_values(by=constants.DATETIME)[[constants.LATITUDE, constants.LONGITUDE]].values
    lengths = np.array([getDistanceByHaversine(lats_lngs[i], lats_lngs[i - 1]) for i in range(1, len(lats_lngs))])
    return lengths


def jump_lengths(traj, show_progress=True, merge=False):
    """
    Compute the jump lengths (also called trip distances) (in kilometers) given a TrajDataFrame.
    A jump length (or trip distance) is defined as the geographic distance between the position of a visit and the next one.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame

    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :param merge: if True merge the individuals' lists into one list (default: False)
    :type merge: boolean
    
    :return: the jump lengths for each individual (a NaN indicate that an individual visited just one location) or a list with all jumps together
    :rtype: pandas DataFrame or list

    Examples:
        Computing the jump lenghts of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import jump_lengths
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> jump_lengths(tdf).head()
       uid                                       jump_lengths
    0    1  [0.0, 2.235208041718699, 0.996946934364258, 68...
    1    2  [0.0, 0.0, 5.649741880693563, 1.41265548245858...
    2    3  [2.2311820297176572, 2.824995904497732, 4.9916...
    3    4  [48.52874706948018, 1.4124821185158252, 0.9972...
    4    5  [0.0, 1.0004402836501967, 2.825374437917952, 1...

    .. seealso:: :func:`maximum_distance` :func:`distance_straight_line`
https://scholar.google.it/citations?user=88VJDhcAAAAJ&hl=it&oi=ao
    References:
        .. [brockmann2006scaling] Brockmann, D., Hufnagel, L. and Geisel, T.. "The scaling laws of human travel." Nature 439 (2006): 462.
        .. [gonzalez2008understanding] Gonzalez, Marta C., Hidalgo, Cesar A. and Barabasi, Albert-Laszlo. "Understanding individual human mobility patterns." Nature 453, no. 7196 (2008): 779--782.
        .. [pappalardo2013understanding] Pappalardo, L., Rinzivillo, S., Qu, Z., Pedreschi, D., Giannotti, F. "Understanding the patterns of car travel." European Physics Journal Special Topics 215, no. 61 (2013)
        .. [shin2008levy] R. Shin, S. Hong, K. Lee, S. Chong, "On the levy-walk nature of human mobility: Do humans walk like monkeys?", in: Proc. IEEE INFOCOM, 2008, pp. 924–932.
        .. [bazzani2010statistical] A. Bazzani, B. Giorgini, S. Rambaldi, R. Gallotti, L. Giovannini, "Statistical laws in urban mobility from microscopic gps data in the area of florence", Journal of Statistical Mechanics: Theory and Experiment 2010 (05) (2010) P05001.
        .. [turchin1998measuring] P. Turchin, "Quantitative Analysis of Movement: Measuring and Modeling Population Redistribution in Animals and Plants", Sinauer Associates, Sunderland, Massachusetts, USA, 1998.
        .. [song2010modelling] Song, Chaoming, Koren, Tal, Wang, Pu and Barabasi, Albert-Laszlo. "Modelling the scaling properties of human mobility." Nature Physics 6 , no. 10 (2010): 818--823.
        .. [zhao2015explaining] K. Zhao, M. Musolesi, P. Hui, W. Rao, S. Tarkoma, Explaining the power-law distribution of human mobility through transportation modality decomposition, Scientific reports 5. 2015
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _jump_lengths_individual(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _jump_lengths_individual(x))
    
    df = pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})
    
    if merge:
        # merge all lists 
        jl_list =[]
        for x in df.jump_lengths:
            jl_list.extend(x)
        return jl_list
        
    return df


def maximum_distance(traj, show_progress=True):
    """
    Compute the maximum distance (in kilometers) traveled by the individuals.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the maximum traveled distance for each individual (a NaN indicate that an individual visited just one location)
    :rtype: pandas DataFrame

    Examples:
        Computing the maximum distance traveled by each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import maximum_distance
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> maximum_distance(tdf).head()
       uid  maximum_distance
    0    1         84.923674
    1    2         73.190512
    2    3         92.713548
    3    4         79.425210
    4    5         73.036719

    .. seealso:: :func:`jump_lengths` :func:`distance_straight_line`

    References:
        .. [williams2014measures] Williams, Nathalie E., Thomas, Timothy A., Dunbar, Matthew, Eagle, Nathan and Dobra, Adrian. "Measures of Human Mobility Using Mobile Phone Records Enhanced with GIS Data." CoRR abs/1408.5420 (2014).
        .. [lu2012predictability] Lu, Xin, Bengtsson, Linus and Holme, Petter. "Predictability of population displacement after the 2010 haiti earthquake." National Academy of Sciences 109 , no. 29 (2012): 11576--11581.
    """
    def get_max_distance(traj):
        jumps = _jump_lengths_individual(traj)
        if len(jumps) > 0:
            return max(jumps)
        return np.NaN
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: get_max_distance(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: get_max_distance(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})


def distance_straight_line(traj, show_progress=True):
    """
    Compute the distance (in kilometers) traveled straight line by each individual,
    i.e., the sum of the distances traveled by the individual

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the straight line distance traveled by the individual (a NaN indicate that an individual visited just one location)
    :rtype: pandas DataFrame

    Examples:
        Computing the distance straight line of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import distance_straight_line
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> distance_straight_line(tdf).head()
       uid  distance_straight_line
    0    1             7917.805713
    1    2             5330.312205
    2    3             6481.041936
    3    4             3370.984587
    4    5             7300.563980

    .. seealso:: :func:`jump_lengths` :func:`maximum_distance`

    References:
        .. [williams2014measures] Williams, Nathalie E., Thomas, Timothy A., Dunbar, Matthew, Eagle, Nathan and Dobra, Adrian. "Measures of Human Mobility Using Mobile Phone Records Enhanced with GIS Data." CoRR abs/1408.5420 (2014).
    """
    def get_sum_distances(traj):
        jumps = _jump_lengths_individual(traj)
        if len(jumps) > 0:
            return sum(jumps)
        return 0.0
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: get_sum_distances(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: get_sum_distances(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})


def _waiting_times_individual(traj):
    """
    Compute the waiting times for a single individual, given their TrajDataFrame
    :param traj: the trajectories of the individual
    :type TrajDataFrame
    :return: a pandas DataFrame
    """
    if len(traj) == 1:
        return []
    times = traj.sort_values(by=constants.DATETIME)[constants.DATETIME]
    wtimes = times.diff()[1:].values.astype('timedelta64[s]').astype('float')
    return wtimes


def waiting_times(traj, show_progress=True, merge=False):
    """
    Compute the waiting times (or inter-times) between the movements of an individual.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame

    :param show_progress: if True show a progress bar
    :type show_progress: boolean

    :param merge: if True merge the individuals' lists into one list (default: False)
    :type merge: boolean

    :return: the list of waiting times of the individual (a NaN indicate that an individual visited just one location) or a list with all waiting times together
    :rtype: pandas DataFrame or list

    Examples:
        Computing the waiting times of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import waiting_times
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> waiting_times(df).head()
       uid                                      waiting_times
    0    1  [2005.0, 2203.0, 1293.0, 823.0, 2636.0, 893.0,...
    1    2  [1421.0, 1948.0, 1508.0, 815.0, 8954.0, 1248.0...
    2    3  [5014.0, 838.0, 1425.0, 1250.0, 993.0, 1596.0,...
    3    4  [5623.0, 617.0, 721.0, 1954.0, 956.0, 1479.0, ...
    4    5  [1461.0, 13354.0, 1258.0, 7966.0, 768.0, 615.0...sys._getframe().f_code.co_name

    References:
        .. [song2010modelling] Song, Chaoming, Koren, Tal, Wang, Pu and Barabasi, Albert-Laszlo. "Modelling the scaling properties of human mobility." Nature Physics 6 , no. 10 (2010): 818--823.
        .. [pappalardo2016human] Pappalardo, Luca, Rinzivillo, Salvatore, Simini, Filippo "Human Mobility Modelling: exploration and preferential return meet the gravity model." Procedia Computer Science, Volume 83, 2016, Pages 934-939
        .. [pappalardo2017data] Pappalardo, Luca and Simini, Filippo. "Data-driven generation of spatio-temporal routines in human mobility.." Data Min. Knowl. Discov. 32 , no. 3 (2018): 787-829.
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _waiting_times_individual(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _waiting_times_individual(x))
    
    df = pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})
    
    if merge:
        wl_list =[]
        for x in df.waiting_times:
            wl_list.extend(x)
        return wl_list
    
    return df


def _number_of_locations_individual(traj):
    """
    Compute the number of visited locations of a single individual given their TrajDataFrame

    :param traj: the trajectories of the individual
    :type traj: TrajDataFrame

    :return: int
    """
    n_locs = len(traj.groupby([constants.LATITUDE, constants.LONGITUDE]).groups)
    return n_locs


def number_of_locations(traj, show_progress=True):
    """
    Compute the number of distinct locations visited by an individual.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the number of distinct locations visited by the individuals
    :rtype: pandas DataFrame

    Examples:
        Computing the number of locations of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import number_of_locations
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file(datadata,  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> number_of_locations(tdf).head()
       uid  number_of_locations
    0    1                  101
    1    2                  123
    2    3                   90
    3    4                   75
    4    5                  106

    References:
    ----------
    .. [gonzalez2008understanding] Gonzalez, Marta C., Hidalgo, Cesar A. and Barabasi, Albert-Laszlo. "Understanding individual human mobility patterns." Nature 453, no. 7196 (2008): 779--782.
    .. [pappalardo2013understanding] Pappalardo, L., Rinzivillo, S., Qu, Z., Pedreschi, D., Giannotti, F. "Understanding the patterns of car travel." European Physics Journal Special Topics 215, no. 61 (2013).
    .. [williams2014measures] Williams, Nathalie E., Thomas, Timothy A., Dunbar, Matthew, Eagle, Nathan and Dobra, Adrian. "Measures of Human Mobility Using Mobile Phone Records Enhanced with GIS Data." CoRR abs/1408.5420 (2014).
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _number_of_locations_individual(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _number_of_locations_individual(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})


def _home_location_individual(traj, start_night='22:00', end_night='07:00'):
    """
    Compute the home location of a single individual, given their TrajDataFrame
    :param traj: the trajectories of the individual
    :type traj: TrajDataFrame

    :param start_night:
    :param end_night:
    :return: tuple
    """
    night_visits = traj.set_index(pd.DatetimeIndex(traj.datetime)).between_time(start_night, end_night)
    if len(night_visits) != 0:
        lat, lng = night_visits.groupby([constants.LATITUDE, constants.LONGITUDE]).count().sort_values(by=constants.UID, ascending=False).iloc[0].name
    else:
        lat, lng = traj.groupby([constants.LATITUDE, constants.LONGITUDE]).count().sort_values(by=constants.UID, ascending=False).iloc[0].name
    home_coords = (lat, lng)
    return home_coords


def home_location(traj, start_night='22:00', end_night='07:00', show_progress=True):
    """
    Compute the home location of an individual

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param str start_night: the starting hour for the night
    :param str end_night: the ending hour for the night
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the home location of the individuals
    :rtype: pandas DataFrame

    Examples:
        Computing the home location of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import home_location
    >>> df = skmob.read_trajectories('data_test/gps_test_dataset.csvdata', user_id='user', longitude='lon')
    >>> home_location(df).head()
       uid        lat        lng
    0    1  46.126505  11.867149
    1    2  46.442010  10.998368
    2    3  45.818599  11.130413
    3    4  45.873280  11.093846
    4    5  46.215770  11.067935

    References:
        .. [csaji2012exploring] Csáji, Balázs Csanád, Browet, Arnaud, Traag, Vincent A., Delvenne, Jean-Charles, Huens, Etienne, Dooren, Paul Van, Smoreda, Zbigniew and Blondel, Vincent D. "Exploring the Mobility of Mobile Phone Users." CoRR abs/1211.6014 (2012).
        .. [phithakkitnukoon2012socio] Phithakkitnukoon, Santi, Smoreda, Zbigniew and Olivier, Patrick. "Socio-geography of human mobility: A study using longitudinal mobile phone data." PloS ONE 7 , no. 6 (2012): e39253.
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _home_location_individual(x, start_night=start_night, end_night=end_night))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _home_location_individual(x, start_night=start_night, end_night=end_night))
    return df.apply(pd.Series).reset_index().rename(columns={0: constants.LATITUDE, 1: constants.LONGITUDE})


def _max_distance_from_home_individual(traj, start_night='22:00', end_night='07:00'):
    """
    Compute the maximum distance from home traveled by a single individual, given their TrajDataFrame

    :param traj: the trajectories of the individual
    :type traj: TrajDataFrame

    :param str start_night: the starting hour for the night
    :param str end_night: the ending hour for the night
    
    :return: float
    """
    home = home_location(traj, start_night=start_night, end_night=end_night, show_progress=False).iloc[0]
    lats_lngs = traj.sort_values(by=constants.DATETIME)[[constants.LATITUDE, constants.LONGITUDE]].values
    lengths = np.array([getDistanceByHaversine((lat, lng), (home[constants.LATITUDE], home[constants.LONGITUDE])) for i, (lat, lng) in enumerate(lats_lngs)])
    return lengths.max()


def max_distance_from_home(traj, start_night='22:00', end_night='07:00', show_progress=True):
    """
    Compute the maximum distance from home (in kilometers) traveled by an individual.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param str start_night: the starting hour for the night
    :param str end_night: the ending hour for the night
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the maximum distance from home of the individuals
    :rtype: pandas DataFrame

    Examples:
        Computing the maximum distance from home of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import max_distance_from_home
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data_test/brightkite_data.csv'data,  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> max_distance_from_home(tdf).head()
       uid  max_distance_from_home
    0    1               46.409510
    1    2               68.499333
    2    3               56.806038
    3    4               78.949592
    4    5               69.393777
date_time
    .. seealso:: :func:`maximum_distance`, :func:`home_location`

    References:
        .. [canzian2015trajectories] Luca Canzian and Mirco Musolesi. "Trajectories of depression: unobtrusive monitoring of depressive states by means of smartphone mobility traces analysis." In Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '15), 1293--1304, 2015.
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _max_distance_from_home_individual(x, start_night=start_night, end_night=end_night))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _max_distance_from_home_individual(x, start_night=start_night, end_night=end_night))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})


def number_of_visits(traj, show_progress=True):
    """
    Compute the number of visits or points in an individual's trajectory

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the number of visits or points per each individual
    :rtype: pandas DataFrame

    Examples:
            Computing the number of visits per location from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import number_of_visits
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data_test/brightkite_data.csv'data,  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> number_of_visits(tdf).head()
       uid  number_of_visits
    0    1               340
    1    2               316
    2    3               355
    3    4               375
    4    5               245
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: len(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: len(x))
    return pd.DataFrame(df).reset_index().rename(columns={0: sys._getframe().f_code.co_name})


def _location_frequency_individual(traj, normalize=True):
    """
    Compute the visitation frequency of each location for a single individual, given their TrajDataFrame
    :param traj: the trajectories of the individual
    :type TrajDataFrame
    :param normalize: if True compute the ratio of visits, otherwise the row count of visits to each location
    :return: pandas DataFrame
    """
    freqs = traj.groupby([constants.LATITUDE,
                          constants.LONGITUDE]).count()[constants.DATETIME].sort_values(ascending=False)
    if normalize:
        freqs /= freqs.sum()
    return freqs


def location_frequency(traj, normalize=True, show_progress=True):
    """
    Visitation frequency of each location

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    
    :param boolean normalize: if True, the frequencies are normalized (divided by the individual's total number of visits)
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the location frequency for each location of the individuals
    :rtype: pandas DataFrame

    Examples:
            Computing the number of visits per location from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import location_frequency
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data_test/brightkite_data.csv'data,  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> location_frequency(df).head()
       uid        lat        lng  datetime
    0    1  46.180259  11.040689  0.255882
    1    1  46.180027  11.053637  0.055882
    2    1  46.337311  10.799639  0.044118
    3    1  46.307321  10.980472  0.032353
    4    1  46.144740  11.013478  0.026471

    .. seealso:: :func:`visits_per_location`

    References:
        .. [song2010modelling] Song, Chaoming, Koren, Tal, Wang, Pu and Barabasi, Albert-Laszlo. "Modelling the scaling properties of human mobility." Nature Physics 6 , no. 10 (2010): 818--823.
        .. [pappalardo2018data] Pappalardo, Luca and Simini, Filippo. "Data-driven generation of spatio-temporal routines in human mobility.." Data Min. Knowl. Discov. 32 , no. 3 (2018): 787-829.
    """
    if show_progress:
        df = pd.DataFrame(traj.groupby(constants.UID).progress_apply(lambda x: _location_frequency_individual(x, normalize=normalize)))
    else:
        df = pd.DataFrame(traj.groupby(constants.UID).apply(lambda x: _location_frequency_individual(x, normalize=normalize)))
    return df.rename(columns={constants.DATETIME: 'location_frequency'})


def _individual_mobility_network_individual(traj, self_loops=False):
    """
    Compute the individual mobility network of a single individual, given their TrajDataFrame
    :param traj: the trajectories of the individual
    :type TrajDataFrame
    :param self_loops: if True also considers the self loops
    :return: pandas DataFrame
    """
    loc2loc2weight = defaultdict(lambda: defaultdict(lambda: 0))
    traj = traj.sort_values(by=constants.DATETIME)
    lats_lngs = traj[[constants.LATITUDE, constants.LONGITUDE]].values

    i = 1
    for lat, lng in lats_lngs[1:]:
        prev = tuple(lats_lngs[i - 1])
        current = (lat, lng)
        if prev != current:
            loc2loc2weight[prev][current] += 1
        elif self_loops:
            loc2loc2weight[prev][current] += 1
        else:
            pass
        i += 1

    rows = []
    for loc1, loc2weight in loc2loc2weight.items():
        for loc2, weight in loc2weight.items():
            rows.append([loc1[0], loc1[1], loc2[0], loc2[1], weight])
    return pd.DataFrame(rows, columns=[constants.LATITUDE + '_origin', constants.LONGITUDE + '_origin',
                                       constants.LATITUDE + '_dest', constants.LONGITUDE + '_dest', 'n_trips'])


def individual_mobility_network(traj, self_loops=False, show_progress=True):
    """
    Compute the individual mobility network of an individual.

    :param traj: the trajectories of the individuals
    :type traj: pandas DataFrame
    :param boolean self_loops: if True adds self loops also
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the graph describing the individual mobility network
    :rtype: networkx Graph

    Examples:
            Computing the number of visits per location from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import individual_mobility_network
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data_test/brightkite_data.csv'data,  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> individual_mobility_network(tdf).head()
    uid
    324    ((43.79221124464029, 11.245646834881), (4.. seealso:: :func:`visits_per_location`3.404...
    333             ((43.5439336194691, 12.084056016463402))
    344    ((43.337466026311205, 11.337722717145901), (43...
    582    ((43.754596610925894, 11.270932719966302), (43...
    604    ((43.9025535031616, 11.074993570253), (43.8201...
    dtype: object

    References:
        .. [rinzivillo2014purpose] Rinzivillo, Salvatore, Gabrielli, Lorenzo, Nanni, Mirco, Pappalardo, Luca, Pedreschi, Dino and Giannotti, Fosca. "The purpose of motion: Learning activities from Individual Mobility Networks." Proceedings of the 2014 IEEE International Conference on Data Science and Advanced Analytics (DSAA).
        .. [bagrow2012mesoscopic] Bagrow, James P. and Lin, Yu-Ru. "Mesoscopic Structure and Social Aspects of Human Mobility." PLoS ONE 7 , no. 5 (2012): e37676.
        .. [song2010limits] Song, Chaoming, Qu, Zehui, Blumm, Nicholas and Barabási, Albert-László. "Limits of Predictability in Human Mobility." Science 327 , no. 5968 (2010): 1018-1021.
    """
    if show_progress:
        return traj.groupby(constants.UID).progress_apply(lambda x: _individual_mobility_network_individual(x,
                self_loops=self_loops)).reset_index().drop('level_1', axis=1)
    else:
        return traj.groupby(constants.UID).apply(lambda x: _individual_mobility_network_individual(x,
                self_loops=self_loops)).reset_index().drop('level_1', axis=1)


def _recency_rank_individual(traj):
    traj = traj.sort_values(constants.DATETIME, ascending=False).drop_duplicates(subset=[constants.LATITUDE,
                                                                                         constants.LONGITUDE],
                                                                                 keep="first")
    traj['recency_rank'] = range(1, len(traj) + 1)
    return traj[[constants.LATITUDE, constants.LONGITUDE, 'recency_rank']]


def recency_rank(traj, show_progress=True):
    """
    Compute the recency rank of the locations of the individual

    :param traj: the trajectories of the individuals
    :type traj: pandas DataFrame
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the recency rank for each location of the individuals
    :rtype: pandas DataFrame

    Examples:
            Computing the number of visits per location from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import recency_rank
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file('../data_test/brightkite_data.csv'data,  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> recency_rank(df).head()
                  lat        lng  recency_rank
    uid
    324 18  43.776687  11.235343             1
        17  43.792211  11.245647             2
        16  43.824934  11.282771             3
        15  43.797951  11.240277             4
        14  43.530801  10.319265             5

    .. seealso:: :func:`frequency_rank`

    References:
        .. [barbosa2015effect] Barbosa, Hugo, de Lima-Neto, Fernando B., Evsukoff, Alexandre, Menezes, Ronaldo."The effect of recency to human mobility", EPJ Data Science 4:21 (2015)
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _recency_rank_individual(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _recency_rank_individual(x))
    return pd.DataFrame(df)


def _frequency_rank_individual(traj):
    """
    Compute the frequency rank of the locations of a single individual, given their TrajDataFrame
    :param traj: trajectories of the individual
    :type: TrajDataFrame
    :return: TrajDataFrame
    """
    traj = traj.groupby([constants.LATITUDE, constants.LONGITUDE]).count().sort_values(by=constants.DATETIME, ascending=False).reset_index()
    traj['frequency_rank'] = range(1, len(traj) + 1)
    return traj[[constants.LATITUDE, constants.LONGITUDE, 'frequency_rank']]


def frequency_rank(traj, show_progress=True):
    """
    Compute the frequency rank of the locations of the individual

    :param traj: the trajectories of the individuals
    :type traj: pandas DataFrame
    
    :param show_progress: if True show a progress bar
    :type show_progress: boolean
    
    :return: the frequency rank for each location of the individuals
    :rtype: pandas DataFrame

    Examples:
            Computing the number of visits per location from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.individual import frequency_rank
    >>> from skmob import TrajDataFrame
    >>> tdf = TrajDataFrame.from_file(data, sep=',',  user_id='user', dadataheck-in time', latitude='latitude', longitude='longitude')
    >>> frequency_rank(df).head()
                lat        lng  frequency_rank
    uid
    324 0  42.970399  10.695334               1
        1  43.404161  10.873485               2
        2  43.530801  10.319265               3
        3  43.776687  11.235343               4
        4  43.797951  11.240277               5

    .. seealso:: :func:`recency_rank`

    References:
        .. [barbosa2015effect] Barbosa, Hugo, de Lima-Neto, Fernando B., Evsukoff, Alexandre, Menezes, Ronaldo."The effect of recency to human mobility", EPJ Data Science 4:21 (2015)
    """
    if show_progress:
        df = traj.groupby(constants.UID).progress_apply(lambda x: _frequency_rank_individual(x))
    else:
        df = traj.groupby(constants.UID).apply(lambda x: _frequency_rank_individual(x))
    return df

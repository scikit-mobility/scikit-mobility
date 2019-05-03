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


def uncorrelated_location_entropy(traj, normalize=False):
    """number of distinct locations visited by the
    The temporal-uncorrelated location entropy :math:`LE_{unc}(j)` of a location :math:`j` is the historical probability
    that an individual :math:`u` visited location :math:`j`.

    :param traj: the trajectories of the individuals
    :type traj: pandas DataFrame
    :param boolean normalize: if True normalize the entropy by dividing by log2(N), where N is the number of
        distinct users that visited a location

    :return: the temporal-uncorrelated location entropies of the individuals
    :rtype: pandas Series

    Examples:
        Computing the temporal-uncorrelated location entropy of each individual from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.collective import uncorrelated_location_entropy
    >>> tdf = TrajDataFrame.from_file('../data/brightkite_data.csv', sep=',',  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> uncorrelated_location_entropy(tdf, normalize=True).head()
    lat        lng
    42.333550  10.919689    0.000000
    42.342177  10.898593    0.638517
    42.351689  10.920185    0.506808
    42.353302  10.909137    0.597521
    42.357550  10.922648    0.476653
    dtype: float64

    .. seealso:: :func:`random_entropy`, :func:`real_entropy`, :func:`uncorrelated_entropy`

    References:
        .. [cho2011frienship] Eunjoon Cho, Seth A. Myers, and Jure Leskovec. "Friendship and mobility: user movement in location-based social networks." In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '11). 1082-1090, 2011.
    """
    df = pd.DataFrame(traj.groupby([constants.LATITUDE, constants.LONGITUDE]).progress_apply(lambda x: _uncorrelated_location_entropy_individual(x, normalize=normalize)))
    column_name = sys._getframe().f_code.co_name
    if normalize:
        column_name = 'norm_%s' % sys._getframe().f_code.co_name
    return df.reset_index().rename(columns={0: column_name})


def _square_displacement(traj, delta_t):
    r0 = traj.iloc[0]
    t = r0[constants.DATETIME] + delta_t
    rt = traj[traj[constants.DATETIME] <= t].iloc[-1]
    square_displacement = getDistanceByHaversine((r0.lat, r0.lng), (rt.lat, rt.lng)) ** 2
    return square_displacement


def mean_square_displacement(traj, days=0, hours=1, minutes=0):
    """
    Compute the mean square displacement across the individuals.

    :param traj: the trajectories of the individuals
    :type traj: pandas DataFrame
    :param int days: the days since the starting time
    :param int hours: the hours since the days since the starting time
    :param int minutes: the minutes sine the hours since the days since the starting time

    :return float: the mean square displacement

    Examples:
        Computing the mean square displacement from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.collective import mean_square_displacement
    >>> df = skmob.read_trajectories('data/gps_test_dataset.csv', sep=',', user_id='user', longitude='lon')
    >>> mean_square_displacement(df, hours=1).head()
    5.033659854643163

    References:
        .. [brockmann2006scaling] Brockmann, D., Hufnagel, L. and Geisel, T.. "The scaling laws of human travel." Nature 439 (2006): 462.
        .. [song2010modelling] Song, Chaoming, Koren, Tal, Wang, Pu and Barabasi, Albert-Laszlo. "Modelling the scaling properties of human mobility." Nature Physics 6 , no. 10 (2010): 818--823.
        .. [gonzalez2008understanding] Gonzalez, Marta C., Hidalgo, Cesar A. and Barabasi, Albert-Laszlo. "Understanding individual human mobility patterns." Nature 453, no. 7196 (2008): 779--782.
        .. [shin2008levy] R. Shin, S. Hong, K. Lee, S. Chong, "On the levy-walk nature of human mobility: Do humans walk like monkeys?", in: Proc. IEEE INFOCOM, 2008, pp. 924–932.
        .. [maruyana2003truncated] Y. Maruyama, J. Murakami, "Truncated levy walk of a nanocluster bound weakly to an atomically flat surface: Crossover from superdiffusion to normal diffusion", Physical Review B 67 (8) (2003) 085406
        .. [vazquez1999diffusion] A. Vazquez, O. Sotolongo-Costa, F. Brouers, "Diffusion regimes in levy flights with  trapping",  Physica  A:  Statistical  Mechanics  and  its  Applications  264  (3) (1999) 424–431.
    """
    delta_t = timedelta(days=days, hours=hours, minutes=minutes)
    return traj.groupby(constants.UID).progress_apply(lambda x: _square_displacement(x, delta_t)).mean()


def visits_per_location(trajs):
    """
    Compute the number of visits in each location

    :param traj: the trajectories of the individuals
    :type traj: pandas DataFrame

    :return: the number of visits per location
    :rtype: pandas Series

    See also
    --------
    locations_population

    Examples:
            Computing the number of visits per location from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.collective import visits_per_location
    >>> df = skmob.read_trajectories('data/gps_test_dataset.csv', sep=',', user_id='user', longitude='lon')
    >>> visits_per_location(df).head()
                        n_visits
    lat       lng
    43.717999 10.902612      5338
    42.785016 11.110376      4717
    42.934046 10.783248      4354
    43.847140 11.142547      4201
    42.930131 10.764460      4169

    .. seealso:: :func:`homes_per_location`

    References:
        .. [pappalardo2018data] Pappalardo, Luca and Simini, Filippo. "Data-driven generation of spatio-temporal routines in human mobility.." Data Min. Knowl. Discov. 32 , no. 3 (2018): 787-829.
    """
    return trajs.groupby([constants.LATITUDE,
                          constants.LONGITUDE]).count().sort_values(by=constants.UID,
                                                                    ascending=False)[[constants.UID]].reset_index().rename({constants.UID: 'n_visits'},
                                                                                                             axis=1)


def homes_per_location(traj, start_night='22:00', end_night='07:00'):
    """
    Compute the number of homes in each location. A "home" location is the location that an individual visits the most.

    :param traj: the trajectories of the individuals
    :type traj: TrajDataFrame
    :param str start_night: the starting hour for the night
    :param str end_night: the ending hour for the night

    :return: the number of homes per location
    :rtype: pandas Series

    Examples:
            Computing the number of visits per location from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.collective import homes_per_location
    >>> from skmob.measures.individual import radius_of_gyration
    >>> tdf = TrajDataFrame.from_file(datadata,  user_id='user', datetime='check-in time', latitude='latitude', longitude='longitude')
    >>> homes_per_location(tdf).head()

    .. seealso:: :func:`homes_per_location`, :func:`home_location`

    References:
        .. [1] Pappalardo, Luca, Rinzivillo, Salvatore, Simini, Filippo, "Human Mobility Modelling: exploration and preferential return meet the gravity model." Procedia Computer Science, Volume 83, 2016, Pages 934-939 http://dx.doi.org/10.1016/j.procs.2016.04.188.
    """
    return home_location(traj,
                         start_night=start_night,
                         end_night=end_night).groupby([constants.LATITUDE,
                                                       constants.LONGITUDE]).count().sort_values(constants.UID,
                                                                                                 ascending=False).reset_index().rename(
        columns={constants.UID: 'n_homes'})


def visits_per_time_unit(traj, time_unit='1h'):
    """
    Compute the number of visits per time unit made in the mobility dataset.

    :param traj: the trajectories of the individuals
    :type traj: pandas DataFrame
    :param str time_unit: the time unit to use for grouping the time slots (default: '1h', it creates slots of 1 hour; range: for full specification of available time units, see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    :return: the number of visits per time unit
    :rtype: pandas Series

    Examples:
            Computing the number of visits per location from a DataFrame of trajectories

    >>> import skmob
    >>> from skmob.measures.collective import visits_per_time_unit
    >>> df = skmob.read_trajectories('data_test/gps_test_dataset.csvdata', user_id='user', longitude='lon')
    >>> visits_per_time_unit(df).head()
    datetime
    2011-05-01 00:00:00    974
    2011-05-01 01:00:00    519
    2011-05-01 02:00:00    374
    2011-05-01 03:00:00    301
    2011-05-01 04:00:00    492
    Freq: H, Name: uid, dtype: int64

    References:
        .. [1] Pappalardo, Luca, Rinzivillo, Salvatore, Simini, Filippo, "Human Mobility Modelling: exploration and preferential return meet the gravity model." Procedia Computer Science, Volume 83, 2016, Pages 934-939 http://dx.doi.org/10.1016/j.procs.2016.04.188.
    """
    return pd.DataFrame(traj.set_index(pd.DatetimeIndex(traj[constants.DATETIME])).groupby(pd.Grouper(freq=time_unit)).count()[constants.UID]).rename(columns={constants.UID: 'n_visits'})


def origin_destination_matrix(traj, self_loops=False):
    """
    Compute an origin-destination matrix from the trajectories of the individuals.

    :param traj: the trajectories of the individuals
    :type traj: pandas DataFrame
    :param boolean directed: if True returns a directed network, otherwise an undirected network

    :return: the graph describing the origin destination matrix
    :rtype: networkx Graph

    Examples:
            Computing the number of visits per location from a DataFrame of trajectories

    >>> import skmob
    >>> import networkx as nx
    >>> from skmob.measures.collective import origin_destination_matrix
    >>> df = skmob.read_trajectories('data_test/gps_test_dataset.csvdata', user_id='user', longitude='lon')
    >>> G = origin_destination_matrix(df)
    >>> print(nx.info(G))
    Name:
    Type: Graph
    Number of nodes: 20318
    Number of edges: 640107
    Average degree:  63.0089

    References:
        .. [calabrese2011estimating] Calabrese, Francesco, Lorenzo, Giusy Di, Liu, Liang and Ratti, Carlo. "Estimating Origin-Destination Flows Using Mobile Phone Location Data." IEEE Pervasive Computing 10 , no. 4 (2011): 36-44.
        .. [bonnel2015passive] Patrick Bonnel, Etienne Hombourger, Ana-Maria Olteanu-Raimond, Zbigniew Smoreda. "Passive Mobile Phone Dataset to Construct Origin-destination Matrix: Potentials and Limitations." Transportation Research Procedia 11 (2015): 381-398.
        .. [ortuzar2011modeling] J. de Dios Ortuzar,  L. Willumsen,  "Modeling Transport", John Wiley and Sons Ltd, New York, 2011.
        .. [iqbal2014development] M. S. Iqbal, C. F. Choudhury, P. Wang, M. C. Gonzalez, "Development of origin-destination matrices using mobile phone call data", Transportation Research Part C: Emerging Technologies 40 (2014) 63–74.
        .. [white2002extracting] J. White, I. Wells, "Extracting origin destination information from mobile phone data", in: Road Transport Information and Control, 2002. Eleventh International Conference on (Conf. Publ. No. 486), 2002, pp. 30–34.
        .. [caceres2007deriving] N. Caceres, J. Wideberg, F. G. Benitez, "Deriving origin destination data from a mobile phone network", Intelligent Transport Systems, IET 1 (1) (2007) 15–26.
        .. [jiang2013review]  S. Jiang, G. A. Fiore, Y. Yang, J. Ferreira Jr, E. Frazzoli, M. C. Gonzalez, "A review  of  urban  computing  for  mobile  phone  traces:  current  methods,  challenges and opportunities", in:  Proceedings of the 2nd ACM SIGKDD International Workshop on Urban Computing, ACM, 2013, p. 2.
        .. [lenormand2014cross] M. Lenormand, M. Picornell, O. G. Cantu-Ros, A. Tugores, T. Louail, R. Herranz, M. Barthelemy, E. Frıas-Martınez, J. J. Ramasco, "Cross-Checking Different Sources of Mobility Information", PLoS ONE 9 (8) (2014) e105184.
        .. [alexander2015origin] L. Alexander, S. Jiang, M. Murga, M. C. Gonzalez, "Origin-destination trips by purpose and time of day inferred from mobile phone data", Transportation Research Part C: Emerging Technologies 58, Part B (2015) 240–250
        .. [toole2015path] J. L. Toole, S. Colak, B. Sturt, L. P. Alexander, A. Evsukoff, M. C. Gonzalez, "The  path  most  traveled:  Travel  demand  estimation  using  big  data  resources", Transportation Research Part C: Emerging Technologies 58, Part B (2015) 162–177.
    """
    loc2loc2weight = defaultdict(lambda: defaultdict(lambda: 0))

    def _update_od_matrix(traj):
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

    traj.sort_values(by=constants.DATETIME).groupby(constants.UID).progress_apply(lambda x: _update_od_matrix(x))
    rows = []
    for loc1, loc2weight in loc2loc2weight.items():
        for loc2, weight in loc2weight.items():
            rows.append([loc1[0], loc1[1], loc2[0], loc2[1], weight])
    return pd.DataFrame(rows, columns=[constants.LATITUDE + '_origin', constants.LONGITUDE + '_origin',
                                       constants.LATITUDE + '_dest', constants.LONGITUDE + '_dest', 'n_trips'])


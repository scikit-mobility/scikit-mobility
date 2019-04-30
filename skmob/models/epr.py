import powerlaw
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
import math
from math import sqrt, sin, cos, pi, asin, pow, ceil, log
import csv
from tqdm import tqdm
from ..utils import constants
from scipy.sparse import lil_matrix
import random
import logging
import inspect
from ..core.trajectorydataframe import TrajDataFrame

latitude = constants.LATITUDE
longitude = constants.LONGITUDE
date_time = constants.DATETIME
user_id = constants.UID


def earth_distance(lat_lng1, lat_lng2):
    """
    Compute the distance (in km) along earth between two lat/lon pairs
    :param lat_lng1: tuple
        the first lat/lon pair
    :param lat_lng2: tuple
        the second lat/lon pair

    :return: float
        the distance along earth in km
    """
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...

def _earth_distance_vectorized(lat_lng1, lat_lng2):

    s_lat, s_lng = lat_lng1
    e_lat, e_lng = lat_lng2
    # approximate radius of earth in km
    R = 6373.0

    s_lat = s_lat*np.pi/180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2
    return 2 * R * np.arcsin(np.sqrt(d))


def compute_od_matrix(spatial_tessellation, use_relevance=True):
    """
    Compute a weighted origin destination matrix where an element A_{ij} is the
    probability p_{ij} of moving between two locations in the spatial tessellation
    given as input.

    Parameters
    ----------
    spatial_tessellation: dict
        a dictionary of location identifiers to a dictionary containing latitude,
        longitude and density of the location

    use_relevance: boolean
        if True, then use the relevance of locations, otherwise don't (default: True)

    Returns
    -------
    od_matrix: numpy array
        a bidimensional numpy array describing the weighted origin destination matrix
    """
    n = len(spatial_tessellation)
    od_matrix = np.zeros((n, n))
    for id_i in tqdm(spatial_tessellation):
        lat_i, lon_i = spatial_tessellation[id_i][latitude], spatial_tessellation[id_i][longitude]
        d_i = spatial_tessellation[id_i]['relevance']
        for id_j in spatial_tessellation:
            if id_j != id_i:
                lat_j, lon_j = spatial_tessellation[id_j][latitude], spatial_tessellation[id_j][longitude]
                if use_relevance:
                    d_j = spatial_tessellation[id_j]['relevance']
                    p_ij = (d_i * d_j) / (earth_distance((lat_i, lon_i), (lat_j, lon_j)) ** 2)
                else:
                    p_ij = 1.0 / (earth_distance((lat_i, lon_i), (lat_j, lon_j)) ** 2)
                od_matrix[id_i, id_j] = p_ij

        # normalization by row
        sum_odm = np.sum(od_matrix[id_i])
        if sum_odm > 0.0:
            od_matrix[id_i] /= sum_odm
    return od_matrix


def load_spatial_tessellation(filename='location2info_trentino', delimiter=','):
    """
    Load into a dictionary the locations and corresponding information (latitude, longitude, relevance)

    Parameters
    ----------
    filename: str
        the filename where the location info is stored

    Returns
    -------
    dict
        the dictionary of locations
    """
    spatial_tessellation = {}
    f = csv.reader(open(filename), delimiter=delimiter)
    f.__next__()  # delete header
    i = 0
    for line in tqdm(f):  # tqdm print a progress bar
        relevance = int(line[2])
        if relevance > 0:  # eliminate locations with zero relevance
            spatial_tessellation[i] = {latitude: float(line[0]),
                                       longitude: float(line[1]),
                                       'relevance': relevance}
            i += 1
    return spatial_tessellation



class SpatialEPR:
    """
    The sEPR model of individual human mobility

    :param name: str
        the name of the instantiation of the sEPR model (default: "Spatial EPR model")

    :param rho: float
        in the formula :math:`\rho S^{-\gamma}`, where :math:`S` is the number of distinct locations
        previously visited by the agent, the parameter :math:`\rho` (:math:`0 < \rho \leq 1`) controls
        the agent's tendency to explore a new location during the next move versus
        returning to a previously visited location (default: :math:`\rho = 0.6`, value estimated from empirical data)

    :param gamma: float
        in the formula :math:`Density\rho S^{-\gamma}`, where :math:`S` is the number of distinct locations
        previously visited by the agent, the parameter :math:`\gamma` (:math:`\gamma \geq 0`) controls
        the agent's tendency to explore a new location during the next move versus
        returning to a previously visited location (default: 0.21, value estimated from empirical data)

    :param beta: float
        the parameter :math:`\beta` of the waiting time distribution (default: :math:`\beta = 0.8`, value estimated from empirical data)

    :param tau: int
        the parameter :math:`\tau` of the waiting time distribution (default: :math:`\tau = 17`, expressed in hours, value estimated from empirical data)

    :param min_wait_time_minutes: int
        minimum waiting time in minutes

    :ivar: name: str
        the name of the instantiation of the model

    :ivar: trajectory_: pandas DataFrame
        the trajectory generated by the model, describing the trajectory of the agents

    :ivar: spatial_tessellation: dict
        the spatial tessellation used during the simulation

    :ivar rho: float
        in the formula :math:`\rho S^{-\gamma}`, where :math:`S` is the number of distinct locations
        previously visited by the agent, the parameter :math:`\rho` (:math:`0 < \rho \leq 1`) controls
        the agent's tendency to explore a new location during the next move versus
        returning to a previously visited location (default: :math:`\rho = 0.6`, value estimated from empirical data)

    :ivar gamma: float
        in the formula :math:`\rho S^{-\gamma}`, where :math:`S` is the number of distinct locations
        previously visited by the agent, the parameter :math:`\gamma` (:math:`\gamma \geq 0`) controls
        the agent's tendency to explore a new location during the next move versus
        returning to a previously visited location (default: 0.21, value estimated from empirical data)

    :ivar beta: float
        the parameter :math:`\beta` of the waiting time distribution (default: :math:`\beta = 0.8`, value estimated from empirical data)

    :ivar tau: int
        the parameter :math:`\tau` of the waiting time distribution (default: :math:`\tau = 17`, expressed in hours, value estimated from empirical data)

    :ivar min_wait_time_minutes: int
        minimum waiting time in minutes

    Examples:

    >>> from skmob.models.epr import SpatialEPR, load_spatial_tessellation, compute_od_matrix
    >>> import datetime
    >>> spatial_tessellation = load_spatial_tessellation('../datasets/location2info_trentino')
    >>> od_matrix = compute_od_matrix(spatial_tessellation, use_relevance=False)
    >>> sepr = SpatialEPR()
    >>> start_date = datetime.datetime.strptime('04-01-2018 08:00:00', '%m-%d-%Y %H:%M:%S')
    >>> end_date = start_date + datetime.timedelta(days=14)
    >>> sepr.generate(start_date, end_date, spatial_tessellation, od_matrix=od_matrix)
    >>> sepr.trajectories_.head()
       uid                   datetime        lat        lng
    0    1 2018-04-01 08:00:00.000000  45.988849  11.601780
    1    1 2018-04-01 12:49:50.225764  46.006240  11.628428
    2    1 2018-04-01 15:41:24.644204  46.005942  11.641331
    3    1 2018-04-01 16:03:36.448021  45.996652  11.653802
    4    1 2018-04-01 20:42:35.457872  46.006240  11.628428

    .. seealso:: :class:`EPR`

    References:
        .. [pappalardo2015returners] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F., Barabasi, A.-L. "Returners and Explorers dichotomy in human mobility.", Nature Communications, 6:8166, doi: 10.1038/ncomms9166 (2015).
        .. [pappalardo2016modelling] Pappalardo, L., Simini, F. Rinzivillo, S., "Human Mobility Modelling: exploration and preferential return meet the gravity model", Procedia Computer Science 83, doi: 10.1016/j.procs.2016.04.188 (2016).
    """

    def __init__(self, name='Spatial EPR model',
                 rho=0.6, gamma=0.21,
                 beta=0.8, tau=17,
                 min_wait_time_minutes=10):

        self._name = name
        
        self._rho = rho
        self._gamma = gamma
        self._tau = tau
        self._beta = beta

        self._location2visits = defaultdict(int)
        self._od_matrix = None
        self._is_sparse = True
        self._spatial_tessellation = None
        self._starting_loc = None

        # Minimum waiting time (in hours)
        self._min_wait_time = min_wait_time_minutes / 60.0  # minimum waiting time
        self._time_generator = powerlaw.Truncated_Power_Law(xmin=self._min_wait_time,
                                                            parameters=[1. + self._beta, 1.0 / self._tau])

        self._trajectories_ = []
        self._log_file = None

    @property
    def name(self):
        return self._name

    @property
    def rho(self):
        return self._rho

    @property
    def gamma(self):
        return self._gamma

    @property
    def tau(self):
        return self._tau

    @property
    def beta(self):
        return self._beta

    @property
    def min_wait_time(self):
        return self._min_wait_time

    @property
    def spatial_tessellation_(self):
        return self._spatial_tessellation


    def _weighted_random_selection(self):
        """
        Select a random location given their visitation frequency. Used by the return mechanism.

        :return: int
            a random location
        """
        locations = np.fromiter(self._location2visits.keys(), dtype=int)
        weights = np.fromiter(self._location2visits.values(), dtype=float)
        weights = weights / np.sum(weights)
        location = np.random.choice(locations, size=1, p=weights)
        return int(location[0])

    def _preferential_return(self):
        """
        Choose the location the agent returns to, according to the visitation frequency
        of the previously visited locations.

        :return: int
            the identifier of the next location
        """
        next_location = self._weighted_random_selection()
        if self._log_file is not None:
            logging.info('RETURN to %s (%s, %s)' % (next_location,
                                                        self._spatial_tessellation[next_location][latitude],
                                                        self._spatial_tessellation[next_location][longitude]))
            logging.info('\t frequency = %s' % self._location2visits[next_location])
        return next_location

    def _exploration(self, current_location):
        """
        Choose the new location the agent explores, according to the probabilities of visiting a location
        on the spatial tessellation given the agent's current location.

        :param current_location: int
            the identifier of the agent's current location

        :return: int
            the identifier of the new location to explore
        """
        if self._is_sparse:  # t_2413if the od matrix is not precomputed

            prob_array = self._od_matrix.getrowview(current_location)
            if prob_array.nnz == 0:
                # if the row has been not populated
                self._populate_od_matrix(current_location)
            locations = np.arange(len(self._spatial_tessellation))
            weights = prob_array.toarray()[0]
            location = np.random.choice(locations, size=1, p=weights)[0]

        else:  # if the matrix is precomputed

            locations = np.arange(len(self._od_matrix[current_location]))
            weights = self._od_matrix[current_location]
            location = np.random.choice(locations, size=1, p=weights)[0]
            # update the od matrix such that the location is not explorable anymore

        if self._log_file is not None:
            logging.info('EXPLORATION to %s (%s, %s)' % (location,
                                                         self._spatial_tessellation[location][latitude],
                                                         self._spatial_tessellation[location][longitude]))
        return location

    def _populate_od_matrix(self, location):
        """
        Populate the origin-destination (od) matrix with the probability to move from the location in input
        to all other locations in the spatial tessellation. Used when the precomputed matrix is NOT specified in
        input.

        :param location: int
            the identifier of a location on the spatial tessellation
        """
        # update the od matrix
        lat_i, lon_i = self._spatial_tessellation[location][latitude], self._spatial_tessellation[location][longitude]
        probs = []
        for id_j in self._spatial_tessellation:
            if id_j != location:
                lat_j, lon_j = self._spatial_tessellation[id_j][latitude], self._spatial_tessellation[id_j][longitude]
                p_ij = 1.0 / (earth_distance((lat_i, lon_i), (lat_j, lon_j)) ** 2)
                probs.append(p_ij)
            else:
                probs.append(0.0)

        # normalization by row
        sum_odm = sum(probs)
        if sum_odm > 0.0:
            self._od_matrix[location, :] = np.array(probs) / sum_odm

    def _get_trajdataframe(self, parameters):
        """
        Transform the trajectories list into a pandas DataFrame.

        :return: a pandas DataFrame describing the trajectories
        :rtype pandas DataFrame
        """
        df = pd.DataFrame(self._trajectories_, columns=[user_id, date_time, 'location'])
        df[[latitude, longitude]] = df.location.apply(lambda s: pd.Series({latitude: self._spatial_tessellation[s][latitude],
                                                                    longitude: self._spatial_tessellation[s][longitude]}))
        df = df.sort_values(by=[user_id, date_time]).drop('location', axis=1)
        return TrajDataFrame(df, parameters=parameters)

    def predict(self, *args):
        ## TODO
        traj, n_steps = args
        locs_info = traj.groupby([latitude, longitude]).count().reset_index().reset_index().drop('uid', axis=1)
        locs_info.rename({'index': 'loc_id', datetime: 'n_visits'}, axis=1, inplace=True)
        traj = pd.merge(traj, locs_info, on=[latitude, longitude])

        n_visited_locations = len(locs_info)

        # choose a probability to return o explore
        p_new = random.uniform(0, 1)

        if p_new <= self._rho * math.pow(n_visited_locations, -self._gamma):  # choose to return or explore
            # EXPLORATION
            agent_id, current_time, current_location = traj[[user_id, datetime, 'loc_id']].iloc[-1]  # the last visited location
            next_location = self._exploration(current_location)
            return next_location

    def _choose_location(self):
        """
        Choose the next location to visit given the agent's current location.

        :return: int
            the identifier of the next location to visit
        """
        # initialize variables
        n_visited_locations = len(self._location2visits)  # number of already visited locations

        if n_visited_locations == 0:
            self._starting_loc = self._exploration(self._starting_loc)
            return self._starting_loc

        # choose a probability to return o explore
        p_new = random.uniform(0, 1)

        # choose to return or explore
        if p_new <= self._rho * math.pow(n_visited_locations, -self._gamma) and n_visited_locations != self._od_matrix.shape[0]:
            # EXPLORATION
            agent_id, current_time, current_location = self._trajectories_[-1]  # the last visited location
            next_location = self._exploration(current_location)
            while next_location in self._location2visits:
                next_location = self._exploration(current_location)
            return next_location
        else:
            # PREFERENTIAL RETURN
            next_location = self._preferential_return()
            return next_location

    def _choose_waiting_time(self):
        """
        Choose the time (in hours) the agent has to wait before the next move.

        :return: float
            the time to wait (in hours) before the next movement.
        """
        time_to_wait = self._time_generator.generate_random()[0]
        return time_to_wait

    def generate(self, start_date, end_date, spatial_tessellation, n_agents=1, starting_location=None, od_matrix=None,
                 random_state=None, log_file=None, verbose=False):
        """
        Start the simulation for the agents, with a duration determined by "start_date" and "end_date".

        :param start_date: datetime
            the starting date of the simulation

        :param end_date: datetime
            the ending date of the simulation

        :param spatial_tessellation: dict
            the spatial tessellation, a dictionary of locations to info (lat, lng, relevance)

        :param n_agents: int
            the number of agents to generate

        :param starting_location
            the identifier of the starting location for the simulation (as specified in the spatial tessellation)
        :type starting_location: int or None

        :param od_matrix: the od_matrix to use for deciding the movements. If None, it is computed "on the fly" during the simulation
        :type od_matrix: numpy array or None

        :param random_state: if int, random_state is the seed used by the random number generator; if None, the random number generator is the RandomState instance used by np.random and random.random (default: None)
        :type random_state: int or None
        """
        # Save function arguments and values in a dictionary
        frame = inspect.currentframe()
        args, _, _, arg_values = inspect.getargvalues(frame)
        parameters = dict([])
        parameters['model'] = {'class': self.__class__.__init__,
                               'generate': {i: arg_values[i] for i in args[1:] if i not in ['spatial_tessellation', 
                                                                                           'od_matrix', 'log_file', 'verbose']}}

        # if specified, fix the random seeds to guarantee reproducibility of simulation
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        if log_file is not None:
            self._log_file = log_file
            logging.basicConfig(format='%(message)s', filename=log_file, filemode='w', level=logging.INFO)

        # initialization of trajectories
        self._trajectories_ = []

        # setting of spatial tessellation
        self._spatial_tessellation = spatial_tessellation

        # initialization of od matrix
        if od_matrix is None:
            n = len(self._spatial_tessellation)
            self._od_matrix = lil_matrix(np.zeros((n, n)))
            self._is_sparse = True
        else:
            self._od_matrix = od_matrix
            self._is_sparse = False

        # for each agent
        loop = range(1, n_agents + 1)
        if verbose:
            loop = tqdm(range(1, n_agents + 1))
        
        for agent_id in loop:  # tqdm print a progress bar

            self._location2visits = defaultdict(int)
            self._spatial_tessellation = spatial_tessellation
            if starting_location is None:
                self._starting_loc = np.random.choice(np.fromiter(self._spatial_tessellation.keys(), dtype=int), size=1)[0]
            else:
                self._starting_loc = starting_location

            current_date = start_date
            self._trajectories_.append((agent_id, current_date, self._starting_loc))
            self._location2visits[self._starting_loc] += 1

            waiting_time = self._choose_waiting_time()
            current_date += datetime.timedelta(hours=waiting_time)

            # until the time period expires
            while current_date < end_date:
                next_location = self._choose_location()
                self._trajectories_.append((agent_id, current_date, next_location))
                self._location2visits[next_location] += 1

                waiting_time = self._choose_waiting_time()
                current_date += datetime.timedelta(hours=waiting_time)

        tdf = self._get_trajdataframe(parameters)
        return tdf


class DensityEPR:
    """
    The dEPR model if individual human mobility

    :param name: str
        the name of the instantiation of the dEPR model (default: "Density EPR model")

    :param rho: float
        in the formula :math:`\rho S^{-\gamma}`, where :math:`S` is the number of distinct locations
        previously visited by the agent, the parameter :math:`\rho` (:math:`0 < \rho \leq 1`) controls
        the agent's tendency to explore a new location during the next move versus
        returning to a previously visited location (default: :math:`\rho = 0.6`, value estimated from empirical data)

    :param gamma: float
        in the formula :math:`\rho S^{-\gamma}`, where :math:`S` is the number of distinct locations
        previously visited by the agent, the parameter :math:`\gamma` (:math:`\gamma \geq 0`) controls
        the agent's tendency to explore a new location during the next move versus
        returning to a previously visited location (default: 0.21, value estimated from empirical data)

    :param beta: float
        the parameter :math:`\beta` of the waiting time distribution (default: :math:`\beta = 0.8`, value estimated from empirical data)

    :param tau: int
        the parameter :math:`\tau` of the waiting time distribution (default: :math:`\tau = 17`, expressed in hours, value estimated from empirical data)

    :param min_wait_time_minutes: int
        minimum waiting time in minutes

    :ivar: name: str
        the name of the instantiation of the model

    :ivar: trajectory_: pandas DataFrame
        the trajectory generated by the model, describing the trajectory of the agents

    :ivar: spatial_tessellation: dict
        the spatial tessellation used during the simulation

    :ivar rho: float
        in the formula :math:`\rho S^{-\gamma}`, where :math:`S` is the number of distinct locations
        previously visited by the agent, the parameter :math:`\rho` (:math:`0 < \rho \leq 1`) controls
        the agent's tendency to explore a new location during the next move versus
        returning to a previously visited location (default: :math:`\rho = 0.6`, value estimated from empirical data)

    :ivar gamma: float
        in the formula :math:`\rho S^{-\gamma}`, where :math:`S` is the number of distinct locations
        previously visited by the agent, the parameter :math:`\gamma` (:math:`\gamma \geq 0`) controls
        the agent's tendency to explore a new location during the next move versus
        returning to a previously visited location (default: 0.21, value estimated from empirical data)

    :ivar beta: float
        the parameter :math:`\beta` of the waiting time distribution (default: :math:`\beta = 0.8`, value estimated from empirical data)

    :ivar tau: int
        the parameter :math:`\tau` of the waiting time distribution (default: :math:`\tau = 17`, expressed in hours, value estimated from empirical data)

    :ivar min_wait_time_minutes: int
        minimum waiting time in minutes

    Examples:

    >>> from skmob.models.epr import DensityEPR, load_spatial_tessellation, compute_od_matrix
    >>> import datetime
    >>> spatial_tessellation = load_spatial_tessellation('../datasets/location2info_trentino')
    >>> od_matrix = compute_od_matrix(spatial_tessellation, use_relevance=True)
    >>> depr = DensityEPR()
    >>> start_date = datetime.datetime.strptime('04-01-2018 08:00:00', '%m-%d-%Y %H:%M:%S')
    >>> end_date = start_date + datetime.timedelta(days=14)
    >>> depr.generate(start_date, end_date, spatial_tessellation, od_matrix=od_matrix)
    >>> depr.trajectory_.head()
       uid                   datetime        lat        lng
    0    1 2018-04-01 08:00:00.000000  46.256276  11.303006
    1    1 2018-04-01 08:17:12.177740  46.256276  11.303006
    2    1 2018-04-01 08:40:16.715161  46.142144  11.155812
    3    1 2018-04-01 08:54:49.254859  46.142144  11.155812
    4    1 2018-04-01 09:20:16.812744  46.080622  11.075811

    .. seealso:: :class:`EPR`

    References:
        .. [pappalardo2015returners] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F., Barabasi, A.-L. "Returners and Explorers dichotomy in human mobility.", Nature Communications, 6:8166, doi: 10.1038/ncomms9166 (2015).
        .. [pappalardo2016modelling] Pappalardo, L., Simini, F. Rinzivillo, S., "Human Mobility Modelling: exploration and preferential return meet the gravity model", Procedia Computer Science 83, doi: 10.1016/j.procs.2016.04.188 (2016).
    """

    def __init__(self, name='Density EPR model', rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=10):

        self._name = name
        
        self._rho = rho
        self._gamma = gamma
        self._tau = tau
        self._beta = beta

        self._location2visits = defaultdict(int)
        self._od_matrix = None
        self._is_sparse = True
        self._spatial_tessellation = None
        self._starting_loc = None

        # Minimum waiting time (in hours)
        self._min_wait_time = min_wait_time_minutes / 60.0  # minimum waiting time
        self._time_generator = powerlaw.Truncated_Power_Law(xmin=self._min_wait_time,
                                                            parameters=[1. + self._beta, 1.0 / self._tau])

        self._trajectories_ = []
        self._log_file = None

    @property
    def name(self):
        return self._name

    @property
    def rho(self):
        return self._rho

    @property
    def gamma(self):
        return self._gamma

    @property
    def tau(self):
        return self._tau

    @property
    def beta(self):
        return self._beta

    @property
    def min_wait_time(self):
        return self._min_wait_time

    @property
    def spatial_tessellation_(self):
        return self._spatial_tessellation

    @property
    def trajectories_(self):
        return self._trajectories_

    def _weighted_random_selection(self):
        """
        Select a random location given their visitation frequency. Used by the return mechanism.

        :return: int
            a random location
        """
        locations = np.fromiter(self._location2visits.keys(), dtype=int)
        weights = np.fromiter(self._location2visits.values(), dtype=float)
        weights = weights / np.sum(weights)
        location = np.random.choice(locations, size=1, p=weights)
        return int(location[0])

    def _preferential_return(self):
        """
        Choose the location the agent returns to, according to the visitation frequency
        of the previously visited locations.

        :return: int
            the identifier of the next location
        """
        next_location = self._weighted_random_selection()
        if self._log_file is not None:
            logging.info('RETURN to %s (%s, %s)' % (next_location,
                                                        self._spatial_tessellation[next_location][latitude],
                                                        self._spatial_tessellation[next_location][longitude]))
            logging.info('\t frequency = %s' % self._location2visits[next_location])
        return next_location

    def _preferential_exploration(self, current_location):
        """
        Choose the new location the agent explores, according to the probabilities
        in the od matrix.

        :param current_location : int
            the identifier of the current location of the individual

        :return: int
            the identifier of the new location to explore
        """

        if self._is_sparse:
            prob_array = self._od_matrix.getrowview(current_location)
            if prob_array.nnz == 0:
                # if the row has been not populated
                self._populate_od_matrix(current_location)
            locations = np.arange(len(self._spatial_tessellation))
            weights = prob_array.toarray()[0]
            location = np.random.choice(locations, size=1, p=weights)[0]

        else:  # if the matrix is precomputed
            locations = np.arange(len(self._od_matrix[current_location]))
            weights = self._od_matrix[current_location]
            location = np.random.choice(locations, size=1, p=weights)[0]

        if self._log_file is not None:
            logging.info('EXPLORATION to %s (%s, %s)' % (location,
                                                         self._spatial_tessellation[location][latitude],
                                                         self._spatial_tessellation[location][longitude]))

        return location

    def _populate_od_matrix(self, location):
        """
        Populate the od matrix with the probability to move from the location in input to all other locations
        in the spatial tessellation

        :param location: int
            the identifier of a location
        """
        lat_i, lon_i = self._spatial_tessellation[location][latitude], self._spatial_tessellation[location][longitude]
        d_i = self._spatial_tessellation[location]['relevance']

        probs = []
        for id_j in self._spatial_tessellation:
            if id_j != location:
                lat_j, lon_j = self._spatial_tessellation[id_j][latitude], self._spatial_tessellation[id_j][longitude]
                d_j = self._spatial_tessellation[id_j]['relevance']
                p_ij = (d_i * d_j) / (earth_distance((lat_i, lon_i), (lat_j, lon_j)) ** 2)
                probs.append(p_ij)
            else:
                probs.append(0.0)

        # normalization by row
        sum_odm = sum(probs)
        if sum_odm > 0.0:
            self._od_matrix[location, :] = np.array(probs) / sum_odm

    def _get_trajdataframe(self, parameters):
        """
        Transform the trajectories list into a pandas DataFrame.

        :return: a pandas DataFrame describing the trajectories
        :rtype pandas DataFrame
        """
        df = pd.DataFrame(self._trajectories_, columns=[user_id, date_time, 'location'])
        df[[latitude, longitude]] = df.location.apply(lambda s: pd.Series({latitude: self._spatial_tessellation[s][latitude],
                                                                    longitude: self._spatial_tessellation[s][longitude]}))
        df = df.sort_values(by=[user_id, date_time]).drop('location', axis=1)
        return TrajDataFrame(df, parameters=parameters)

    def _choose_location(self):
        """
        Choose the next location to visit given the agent's current location.

        :return: int
            the identifier of the next location to visit
        """
        n_visited_locations = len(self._location2visits)  # number of already visited locations

        if n_visited_locations == 0:
            self._starting_loc = self._preferential_exploration(self._starting_loc)
            return self._starting_loc

        # choose a probability to return or explore
        p_new = random.uniform(0, 1)

        if p_new <= self._rho * math.pow(n_visited_locations, -self._gamma) and n_visited_locations != self._od_matrix.shape[0]:  # choose to return or explore
            # PREFERENTIAL EXPLORATION
            agent_id, current_time, current_location = self._trajectories_[-1]  # the last visited location
            next_location = self._preferential_exploration(current_location)
            while next_location in self._location2visits:
                next_location = self._preferential_exploration(current_location)
            return next_location

        else:
            # PREFERENTIAL RETURN
            next_location = self._preferential_return()
            return next_location

    def _choose_waiting_time(self):
        """
        Choose the time (in hours) the agent has to wait before the next movement.

        :return: float
            the time to wait before the next movement.
        """
        time_to_wait = self._time_generator.generate_random()[0]
        return time_to_wait

    def generate(self, start_date, end_date, spatial_tessellation, n_agents=1, starting_location=None, od_matrix=None,
                 random_state=None, log_file=None, verbose=False):
        """
        Start the simulation of the agent at time "start_date" till time "end_date".

        :param start_date : datetime
            the starting date of the simulation

        :param end_date : datetime
            the ending date of the simulation

        :param spatial_tessellation : dict
            the spatial tessellation, a dictionary of location to info (lat, lng, relevance)

        :param n_agents: int
            the number of agents to generate

        :param starting_location
            the identifier of the starting location for the simulation (as specified in the spatial tessellation)
        :type starting_location: int or None

        :param od_matrix: the od_matrix to use for deciding the movements. If None, it is computed "on the fly" during the simulation
        :type od_matrix: numpy array or None

        :param random_state: if int, random_state is the seed used by the random number generator; if None, the random number generator is the RandomState instance used by np.random and random.random (default: None)
        :type random_state: int or None
        """
        
        # Save function arguments and values in a dictionary
        frame = inspect.currentframe()
        args, _, _, arg_values = inspect.getargvalues(frame)
        parameters = dict([])
        parameters['model'] = {'class': self.__class__.__init__,
                               'generate': {i: arg_values[i] for i in args[1:] if i not in ['spatial_tessellation', 
                                                                                           'od_matrix', 'log_file']}}
        
        # if specified, fix the random seeds to guarantee reproducibility of simulation
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        if log_file is not None:
            self._log_file = log_file
            logging.basicConfig(format='%(message)s', filename=log_file, filemode='w', level=logging.INFO)

        # initialization of trajectories
        self._trajectories_ = []

        # setting of spatial tessellation
        self._spatial_tessellation = spatial_tessellation

        # initialization of od matrix
        if od_matrix is None:
            n = len(self._spatial_tessellation)
            self._od_matrix = lil_matrix((n, n))
            self._is_sparse = True
        else:
            self._od_matrix = od_matrix
            self._is_sparse = False

        # for each agent
        loop = range(1, n_agents + 1)
        if verbose:
            loop = tqdm(range(1, n_agents + 1))
            
        for agent_id in loop:
            self._location2visits = defaultdict(int)
            if starting_location is None:
                self._starting_loc = np.random.choice(np.fromiter(self._spatial_tessellation.keys(), dtype=int), size=1)[0]
            else:
                self._starting_loc = starting_location

            current_date = start_date
            self._trajectories_.append((agent_id, current_date, self._starting_loc))
            self._location2visits[self._starting_loc] += 1

            waiting_time = self._choose_waiting_time()
            current_date += datetime.timedelta(hours=waiting_time)

            while current_date < end_date:
                next_location = self._choose_location()
                self._trajectories_.append((agent_id, current_date, next_location))
                self._location2visits[next_location] += 1

                waiting_time = self._choose_waiting_time()
                current_date += datetime.timedelta(hours=waiting_time)

        tdf = self._get_trajdataframe(parameters)
        return tdf


class Ditras:
    """
    The DITRAS (DIary-based TRAjectory Simulator) model of individual human mobility

    :param name : str
        the name of the instantiation of the rEPR model (default: "recency EPR model")

    :param rho: float
        in the formula $\rho S^{-\gamma}$, where $S$ is the number of distinct locations
        previously visited by the agent, the rho parameter (0 < rho <= 1) controls
        the agent's tendency to explore a new location during the next move versus returning to a previously visited location.
        (default: 0.6 (value estimated from empirical data)

    :param gamma: float
        in the formula $\rho S^{-\gamma}$, where $S$ is the number of distinct locations
        previously visited by the agent, the gamma parameter (gamma >= 0) controls
        the agent's tendency to explore a new location during the next move versus returning to a previously visited location.
        (default: 0.21, value estimated from empirical data by)

    :ivar: name: str
        the name of the instantiation of the model

    :ivar: trajectory_: pandas DataFrame
        the trajectory generated by the model, describing the trajectory of the agents

    :ivar: spatial_tessellation: dict
        the spatial tessellation used during the simulation

    :ivar rho: float
        in the formula $\rho S^{-\gamma}$, where $S$ is the number of distinct locations
        previously visited by the agent, the rho parameter (0 < rho <= 1) controls
        the agent's tendency to explore a new location during the next move versus returning to a previously visited location.
        (default: 0.6 (value estimated from empirical data)

    :ivar gamma: float
        in the formula $\rho S^{-\gamma}$, where $S$ is the number of distinct locations
        previously visited by the agent, the gamma parameter (gamma >= 0) controls
        the agent's tendency to explore a new location during the next move versus returning to a previously visited location.
        (default: 0.21, value estimated from empirical data by)

    Examples:

    >>> from skmob.models.epr import Ditras, load_spatial_tessellation, compute_od_matrix
    >>> from skmob.models.markov_diary_generator import MarkovDiaryGenerator
    >>> import datetime
    >>> spatial_tessellation = load_spatial_tessellation('../datasets/location2info_tuscany')
    >>> od_matrix = compute_od_matrix(spatial_tessellation, filename='od_matrix_tuscany.csv')
    >>> start_date = datetime.datetime(2011, 5, 1, 0)
    >>> end_date = start_date + datetime.timedelta(days=14)
    >>> traj = read_trajectories('../../../data_test/gps_test_dataset.csv', latitude='lat', longitude='lon', user_id='user', dates='datetime', sep=',')
    >>> mdg = MarkovDiaryGenerator()
    >>> mdg.fit(traj, 1000, start_date, end_date )
    >>> ditras = Ditras(mdg)
    >>> ditras.generate(start_date, end_date, spatial_tessellation, od_matrix=od_matrix)
    >>> ditras.trajectory_.head()
        uid            datetime        lat        lng
    0    1 2011-05-01 00:00:00  46.109215  10.986303
    1    1 2011-05-01 06:00:00  46.109215  10.986303
    2    1 2011-05-01 10:00:00  46.073010  10.997935
    3    1 2011-05-01 14:00:00  46.073010  10.997935
    4    1 2011-05-01 15:00:00  46.079187  11.153357

    References:

    .. [pappalardo2018data] Pappalardo, L, Simini, F, Data-driven generation of spatio-temporal routines in human mobility, Data Mining and Knowledge Discovery, 32:3 (2018).
    """

    def __init__(self, diary_generator, rho=0.6, gamma=0.21, name='Ditras model'):
        self._diary_generator = diary_generator
        self._rho = rho
        self._gamma = gamma
        self._name = name

        self._location2visits = defaultdict(int)
        self._od_matrix = None
        self._is_sparse = True
        self._spatial_tessellation = None
        self._starting_loc = None

        self._trajectories_ = []

    @property
    def spatial_tessellation_(self):
        return self._spatial_tessellation

    @property
    def trajectories_(self):
        return self._trajectories_

    @property
    def name(self):
        return self._name

    @property
    def rho(self):
        return self._rho

    @property
    def gamma(self):
        return self._gamma

    def _weighted_random_selection(self):
        """
        Select a random location given their visitation frequency. Used by the return mechanism.

        :param type: str
            the type of random selection

        :return: int
            a random location
        """
        locations = np.fromiter(self._location2visits.keys(), dtype=int)
        weights = np.fromiter(self._location2visits.values(), dtype=float)
        weights = weights / np.sum(weights)
        location = np.random.choice(locations, size=1, p=weights)
        return int(location[0])

    def _preferential_return(self):
        """
        Choose the location the agent returns to, according to the visitation frequency
        of the previously visited locations.

        :return: int
            the identifier of the next location
        """
        next_location = self._weighted_random_selection()
        return next_location

    def _preferential_exploration(self, current_location):
        """
        Choose the new location the agent explores, according to the probabilities
        in the od matrix and given the agent's current location.

        :param current_location : int
            theedges identifier of the current location of the individual

        :return: int
            the identifier of the new location to explore
        """

        if self._is_sparse:
            prob_array = self._od_matrix.getrowview(current_location)
            if prob_array.nnz == 0:
                # if the row has been not populated
                self._populate_od_matrix(current_location)
            locations = np.arange(len(self._spatial_tessellation))
            weights = prob_array.toarray()[0]
            location = np.random.choice(locations, size=1, p=weights)
            return location[0]
        else:  # if the od matrix is precomputed
            locations = np.arange(len(self._od_matrix[current_location]))
            weights = self._od_matrix[current_location]
            location = np.random.choice(locations, size=1, p=weights)
            return location[0]

    def _populate_od_matrix(self, location):
        lat_i, lon_i = self._spatial_tessellation[location][latitude], self._spatial_tessellation[location][longitude]
        d_i = self._spatial_tessellation[location]['relevance']

        probs = []
        for id_j in self._spatial_tessellation:
            if id_j != location:
                lat_j, lon_j = self._spatial_tessellation[id_j][latitude], self._spatial_tessellation[id_j][longitude]
                d_j = self._spatial_tessellation[id_j]['relevance']

                p_ij = (d_i * d_j) / (earth_distance((lat_i, lon_i), (lat_j, lon_j)) ** 2)
                probs.append(p_ij)
            else:
                probs.append(0.0)

        # normalization by row
        sum_odm = sum(probs)
        if sum_odm > 0.0:
            self._od_matrix[location, :] = np.array(probs) / sum_odm

    def _choose_location(self):
        """
        Choose the next location to visit given the actual one.

        :return: int
            the identifier of the next location to visit
        """
        n_visited_locations = len(self._location2visits)  # number of already visited locations

        if n_visited_locations == 0:
            self._starting_loc = self._preferential_exploration(self._starting_loc)
            return self._starting_loc

        # choose a probability to return or explore
        p_new = random.uniform(0, 1)

        if p_new <= self._rho * math.pow(n_visited_locations, -self._gamma) and n_visited_locations != self._od_matrix.shape[0]:  # choose to return or explore
            # PREFERENTIAL EXPLORATION
            agent_id, current_time, current_location = self._trajectories_[-1]  # the last visited location
            next_location = self._preferential_exploration(current_location)
            while next_location in self._location2visits:
                next_location = self._preferential_exploration(current_location)
            return next_location

        else:
            # PREFERENTIAL RETURN
            next_location = self._preferential_return()
            return next_location

    def _get_trajdataframe(self, parameters):
        """
        Transform the trajectories list into a pandas DataFrame.

        :return: a pandas DataFrame describing the trajectories
        :rtype pandas DataFrame
        """
        df = pd.DataFrame(self._trajectories_, columns=[user_id, date_time, 'location'])
        df[[latitude, longitude]] = df.location.apply(lambda s: pd.Series({latitude: self._spatial_tessellation[s][latitude],
                                                                    longitude: self._spatial_tessellation[s][longitude]}))
        df = df.sort_values(by=[user_id, date_time]).drop('location', axis=1)
        return TrajDataFrame(df, parameters=parameters)

    def generate(self, start_date, end_date, spatial_tessellation, n_agents=1, starting_location=None, od_matrix=None,
                 random_state=None, verbose=False):
        """
        Start the simulation of the agent at time "start_date" till time "end_date".

        :param start_date : datetime
            the starting date of the simulation

        :param end_date : datetime
            the ending date of the simulation

        :param spatial_tessellation : dict
            the spatial tessellation, a dictionary of location to info (lat, lng, relevance)

        :param n_agents: int
            the number of agents to generate

        :param starting_location
            the identifier of the starting location for the simulation (as specified in the spatial tessellation)
        :type starting_location: int or None

        :param od_matrix: the od_matrix to use for deciding the movements. If None, it is computed "on the fly" during the simulation
        :type od_matrix: numpy array or None

        :param random_state: if int, random_state is the seed used by the random number generator; if None, the random number generator is the RandomState instance used by np.random and random.random (default: None)
        :type random_state: int or None
        """
        # Save function arguments and values in a dictionary
        frame = inspect.currentframe()
        args, _, _, arg_values = inspect.getargvalues(frame)
        parameters = dict([])
        parameters['model'] = {'class': self.__class__.__init__,
                               'generate': {i: arg_values[i] for i in args[1:] if i not in ['spatial_tessellation',
                                                                                           'od_matrix', 'log_file', 'verbose']}}
        # if specified, fix the random seeds to guarantee reproducibility of simulation
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # initialization of trajectories
        self._trajectories_ = []

        # setting of spatial tessellation
        self._spatial_tessellation = spatial_tessellation

        # infer the time_steps (in hours) from the start_date and the end_date
        delta_t = (end_date - start_date).total_seconds()
        n_hours = int((delta_t / 60.0) / 60.0)

        # initialization of od matrix
        if od_matrix is None:
            n = len(self._spatial_tessellation)
            self._od_matrix = lil_matrix((n, n))
            self._is_sparse = True
        else:
            self._od_matrix = od_matrix
            self._is_sparse = False

        # for each agent
        loop = range(1, n_agents + 1)
        if verbose:
            loop = tqdm(range(1, n_agents + 1))

        # sorted relevances
        relevances = np.array(sorted([[k, v['relevance']] for k,v in spatial_tessellation.items()]))[:,1]
        cs_relevances = np.cumsum(relevances / sum(relevances))

        for agent_id in loop:  # tqdm print a progress bar
            self._location2visits = defaultdict(int)
            if starting_location is None:
                # self._starting_loc = np.random.choice(np.fromiter(self._spatial_tessellation.keys(), dtype=int), size=1)[0]
                self._starting_loc = np.searchsorted(cs_relevances, random.random())
            else:
                self._starting_loc = starting_location

            # generate a mobility diary for the agent
            diary_df = self._diary_generator.generate(n_hours, start_date)

            for i, row in diary_df.iterrows():
                if row.abstract_location == 0:  # the agent is at home
                    self._trajectories_.append((agent_id, row.datetime, self._starting_loc))
                    self._location2visits[self._starting_loc] += 1

                else:  # the agent is not at home
                    next_location = self._choose_location()
                    self._trajectories_.append((agent_id, row.datetime, next_location))
                    self._location2visits[next_location] += 1

        tdf = self._get_trajdataframe(parameters)
        return tdf

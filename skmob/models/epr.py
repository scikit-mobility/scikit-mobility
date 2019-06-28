import powerlaw
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
import math
from tqdm import tqdm
from ..utils import constants, utils
from scipy.sparse import lil_matrix
import random
import logging
import inspect
from ..core.trajectorydataframe import TrajDataFrame
from ..models.gravity import Gravity

from geopy.distance import distance
earth_distance_km = (lambda p0, p1: distance(p0, p1).km)

latitude = constants.LATITUDE
longitude = constants.LONGITUDE
date_time = constants.DATETIME
user_id = constants.UID


def compute_od_matrix(gravity_singly, spatial_tessellation, tile_id_column=constants.TILE_ID,
                      relevance_column=constants.RELEVANCE):
    """
    Compute a matrix where element {ij} is the probability p_{ij} of moving between
    locations in rows i and j in the GeoDataFrame spatial_tessellation given as input.

    Parameters
    ----------

    :param gravity_singly: object
        instance of class collective.Gravity with argument gravity_type='singly constrained'

    :param spatial_tessellation: GeoDataFrame

    :param tile_id_column: str or int
        column of the GeoDataFrame containing the tile_ID of the locations/tiles

    :param relevance_column: str or int
        column of the GeoDataFrame containing the relevance of the locations/tiles

    :return:
    od_matrix: numpy array
        2-dim numpy array with the trip probabilities for each origin-destination pair
    """
    od_matrix = gravity_singly.generate(spatial_tessellation,
                                        tile_id_column=tile_id_column,
                                        tot_outflows_column=None,
                                        relevance_column=relevance_column,
                                        out_format='probabilities')
    return od_matrix


def populate_od_matrix(location, lats_lngs, relevances, gravity_singly):
    """
    Populate the od matrix with the probability to move from the location in input to all other locations
    in the spatial tessellation

    :param location: int
        the identifier of a location

    :param lats_lngs: list or numpy array
        list of coordinates of the centroids of the tiles in a spatial tessellation

    :param relevances: list or numpy array
        list of relevances of the tiles in a spatial tessellation

    :param gravity_singly: object
        instance of class collective.Gravity with argument gravity_type='singly constrained'

    :return:
        a numpy array of trip probabilities between the origin location and each destination
    """
    ll_origin = lats_lngs[location]
    distances = np.array([earth_distance_km(ll_origin, l) for l in lats_lngs])

    scores = gravity_singly.compute_gravity_score(distances, relevances[location], relevances)
    return scores / sum(scores)


class EPR:

    def __init__(self, name='EPR model', rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20):

        self._name = name

        self._rho = rho
        self._gamma = gamma
        self._tau = tau
        self._beta = beta

        self._location2visits = defaultdict(int)
        self._od_matrix = None
        self._is_sparse = True
        self._spatial_tessellation = None
        self.lats_lngs = None
        self.relevances = None
        self._starting_loc = None
        self.gravity_singly = None

        # Minimum waiting time (in hours)
        self._min_wait_time = min_wait_time_minutes / 60.0  # minimum waiting time

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

    def _weighted_random_selection(self, current_location):
        """
        Select a random location given their visitation frequency. Used by the return mechanism.

        :return: int
            a random location
        """
        locations = np.fromiter(self._location2visits.keys(), dtype=int)
        weights = np.fromiter(self._location2visits.values(), dtype=float)

        # remove the current location
        currloc_idx = np.where(locations == current_location)[0][0]
        locations = np.delete(locations, currloc_idx)
        weights = np.delete(weights, currloc_idx)

        weights = weights / np.sum(weights)
        location = np.random.choice(locations, size=1, p=weights)
        return int(location[0])

    def _preferential_return(self, current_location):
        """
        Choose the location the agent returns to, according to the visitation frequency
        of the previously visited locations.

        :return: int
            the identifier of the next location
        """
        next_location = self._weighted_random_selection(current_location)
        if self._log_file is not None:
            logging.info('RETURN to %s (%s, %s)' % (next_location, self.lats_lngs[next_location]))
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
                weights = populate_od_matrix(current_location, self.lats_lngs, self.relevances, self.gravity_singly)
                self._od_matrix[current_location, :] = weights
            else:
                weights = prob_array.toarray()[0]
            locations = np.arange(len(self.lats_lngs))
            location = np.random.choice(locations, size=1, p=weights)[0]

        else:  # if the matrix is precomputed
            locations = np.arange(len(self._od_matrix[current_location]))
            weights = self._od_matrix[current_location]
            location = np.random.choice(locations, size=1, p=weights)[0]

        if self._log_file is not None:
            logging.info('EXPLORATION to %s (%s, %s)' % (location, self.lats_lngs[location]))

        return location

    def _get_trajdataframe(self, parameters):
        """
        Transform the trajectories list into a pandas DataFrame.

        :return: a pandas DataFrame describing the trajectories
        :rtype pandas DataFrame
        """
        df = pd.DataFrame(self._trajectories_, columns=[user_id, date_time, 'location'])
        df[[latitude, longitude]] = df.location.apply(lambda s: pd.Series({latitude: self.lats_lngs[s][0],
                                                                           longitude: self.lats_lngs[s][1]}))
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

        agent_id, current_time, current_location = self._trajectories_[-1]  # the last visited location

        # choose a probability to return or explore
        p_new = random.uniform(0, 1)

        if (p_new <= self._rho * math.pow(n_visited_locations, -self._gamma) and n_visited_locations != \
                self._od_matrix.shape[0]) or n_visited_locations == 1:  # choose to return or explore
            # PREFERENTIAL EXPLORATION
            next_location = self._preferential_exploration(current_location)
            # TODO: remove the part below and exclude visited locations
            #  from the list of potential destinations in _preferential_exploration
            # while next_location in self._location2visits:
            #     next_location = self._preferential_exploration(current_location)
            return next_location

        else:
            # PREFERENTIAL RETURN
            next_location = self._preferential_return(current_location)
            return next_location

    def _time_generator(self):
        return powerlaw.Truncated_Power_Law(xmin=self.min_wait_time,
                                            parameters=[1. + self._beta, 1.0 / self._tau]).generate_random()[0]

    def _choose_waiting_time(self):
        """
        Choose the time (in hours) the agent has to wait before the next movement.

        :return: float
            the time to wait before the next movement.
        """
        time_to_wait = self._time_generator()
        return time_to_wait

    def generate(self, start_date, end_date, spatial_tessellation, gravity_singly={}, n_agents=1,
                 starting_locations=None, od_matrix=None,
                 relevance_column=constants.RELEVANCE,
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
        if starting_locations is not None and len(starting_locations) < n_agents:
            raise IndexError("The number of starting locations is smaller than the number of agents.")

        if gravity_singly == {}:
            self.gravity_singly = Gravity(gravity_type='singly constrained')

        # Save function arguments and values in a dictionary
        frame = inspect.currentframe()
        args, _, _, arg_values = inspect.getargvalues(frame)
        parameters = dict([])
        parameters['model'] = {'class': self.__class__.__init__,
                               'generate': {i: arg_values[i] for i in args[1:] if i not in ['spatial_tessellation',
                                                                                            'od_matrix', 'log_file',
                                                                                            'starting_locations']}}

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
        num_locs = len(spatial_tessellation)
        self.lats_lngs = spatial_tessellation.geometry.apply(utils.get_geom_centroid, args=[True]).values
        if relevance_column is None:
            self.relevances = np.ones(num_locs)
        else:
            self.relevances = spatial_tessellation[relevance_column].fillna(0).values

        # initialization of od matrix
        if od_matrix is None:
            self._od_matrix = lil_matrix((num_locs, num_locs))
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
            if starting_locations is None:
                self._starting_loc = np.random.choice(np.fromiter(range(num_locs), dtype=int), size=1)[0]
            else:
                self._starting_loc = starting_locations.pop()

            self._epr_generate_one_agent(agent_id, start_date, end_date)

        tdf = self._get_trajdataframe(parameters)
        return tdf

    def _epr_generate_one_agent(self, agent_id, start_date, end_date):

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


class DensityEPR(EPR):
    """
    The dEPR model of individual human mobility

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



    .. seealso:: :class:`EPR`

    References:
        .. [song2010modelling] Song, Chaoming, Koren, Tal, Wang, Pu and Barabasi, Albert-Laszlo. "Modelling the scaling properties of human mobility." Nature Physics 6 , no. 10 (2010): 818--823.
        .. [pappalardo2015returners] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F., Barabasi, A.-L. "Returners and Explorers dichotomy in human mobility.", Nature Communications, 6:8166, doi: 10.1038/ncomms9166 (2015).
        .. [pappalardo2016modelling] Pappalardo, L., Simini, F. Rinzivillo, S., "Human Mobility Modelling: exploration and preferential return meet the gravity model", Procedia Computer Science 83, doi: 10.1016/j.procs.2016.04.188 (2016).
    """

    def __init__(self, name='Density EPR model', rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20):

        super().__init__()
        self._name = name


class SpatialEPR(EPR):
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


    .. seealso:: :class:`EPR`

    References:
        .. [song2010modelling] Song, Chaoming, Koren, Tal, Wang, Pu and Barabasi, Albert-Laszlo. "Modelling the scaling properties of human mobility." Nature Physics 6 , no. 10 (2010): 818--823.
        .. [pappalardo2015returners] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F., Barabasi, A.-L. "Returners and Explorers dichotomy in human mobility.", Nature Communications, 6:8166, doi: 10.1038/ncomms9166 (2015).
        .. [pappalardo2016modelling] Pappalardo, L., Simini, F. Rinzivillo, S., "Human Mobility Modelling: exploration and preferential return meet the gravity model", Procedia Computer Science 83, doi: 10.1016/j.procs.2016.04.188 (2016).
    """

    def __init__(self, name='Spatial EPR model', rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20):

        super().__init__()
        self._name = name

    def generate(self, start_date, end_date, spatial_tessellation, gravity_singly={}, n_agents=1,
                 starting_locations=None, od_matrix=None,
                 relevance_column=None,
                 random_state=None, log_file=None, verbose=False):

        return super().generate(start_date, end_date, spatial_tessellation, gravity_singly=gravity_singly,
                                n_agents=n_agents,
                                starting_locations=starting_locations, od_matrix=od_matrix,
                                relevance_column=relevance_column,
                                random_state=random_state, log_file=log_file, verbose=verbose)


class Ditras(EPR):
    """
    The DITRAS (DIary-based TRAjectory Simulator) model of individual human mobility

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
        the name of the instantiation of the model (default: "Ditras model")

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

    Example:

        from skmob.models.epr import Ditras
        from skmob.models.markov_diary_generator import MarkovDiaryGenerator
        from skmob.preprocessing import filtering, compression, detection, clustering

        # Preeprocess the GPS data that will be used to fit the diary generator
        tdf = skmob.TrajDataFrame.from_file('./data/geolife_sample.txt.gz',
                                             latitude='lat', longitude='lon', user_id='user',
                                             datetime='datetime', sep=',')
        ctdf = compression.compress(tdf)
        stdf = detection.stops(ctdf)
        cstdf = clustering.cluster(stdf)

        # Create the diary generator using 2 users
        mdg = MarkovDiaryGenerator()
        mdg.fit(cstdf, 2, lid='cluster')

        # Instantiate the model
        start_time = pd.to_datetime('2019/01/01 08:00:00')
        end_time = pd.to_datetime('2019/01/14 08:00:00')
        ditras = Ditras(mdg)

        tessellation = gpd.GeoDataFrame.from_file("data/NY_counties_2011.geojson")

        # Generate 3 users
        tdf = ditras.generate(start_time, end_time, tessellation, relevance_column='population',
                    n_agents=3, od_matrix=None, verbose=True)

    References:

    .. [pappalardo2018data] Pappalardo, L, Simini, F, Data-driven generation of spatio-temporal routines in human mobility, Data Mining and Knowledge Discovery, 32:3 (2018).
    """

    def __init__(self, diary_generator, name='Ditras model', rho=0.3, gamma=0.21):

        super().__init__()
        self._diary_generator = diary_generator
        self._name = name
        self._rho = rho
        self._gamma = gamma

    def _epr_generate_one_agent(self, agent_id, start_date, end_date):

        # infer the time_steps (in hours) from the start_date and the end_date
        delta_t = (end_date - start_date).total_seconds()
        n_hours = int((delta_t / 60.0) / 60.0)

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

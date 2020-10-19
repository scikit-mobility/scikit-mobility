import powerlaw
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
import math
from tqdm import tqdm
from ..utils import constants, utils, gislib
from scipy.sparse import lil_matrix
import logging
import inspect
from ..core.trajectorydataframe import TrajDataFrame
from ..models.gravity import Gravity

earth_distance_km = gislib.getDistance

latitude = constants.LATITUDE
longitude = constants.LONGITUDE
date_time = constants.DATETIME
user_id = constants.UID


def compute_od_matrix(gravity_singly, spatial_tessellation, tile_id_column=constants.TILE_ID,
                      relevance_column=constants.RELEVANCE):
    """
    Compute a matrix :math:`M` where element :math:`M_{ij}` is the probability p_{ij} of moving between
    locations :math:`i` and location :math:`j`, where each location refers to a row in `spatial_tessellation`.

    Parameters
    ----------
    gravity_singly : object
        instance of class `collective.Gravity` with argument `gravity_type='singly constrained'`.

    spatial_tessellation : GeoDataFrame
        the spatial tessellation describing the division of the territory in locations.

    tile_id_column : str or int, optional
        column of in `spatial_tessellation` containing the identifier of the location/tile. The default value is constants.TILE_ID.

    relevance_column : str or int, optional
        column in `spatial_tessellation` containing the relevance of the location/tile.

    Returns
    -------
    od_matrix : numpy array
        two-dimensional numpy array with the trip probabilities for each origin-destination pair.
    """
    od_matrix = gravity_singly.generate(spatial_tessellation,
                                        tile_id_column=tile_id_column,
                                        tot_outflows_column=None,
                                        relevance_column=relevance_column,
                                        out_format='probabilities')
    return od_matrix


def populate_od_matrix(location, lats_lngs, relevances, gravity_singly):
    """
    Populate the origin-destination matrix with the probability to move from the location in input to all other locations in the spatial tessellation.
    
    Parameters
    ----------
    location : int
        the identifier of a location.

    lats_lngs : list or numpy array
        list of coordinates of the centroids of the tiles in a spatial tessellation.

    relevances : list or numpy array
        list of relevances of the tiles in a spatial tessellation.

    gravity_singly : object
        instance of class `collective.Gravity` with argument `gravity_type='singly constrained'`.
    
    Returns
    -------
        a numpy array of trip probabilities between the origin location and each destination.
    """
    ll_origin = lats_lngs[location]
    distances = np.array([earth_distance_km(ll_origin, l) for l in lats_lngs])

    scores = gravity_singly._compute_gravity_score(distances, relevances[location, None], relevances)[0]
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
        Select a random location given the agent's visitation frequency. Used by the return mechanism.
        
        Parameters
        ----------
        current_location : int
            identifier of a location.
            
        Returns
        -------
        int
            a location randomly chosen according to its relevance.
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
        
        Parameters
        ----------
        current_location : int
            the current location where the agent is.
            
        Returns
        -------
        int
            the identifier of the next location the agent moves to.
        """
        next_location = self._weighted_random_selection(current_location)
        if self._log_file is not None:
            logging.info('RETURN to %s (%s, %s)' % (next_location, self.lats_lngs[next_location]))
            logging.info('\t frequency = %s' % self._location2visits[next_location])
        return next_location

    def _preferential_exploration(self, current_location):
        """
        Choose the new location the agent explores, according to the probabilities
        in the origin-destination matrix.
        
        Parameters
        ----------
        current_location : int
            the identifier of the current location of the individual.
        
        Returns
        -------
        int
            the identifier of the new location the agent has to explore.
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
        
        Returns
        -------
        pandas DataFrame
            the trajectories of the agent.
        """
        df = pd.DataFrame(self._trajectories_, columns=[user_id, date_time, 'location'])
        df[[latitude, longitude]] = df.location.apply(lambda s: pd.Series({latitude: self.lats_lngs[s][0],
                                                                           longitude: self.lats_lngs[s][1]}))
        df = df.sort_values(by=[user_id, date_time]).drop('location', axis=1)
        return TrajDataFrame(df, parameters=parameters)

    def _choose_location(self):
        """
        Choose the next location to visit given the agent's current location.
        
        Returns
        -------
        int
            the identifier of the next location the agent has to visit.
        """
        n_visited_locations = len(self._location2visits)  # number of already visited locations

        if n_visited_locations == 0:
            self._starting_loc = self._preferential_exploration(self._starting_loc)
            return self._starting_loc

        agent_id, current_time, current_location = self._trajectories_[-1]  # the last visited location

        # choose a probability to return or explore
        p_new = np.random.uniform(0, 1)

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
        
        Returns
        -------
        float
            the time the agent has to wait before the next movement.
        """
        time_to_wait = self._time_generator()
        return time_to_wait

    def generate(self, start_date, end_date, spatial_tessellation, gravity_singly={}, n_agents=1,
                 starting_locations=None, od_matrix=None,
                 relevance_column=constants.RELEVANCE,
                 random_state=None, log_file=None, show_progress=False):
        """
        Start the simulation of a set of agents at time `start_date` till time `end_date`.
        
        Parameters
        ----------
        start_date : datetime
            the starting date of the simulation, in "YYY/mm/dd HH:MM:SS" format.

        end_date : datetime
            the ending date of the simulation, in "YYY/mm/dd HH:MM:SS" format.

        spatial_tessellation : geopandas GeoDataFrame
            the spatial tessellation, i.e., a division of the territory in locations. 
        
        gravity_singly : {} or Gravity, optional
            the (singly constrained) gravity model to use when generating the probability to move between two locations. The default is "{}".
        
        n_agents : int, optional
            the number of agents to generate. The default is 1.

        starting_locations : list or None, optional
            a list of integers, each identifying the location from which to start the simulation of each agent. Note that, if `starting_locations` is not None, its length must be equal to the value of `n_agents`, i.e., you must specify one starting location per agent. The default is None.
        
        od_matrix : numpy array or None, optional
            the origin destination matrix to use for deciding the movements of the agent (element [i,j] is the probability of one trip from location with tessellation index i to j, normalized by origin location) (element [i,j] is the probability of one trip from location with tessellation index i to j, normalized by origin location). If `None`, it is computed "on the fly" during the simulation. The default is None.
        
        relevance_column : str, optional
            the name of the column in `spatial_tessellation` to use as relevance variable. The default is "relevance".
        
        random_state : int or None, optional
            if int, it is the seed used by the random number generator; if None, the random number generator is the RandomState instance used by np.random and random.random. The default is None.
        
        log_file : str or None, optional
            the name of the file where to write a log of the execution of the model. The logfile will contain all decisions (returns or explorations) made by the model. The default is None.
            
        show_progress : boolean, optional
            if True, show a progress bar. The default is False.
        
        Returns
        -------
        TrajDataFrame
            the synthetic trajectories generated by the model
        """
        if starting_locations is not None and len(starting_locations) < n_agents:
            raise IndexError("The number of starting locations is smaller than the number of agents.")

        if gravity_singly == {}:
            self.gravity_singly = Gravity(gravity_type='singly constrained')
        elif type(gravity_singly) is Gravity:
            if gravity_singly.gravity_type == 'singly constrained':
                self.gravity_singly = gravity_singly
            else:
                raise AttributeError("Argument `gravity_singly` should be a skmob.models.gravity.Gravity object with argument `gravity_type` equal to 'singly constrained'.")
        else:
            raise TypeError("Argument `gravity_singly` should be of type skmob.models.gravity.Gravity.")

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
            # random.seed(random_state)
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
            # TODO: check it is a properly formatted stochastic matrix
            self._od_matrix = od_matrix
            self._is_sparse = False

        # for each agent
        loop = range(1, n_agents + 1)
        if show_progress:
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
    """Density-EPR model.
    
    The d-EPR model of individual human mobility consists of the following mechanisms [PSRPGB2015]_ [PSR2016]_: 
    
    **Waiting time choice**. The waiting time :math:`\Delta t` between two movements of the agent is chosen randomly from the distribution :math:`P(\Delta t) \sim \Delta t^{−1 −\\beta} \exp(−\Delta t/ \\tau)`. Parameters :math:`\\beta` and :math:`\\tau` correspond to arguments `beta` and `tau` of the constructor, respectively. 
    
    **Action selection**. With probability :math:`P_{new}=\\rho S^{-\\gamma}`, where :math:`S` is the number of distinct locations previously visited by the agent, the agent visits a new location (Exploration phase), otherwise it returns to a previously visited location (Return phase). Parameters :math:`\\rho` and :math:`\\gamma` correspond to arguments `rho` and `gamma` of the constructor, respectively.
    
    **Exploration phase**. If the agent that is currently in location :math:`i` explores a new location, then the new location :math:`j \\neq i` is selected according to the gravity model with probability :math:`p_{ij} = \\frac{1}{N} \\frac{n_i n_j}{r_{ij}^2}`, where :math:`n_{i (j)}` is the location's relevance, that is, the probability of a population to visit location :math:`i(j)`, :math:`r_{ij}` is the geographic distance between :math:`i` and :math:`j`, and :math:`N = \sum_{i, j \\neq i} p_{ij}` is a normalization constant. The number of distinct locations visited, :math:`S`, is increased by 1.
    
    **Return phase**. If the individual returns to a previously visited location, such a location :math:`i` is chosen with probability proportional to the number of time the agent visited :math:`i`, i.e., :math:`\Pi_i = f_i`, where :math:`f_i` is the visitation frequency of location :math:`i`.
    
    Parameters
    ----------
    name : str, optional
        the name of the instantiation of the d-EPR model. The default value is "Density EPR model".

    rho : float, optional
        it corresponds to the parameter :math:`\\rho \in (0, 1]` in the Action selection mechanism :math:`P_{new} = \\rho S^{-\gamma}` and controls the agent's tendency to explore a new location during the next move versus returning to a previously visited location. The default value is :math:`\\rho = 0.6` [SKWB2010]_.

    gamma : float, optional
        it corresponds to the parameter :math:`\gamma` (:math:`\gamma \geq 0`) in the Action selection mechanism :math:`P_{new} = \\rho S^{-\gamma}` and controls the agent's tendency to explore a new location during the next move versus returning to a previously visited location. The default value is :math:`\gamma=0.21` [SKWB2010]_.

    beta : float, optional
        it corresponds to the parameter :math:`\\beta` of the waiting time distribution in the Waiting time choice mechanism. The default value is :math:`\\beta=0.8` [SKWB2010]_.

    tau : int, optional
        it corresponds to the parameter :math:`\\tau` of the waiting time distribution in the Waiting time choice mechanism. The default value is :math:`\\tau = 17`, expressed in hours [SKWB2010]_.

    min_wait_time_minutes : int
        minimum waiting time between two movements, in minutes.
    
    Attributes
    ----------
    name : str
        the name of the instantiation of the model.

    rho : float
        the input parameter :math:`\\rho`.

    gamma : float
        the input parameters :math:`\gamma`.
        
    beta : float
        the input parameter :math:`\\beta`. 

    tau : int
        the input parameter :math:`\\tau`.

    min_wait_time_minutes : int
        the input parameters `min_wait_time_minutes`.

    Examples
    --------
    >>> import skmob
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> from skmob.models.epr import DensityEPR
    >>> url = >>> url = skmob.utils.constants.NY_COUNTIES_2011
    >>> tessellation = gpd.read_file(url)
    >>> start_time = pd.to_datetime('2019/01/01 08:00:00')
    >>> end_time = pd.to_datetime('2019/01/14 08:00:00')
    >>> depr = DensityEPR()
    >>> tdf = depr.generate(start_time, end_time, tessellation, relevance_column='population', n_agents=100, show_progress=True)
    >>> print(tdf.head())
       uid                   datetime        lat        lng
    0    1 2019-01-01 08:00:00.000000  42.780819 -76.823724
    1    1 2019-01-01 09:45:58.388540  42.728060 -77.775510
    2    1 2019-01-01 10:16:09.406408  42.780819 -76.823724
    3    1 2019-01-01 17:13:39.999037  42.852827 -77.299810
    4    1 2019-01-01 19:24:27.353379  42.728060 -77.775510
    >>> print(tdf.parameters)
    {'model': {'class': <function DensityEPR.__init__ at 0x7f548a49cf28>, 'generate': {'start_date': Timestamp('2019-01-01 08:00:00'), 'end_date': Timestamp('2019-01-14 08:00:00'), 'gravity_singly': {}, 'n_agents': 100, 'relevance_column': 'population', 'random_state': None, 'show_progress': True}}}
    
    References
    ----------
    .. [PSRPGB2015] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F. & Barabasi, A. L. (2015) Returners and Explorers dichotomy in human mobility. Nature Communications 6, https://www.nature.com/articles/ncomms9166
    .. [PSR2016] Pappalardo, L., Simini, F. Rinzivillo, S. (2016) Human Mobility Modelling: exploration and preferential return meet the gravity model. Procedia Computer Science 83, https://www.sciencedirect.com/science/article/pii/S1877050916302216
    .. [SKWB2010] Song, C., Koren, T., Wang, P. & Barabasi, A.L. (2010) Modelling the scaling properties of human mobility. Nature Physics 6, 818-823, https://www.nature.com/articles/nphys1760
    
    See Also
    --------
    EPR, SpatialEPR, Ditras
    """

    def __init__(self, name='Density EPR model', rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20):

        super().__init__()
        self._name = name
        
    def generate(self, start_date, end_date, spatial_tessellation, gravity_singly={}, n_agents=1,
                 starting_locations=None, od_matrix=None, relevance_column=constants.RELEVANCE,
                 random_state=None, log_file=None, show_progress=False):
        """
        Start the simulation of a set of agents at time `start_date` till time `end_date`.
        
        Parameters
        ----------
        start_date : datetime
            the starting date of the simulation, in "YYY/mm/dd HH:MM:SS" format.

        end_date : datetime
            the ending date of the simulation, in "YYY/mm/dd HH:MM:SS" format.

        spatial_tessellation : geopandas GeoDataFrame
            the spatial tessellation, i.e., a division of the territory in locations. 
        
        gravity_singly : {} or Gravity, optional
            the gravity model (singly constrained) to use when generating the probability to move between two locations (note, used by DensityEPR). The default is "{}".
        
        n_agents : int, optional
            the number of agents to generate. The default is 1.

        relevance_column : str, optional
            the name of the column in `spatial_tessellation` to use as relevance variable. The default is "relevance".

        starting_locations : list or None, optional
            a list of integers, each identifying the location from which to start the simulation of each agent. Note that, if `starting_locations` is not None, its length must be equal to the value of `n_agents`, i.e., you must specify one starting location per agent. The default is None.
        
        od_matrix : numpy array or None, optional
            the origin destination matrix to use for deciding the movements of the agent (element [i,j] is the probability of one trip from location with tessellation index i to j, normalized by origin location). If `None`, it is computed "on the fly" during the simulation. The default is None.
        
        random_state : int or None, optional
            if int, it is the seed used by the random number generator; if None, the random number generator is the RandomState instance used by np.random and random.random. The default is None.
        
        log_file : str or None, optional
            the name of the file where to write a log of the execution of the model. The logfile will contain all decisions (returns or explorations) made by the model. The default is None.
            
        show_progress : boolean, optional
            if True, show a progress bar. The default is False.
        
        Returns
        -------
        TrajDataFrame
            the synthetic trajectories generated by the model
        """
        return super().generate(start_date, end_date, spatial_tessellation, gravity_singly=gravity_singly, n_agents=n_agents, starting_locations=starting_locations, od_matrix=od_matrix, relevance_column=relevance_column, random_state=random_state, log_file=log_file, show_progress=show_progress)


class SpatialEPR(EPR):
    """Spatial-EPR model.
    
    The s-EPR model of individual human mobility consists of the following mechanisms [PSRPGB2015]_ [PSR2016]_ [SKWB2010]_:
    
    **Waiting time choice**. The waiting time :math:`\Delta t` between two movements of the agent is chosen randomly from the distribution :math:`P(\Delta t) \sim \Delta t^{−1 −\\beta} \exp(−\Delta t/ \\tau)`. Parameters :math:`\\beta` and :math:`\\tau` correspond to arguments `beta` and `tau` of the constructor, respectively. 
    
    **Action selection**. With probability :math:`P_{new}=\\rho S^{-\\gamma}`, where :math:`S` is the number of distinct locations previously visited by the agent, the agent visits a new location (Exploration phase), otherwise it returns to a previously visited location (Return phase). Parameters :math:`\\rho` and :math:`\\gamma` correspond to arguments `rho` and `gamma` of the constructor, respectively.
    
    **Exploration phase**. If the agent that is currently in location :math:`i` explores a new location, then the new location :math:`j \\neq i` is selected according to the distance from the current location :math:`p_{ij} = \\frac{1}{r_{ij}^2}`, where :math:`r_{ij}` is the geographic distance between :math:`i` and :math:`j`. The number of distinct locations visited, :math:`S`, is increased by 1.
    
    **Return phase**. If the individual returns to a previously visited location, such a location :math:`i` is chosen with probability proportional to the number of time the agent visited :math:`i`, i.e., :math:`\Pi_i = f_i`, where :math:`f_i` is the visitation frequency of location :math:`i`.
    
    .. image:: https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fnphys1760/MediaObjects/41567_2010_Article_BFnphys1760_Fig2_HTML.jpg?as=webp
    Starting at time :math:`t` from the configuration shown in the left panel, indicating that the user visited previously :math:`S=4` locations with frequency :math:`f_i` that is proportional to the size of circles drawn at each location, at time :math:`t + \Delta t` (with :math:`Delta t` drawn from the :math:`P(\Delta t)` fat-tailed distribution) the user can either visit a new location at distance :math:`\Delta r` from his/her present location, or return to a previously visited location with probability :math:`P_{ret}=\\rho S^{-\\gamma}`, where the next location will be chosen with probability :math:`\Pi_i=f_i` (preferential return; lower panel). Figure from [SKWB2010]_.
    
    Parameters
    ----------
    name : str, optional
        the name of the instantiation of the s-EPR model. The default value is "Spatial EPR model".

    rho : float, optional
        it corresponds to the parameter :math:`\\rho \in (0, 1]` in the Action selection mechanism :math:`P_{new} = \\rho S^{-\gamma}` and controls the agent's tendency to explore a new location during the next move versus returning to a previously visited location. The default value is :math:`\\rho = 0.6` [SKWB2010]_.

    gamma : float, optional
        it corresponds to the parameter :math:`\gamma` (:math:`\gamma \geq 0`) in the Action selection mechanism :math:`P_{new} = \\rho S^{-\gamma}` and controls the agent's tendency to explore a new location during the next move versus returning to a previously visited location. The default value is :math:`\gamma=0.21` [SKWB2010]_.

    beta : float, optional
        it corresponds to the parameter :math:`\\beta` of the waiting time distribution in the Waiting time choice mechanism. The default value is :math:`\\beta=0.8` [SKWB2010]_.

    tau : int, optional
        it corresponds to the parameter :math:`\\tau` of the waiting time distribution in the Waiting time choice mechanism. The default value is :math:`\\tau = 17`, expressed in hours [SKWB2010]_.

    min_wait_time_minutes : int
        the input parameters `min_wait_time_minutes`
    
    Attributes
    ----------
    name : str
        the name of the instantiation of the model.

    rho : float
        the input parameter :math:`\\rho`.

    gamma : float
        the input parameters :math:`\gamma`.
        
    beta : float
        the input parameter :math:`\\beta`. 

    tau : int
        the input parameter :math:`\\tau`. 

    min_wait_time_minutes : int
        the input parameters `min_wait_time_minutes`.
    
    Examples
    --------
    >>> import skmob
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> from skmob.models.epr import SpatialEPR
    >>> url = >>> url = skmob.utils.constants.NY_COUNTIES_2011
    >>> tessellation = gpd.read_file(url)
    >>> start_time = pd.to_datetime('2019/01/01 08:00:00')
    >>> end_time = pd.to_datetime('2019/01/14 08:00:00')
    >>> sepr = SpatialEPR()
    >>> tdf = sepr.generate(start_time, end_time, tessellation, n_agents=100, show_progress=True)
    >>> print(tdf.head())
       uid                   datetime        lat        lng
    0    1 2019-01-01 08:00:00.000000  42.267915 -77.383591
    1    1 2019-01-01 13:06:13.973868  42.633510 -77.105324
    2    1 2019-01-01 14:17:41.188668  42.452018 -76.473618
    3    1 2019-01-01 14:49:30.896248  42.633510 -77.105324
    4    1 2019-01-01 15:10:54.133150  43.382528 -78.230656
    >>> print(tdf.parameters)
    {'model': {'class': <function SpatialEPR.__init__ at 0x7f548a49e048>, 'generate': {'start_date': Timestamp('2019-01-01 08:00:00'), 'end_date': Timestamp('2019-01-14 08:00:00'), 'gravity_singly': {}, 'n_agents': 100, 'relevance_column': None, 'random_state': None, 'show_progress': True}}}    
    
    See Also
    --------
    EPR, DensityEPR, Ditras        
    """

    def __init__(self, name='Spatial EPR model', rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20):

        super().__init__()
        self._name = name

    def generate(self, start_date, end_date, spatial_tessellation, gravity_singly={}, n_agents=1,
                 starting_locations=None, od_matrix=None, random_state=None, log_file=None, show_progress=False):
        """
        Start the simulation of a set of agents at time `start_date` till time `end_date`.
        
        Parameters
        ----------
        start_date : datetime
            the starting date of the simulation, in "YYY/mm/dd HH:MM:SS" format.

        end_date : datetime
            the ending date of the simulation, in "YYY/mm/dd HH:MM:SS" format.

        spatial_tessellation : geopandas GeoDataFrame
            the spatial tessellation, i.e., a division of the territory in locations. 
        
        gravity_singly : {} or Gravity, optional
            the gravity model (singly constrained) to use when generating the probability to move between two locations (note, used by DensityEPR). The default is "{}".
        
        n_agents : int, optional
            the number of agents to generate. The default is 1.

        starting_locations : list or None, optional
            a list of integers, each identifying the location from which to start the simulation of each agent. Note that, if `starting_locations` is not None, its length must be equal to the value of `n_agents`, i.e., you must specify one starting location per agent. The default is None.
        
        od_matrix : numpy array or None, optional
            the origin destination matrix to use for deciding the movements of the agent (element [i,j] is the probability of one trip from location with tessellation index i to j, normalized by origin location). If `None`, it is computed "on the fly" during the simulation. The default is None.
        
        random_state : int or None, optional
            if int, it is the seed used by the random number generator; if None, the random number generator is the RandomState instance used by np.random and random.random. The default is None.
        
        log_file : str or None, optional
            the name of the file where to write a log of the execution of the model. The logfile will contain all decisions (returns or explorations) made by the model. The default is None.
            
        show_progress : boolean, optional
            if True, show a progress bar. The default is False.
        
        Returns
        -------
        TrajDataFrame
            the synthetic trajectories generated by the model
        """
        return super().generate(start_date, end_date, spatial_tessellation, gravity_singly=gravity_singly, n_agents=n_agents, starting_locations=starting_locations, od_matrix=od_matrix, relevance_column=None, random_state=random_state, log_file=log_file, show_progress=show_progress)


class Ditras(EPR):
    """Ditras modelling framework.
    
    The DITRAS (DIary-based TRAjectory Simulator) modelling framework to simulate the spatio-temporal patterns of human mobility [PS2018]_. DITRAS consists of two phases: 
    
    **Mobility Diary Generation**. In the first phase, DITRAS generates a *mobility diary* which captures the temporal patterns of human mobility. 
    
    **Trajectory Generation**. In the second phase, DITRAS transforms the mobility diary into a mobility trajectory which captures the spatial patterns of human movements. 

    .. image:: https://raw.githubusercontent.com/jonpappalord/DITRAS/master/DITRAS_schema.png
    
    **Outline of the DITRAS framework**. DITRAS combines two probabilistic models: a diary generator (e.g., :math:`MD(t)`) and trajectory generator (e.g., d-EPR). The diary generator produces a mobility diary :math:`D`. The mobility diary :math:`D` is the input of the trajectory generator together with a weighted spatial tessellation of the territory :math:`L`. From :math:`D` and :math:`L` the trajectory generator produces a synthetic mobility trajectory :math:`S`.

    Parameters
    ----------
    diary_generator : MarkovDiaryGenerator
        the diary generator to use for generating the diary.
    
    name : str, optional
        the name of the instantiation of the Ditras model. The default value is "Ditras".

    rho : float, optional
        it corresponds to the parameter :math:`\\rho \in (0, 1]` in the Action selection mechanism  of the DensityEPR model :math:`P_{new} = \\rho S^{-\gamma}` and controls the agent's tendency to explore a new location during the next move versus returning to a previously visited location. The default value is :math:`\\rho = 0.6` [SKWB2010]_.

    gamma : float, optional
        it corresponds to the parameter :math:`\gamma` (:math:`\gamma \geq 0`) in the Action selection mechanism of the DensityEPR model :math:`P_{new} = \\rho S^{-\gamma}` and controls the agent's tendency to explore a new location during the next move versus returning to a previously visited location. The default value is :math:`\gamma=0.21` [SKWB2010]_.

    Attributes
    ----------
    diary_generator : MarkovDiaryGenerator
        the diary generator to use for generating the diary [PS2018]_.
    
    name : str
        the name of the instantiation of the model.

    rho : float
        the input parameter :math:`\\rho`.

    gamma : float
        the input parameters :math:`\gamma`.

    Examples
    --------
    >>> import skmob
    >>> from skmob.models.epr import Ditras
    >>> from skmob.models.markov_diary_generator import MarkovDiaryGenerator
    >>> from skmob.preprocessing import filtering, compression, detection, clustering
    >>> 
    >>> # load and preprocess data to train the MarkovDiaryGenerator
    >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
    >>> df = pd.read_csv(url, sep=',', compression='gzip')
    >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
    >>> ctdf = compression.compress(tdf)
    >>> stdf = detection.stops(ctdf)
    >>> cstdf = clustering.cluster(stdf)
    >>> 
    >>> # instantiate and train the MarkovDiaryGenerator
    >>> mdg = MarkovDiaryGenerator()
    >>> mdg.fit(cstdf, 2, lid='cluster')
    >>> 
    >>> # set start time, end time and tessellation for the simulation
    >>> start_time = pd.to_datetime('2019/01/01 08:00:00')
    >>> end_time = pd.to_datetime('2019/01/14 08:00:00')
    >>> tessellation = gpd.GeoDataFrame.from_file("data/NY_counties_2011.geojson")
    >>> 
    >>> # instantiate the model
    >>> ditras = Ditras(mdg)
    >>> 
    >>> # run the model
    >>> ditras_tdf = ditras.generate(start_time, end_time, tessellation, relevance_column='population',
                    n_agents=3, od_matrix=None, show_progress=True)
    >>> print(ditras_tdf.head())
       uid            datetime        lat        lng
    0    1 2019-01-01 08:00:00  43.382528 -78.230656
    1    1 2019-01-02 03:00:00  43.309133 -77.680414
    2    1 2019-01-02 23:00:00  43.382528 -78.230656
    3    1 2019-01-03 10:00:00  43.382528 -78.230656
    4    1 2019-01-03 21:00:00  43.309133 -77.680414
    >>> print(ditras_tdf.parameters)
    {'model': {'class': <function Ditras.__init__ at 0x7f0cf0b7e158>, 'generate': {'start_date': Timestamp('2019-01-01 08:00:00'), 'end_date': Timestamp('2019-01-14 08:00:00'), 'gravity_singly': {}, 'n_agents': 3, 'relevance_column': 'population', 'random_state': None, 'show_progress': True}}}
    
    References
    ----------
    .. [PS2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4
    
    See Also
    --------
    DensityEPR, MarkovDiaryGenerator
    """

    def __init__(self, diary_generator, name='Ditras model', rho=0.3, gamma=0.21):

        super().__init__()
        self._diary_generator = diary_generator
        self._name = name
        self._rho = rho
        self._gamma = gamma

    def generate(self, start_date, end_date, spatial_tessellation, gravity_singly={}, n_agents=1,
                 starting_locations=None, od_matrix=None,
                 relevance_column=constants.RELEVANCE,
                 random_state=None, log_file=None, show_progress=False):
        """
        Start the simulation of a set of agents at time `start_date` till time `end_date`.
        
        Parameters
        ----------
        start_date : datetime
            the starting date of the simulation, in "YYY/mm/dd HH:MM:SS" format.

        end_date : datetime
            the ending date of the simulation, in "YYY/mm/dd HH:MM:SS" format.

        spatial_tessellation : geopandas GeoDataFrame
            the spatial tessellation, i.e., a division of the territory in locations. 
        
        gravity_singly : {} or Gravity, optional
            the (singly constrained) gravity model to use when generating the probability to move between two locations. The default is "{}".
        
        n_agents : int, optional
            the number of agents to generate. The default is 1.

        starting_locations : list or None, optional
            a list of integers, each identifying the location from which to start the simulation of each agent. Note that, if `starting_locations` is not None, its length must be equal to the value of `n_agents`, i.e., you must specify one starting location per agent. The default is None.
        
        od_matrix : numpy array or None, optional
            the origin destination matrix to use for deciding the movements of the agent (element [i,j] is the probability of one trip from location with tessellation index i to j, normalized by origin location). If `None`, it is computed "on the fly" during the simulation. The default is None.
        
        relevance_column : str, optional
            the name of the column in `spatial_tessellation` to use as relevance variable. The default is "relevance".
        
        random_state : int or None, optional
            if int, it is the seed used by the random number generator; if None, the random number generator is the RandomState instance used by np.random and random.random. The default is None.
        
        log_file : str or None, optional
            the name of the file where to write a log of the execution of the model. The logfile will contain all decisions (returns or explorations) made by the model. The default is None.
            
        show_progress : boolean, optional
            if True, show a progress bar. The default is False.
        
        Returns
        -------
        TrajDataFrame
            the synthetic trajectories generated by the model
        """
        if starting_locations is not None and len(starting_locations) < n_agents:
            raise IndexError("The number of starting locations is smaller than the number of agents.")

        if gravity_singly == {}:
            self.gravity_singly = Gravity(gravity_type='singly constrained')
        elif type(gravity_singly) is Gravity:
            if gravity_singly.gravity_type == 'singly constrained':
                self.gravity_singly = gravity_singly
            else:
                raise AttributeError("Argument `gravity_singly` should be a skmob.models.gravity.Gravity object with argument `gravity_type` equal to 'singly constrained'.")
        else:
            raise TypeError("Argument `gravity_singly` should be of type skmob.models.gravity.Gravity.")

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
            # random.seed(random_state)
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
        if show_progress:
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

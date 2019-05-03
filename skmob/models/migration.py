import numpy as np
# import csv
try:
    import statsmodels.api as sm
except ImportError:
    import statsmodels as sm
from tqdm import tqdm
import operator
import pandas as pd
# from collections import defaultdict
from ..utils import constants
from ..utils.gislib import getDistanceByHaversine


def ci(i, number_locs):
    """
    The multinomial fit is a poisson fit
    with `number_locs` more parameters needed to
    implement the normalization constraints:
    i.e. the poisson variables in each of the
    `number_locs` origin location must sum to one.

    This function selects the appropriate
    parameter relative to the normalization
    constant in each origin location.
    """
    c = list(np.zeros(number_locs))
    c[i] = 1.
    return c


def exponential_deterrence_func(x, R):
    """
    Exponential deterrence function
    """
    return np.exp(- x / R)


def powerlaw_deterrence_func(x, exponent):
    """
    Power law deterrence function
    """
    return np.power(x, exponent)


class Gravity:
    """
    The Gravity model of human migration.

    Parameters
    ----------
    deterrence_func_type  :  str
        the type of deterrence function.
        available types: 'power_law' (default), 'exponential'

    deterrence_func_args  :  list
        arguments of the deterrence function.

    origin_exp  :  float (1.0), exponent of origin's revelance
        (only relevnt to globally-constrained model).

    destination_exp  :  float (1.0), exponent of destination's revelance.

    gravity_type  :  str
        gravity model type.
        available types: 'singly constrained', 'globally constrained'.
        default: 'singly constrained'

    Notes
    ----------
    The gravity model is a model of human migration derived from Newton's law of gravity,
    and it is used to predict the degree of interaction between two places.
    The gravity model of migration is based upon the idea that as the importance of one
    or both of the location increases, there will also be an increase in movement between them.
    The farther apart the two locations are, however, the movement between them will be less (distance decay).

    References
    ----------
    .. [1] Erlander, Sven, and Neil F. Stewart. "The gravity model in transportation analysis: theory and extensions".
    Vol. 3. Vsp, 1990.

    """

    def __init__(self, deterrence_func_type='power_law', deterrence_func_args=[-2.0],
                 origin_exp=1.0, destination_exp=1.0, gravity_type='singly constrained', 
                name='Gravity model'):

        self._name = name
        self._deterrence_func_type = deterrence_func_type
        self._deterrence_func_args = deterrence_func_args
        self._origin_exp = origin_exp
        self._destination_exp = destination_exp
        self._gravity_type = gravity_type

        # set the deterrence function
        if self._deterrence_func_type == 'power_law':  # power law deterrence function
            self._deterrence_func = powerlaw_deterrence_func
        elif self._deterrence_func_type == 'exponential':  # exponential deterrence function
            self._deterrence_func = exponential_deterrence_func
        else:
            print('Deterrence function type "%s" not available. Power law will be used.\nAvailable deterrence functions are [power_law, exponential]' % self._deterrence_func_type)
            self._deterrence_func = powerlaw_deterrence_func

    @property
    def name(self):
        return self._name

    @property
    def deterrence_func_type(self):
        return self._deterrence_func_type

    @property
    def deterrence_func_args(self):
        return self._deterrence_func_args

    @property
    def origin_exp(self):
        return self._origin_exp

    @property
    def destination_exp(self):
        return self._destination_exp

    @property
    def gravity_type(self):
        return self._gravity_type

    def __str__(self):
        return 'Gravity(name=\"%s\", deterrence_func_type=\"%s\", deterrence_func_args=%s, origin_exp=%s, destination_exp=%s, gravity_type=\"%s\")' % (self._name, self._deterrence_func_type, self._deterrence_func_args, self._origin_exp, self._destination_exp, self._gravity_type)
    
    @staticmethod
    def _compute_distance_matrix(spatial_tessellation):
        """
        Compute the matrix of distances between all pairs of locations

        Parameters
        ----------
        coords: numpy array

        Returns
        -------
        distance_matrix: numpy array
        """
        n = len(spatial_tessellation)
        distance_matrix = np.zeros((n, n))
        for id_i in tqdm(spatial_tessellation):
            lat_i, lng_i = spatial_tessellation[id_i][constants.LATITUDE], spatial_tessellation[id_i][constants.LONGITUDE]
            for id_j in range(id_i + 1, n):
                lat_j, lng_j = spatial_tessellation[id_j][constants.LATITUDE], spatial_tessellation[id_j][constants.LONGITUDE]
                distance = getDistanceByHaversine((lat_i, lng_i), (lat_j, lng_j))
                distance_matrix[id_i, id_j] = distance
                distance_matrix[id_j, id_i] = distance
        return distance_matrix

    
    def generate(self, spatial_tessellation, relevance=constants.RELEVANCE, out_format='flows'):
        
        if out_format not in ['flows', 'probabilities']:
            print('Output format \"%s\" not available. Flows will be used.\nAvailable output formats are [flows, probabilities]' % out_format)
            out_format = "flows"
        
        n_locs = len(spatial_tessellation)
        relevances = np.array([info[relevance] for location, info in spatial_tessellation.items()])

        # if outflows are specified in a "tot_outflows" column then use that column,
        # otherwise use the "relevance" column
        try:
            tot_outflows = np.array([info['tot_outflows'] for location, info in spatial_tessellation.items()])
        except KeyError:
            tot_outflows = relevances

        # compute the distances between all pairs of locations
        distance_matrix = self._compute_distance_matrix(spatial_tessellation)

        trip_probs_matrix = self._deterrence_func(distance_matrix, * self._deterrence_func_args)
        trip_probs_matrix[trip_probs_matrix == trip_probs_matrix[0, 0]] = 0.0

        trip_probs_matrix = np.transpose(trip_probs_matrix * relevances ** self.destination_exp) * relevances ** self._origin_exp

        if self._gravity_type == 'globally constrained':  # globally constrained gravity model
            trip_probs_matrix /= np.sum(trip_probs_matrix)
            # put the NaN to 0.0
            np.putmask(trip_probs_matrix, np.isnan(trip_probs_matrix), 0.0)

            # generate random fluxes according to trip probabilities
            od_matrix = np.reshape(np.random.multinomial(np.sum(tot_outflows), trip_probs_matrix.flatten()), (n_locs, n_locs))

        else:  # singly constrained gravity model
            trip_probs_matrix = np.transpose(trip_probs_matrix / np.sum(trip_probs_matrix, axis=1))
            # put the NaN to 0.0
            np.putmask(trip_probs_matrix, np.isnan(trip_probs_matrix), 0.0)
            # generate random fluxes according to trip probabilities
            od_matrix = np.array([np.random.multinomial(tot_outflows[i], row) for i, row in enumerate(trip_probs_matrix)])
        
        if out_format == 'flows':
            return od_matrix
        else:
            return trip_probs_matrix

    def _update_training_set(self, flow_example):
        id_origin, id_destination, trips = flow_example.origin, flow_example.destination, flow_example.flow

        if id_origin == id_destination:
            return

        try:
            coords_origin = self._spatial_tessellation[id_origin]['lat'], self._spatial_tessellation[id_origin]['lng']
            weight_origin = self._spatial_tessellation[id_origin]['relevance']
        except KeyError:
            print('Missing information for location ' + \
                  '"%s" in spatial tessellation. Skipping ...' % id_origin)
            return

        try:
            coords_destination = self._spatial_tessellation[id_destination]['lat'], self._spatial_tessellation[id_destination]['lng']
            weight_destination = self._spatial_tessellation[id_destination]['relevance']
        except KeyError:
            print('Missing information for location ' + \
                  '"%s" in spatial tessellation. Skipping ...' % id_destination)
            return

        if weight_destination <= 0:
            return

        dist = getDistanceByHaversine(coords_origin, coords_destination)

        if self._gravity_type == 'globally constrained':
            sc_vars = [np.log(weight_origin)]
        else:  # if singly constrained
            sc_vars = ci(id_origin, len(self._spatial_tessellation))

        if self._deterrence_func_type == 'exponential':
            # exponential deterrence function
            self.X += [[1.] + sc_vars + [np.log(weight_destination), -dist]]
        else:
            # power law deterrence function
            self.X += [[1.] + sc_vars + [np.log(weight_destination), np.log(dist)]]

        self.y += [float(trips)]
    
    def fit(self, spatial_tessellation, flow_df, delimiter=',', quotechar='"'):
        """
        Fit the gravity model parameters to the flows in file `filename`.
        Can fit globally or singly constrained gravity models using a
        Generalized Linear Model (GLM) with a Poisson regression.

        Parameters
        ----------
        locations_info  :  pandas DataFrame with info about the spatial tessellation.
            Must contain the columns:
            "id": str, name or identifier of the location
            "lat": float, latitude of the location's centroid
            "lon": float, longitude of the location's centroid
            "relevance": float, number of opportunities at the location
                (e.g., population or total number of visits).
            Optional columns:
            "tot_outflow": float, total outgoing flow from the location.

        filename  :  str; the path to the file where the flows are store.
            File format: "origin location ID", "destination location ID", flow.
            The location ids must match the "id"s in `locations_info`.

        delimiter  :  str; column delimiter in file `filename`. Default: ','.

        quotechar  :  str; one-character string used to quote fields containing
            special characters in file `filename`. Default: '"'.


        Returns
        -------

        X  :  list of independent variables (features) used in the GLM fit.

        y   :  list of dependent varibles (flows) used in the GLM fit.

        poisson_results  :  statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
            statsmodels object with information on the fit's qualityand predictions.

        Referencesfilter_threshold=1,
        ----------

        .. [1] Agresti, Alan.
            "Categorical data analysis."
            Vol. 482. John Wiley & Sons, 2003.

        .. [2] Flowerdew, Robin, and Murray Aitkin.
            "A method of fitting the gravity model based on the Poisson distribution."
            Journal of regional science 22.2 (1982): 191-202.

        """
        self._spatial_tessellation = spatial_tessellation
        self.X, self.y = [], [] # independent (X) and dependent (y) variables
        
        flow_df.progress_apply(lambda flow_example: self._update_training_set(flow_example), 
                               axis=1)

        # Perform GLM fit
        poisson_model = sm.GLM(self.y, self.X, family=sm.families.Poisson(link=sm.families.links.log))
        poisson_results = poisson_model.fit()

        # Set best fit parameters
        if self._gravity_type == 'globally constrained':
            self._origin_exp = poisson_results.params[1]
            self._destination_exp = poisson_results.params[2]
            self._deterrence_func_args = [poisson_results.params[3]]
        else:  # if singly constrained
            self._origin_exp = 1.
            self._destination_exp = poisson_results.params[-2]
            self._deterrence_func_args = [poisson_results.params[-1]]
        
        # we delete the instance variables we do not need anymore
        del self.X
        del self.y
        del self._spatial_tessellation


class Radiation:
    """
    The radiation model for human migration.

    :param name: str
        the name of the instantiation of the radiation model

    Attributes
    ----------
    :ivar name: str
        the name of the instantiation of the model
        default: "Radiation model"

    :ivar edges_: pandas DataFrame
        the generated edges (flows or probabilities)

    Notes
    ------
    The radiation model describes the flows of people between different locations.
    In particular, the fundamental equation of the radiation model gives the average
    flux between two counties:
    .. math::

        <T_{ij}> = \frac{m_i n_j}{(m_i + s_{ij})(m_i + n_j + s_{ij})}

    where where :math:T_{ij} is the total number of commuters from county :math:i, :math:m_i and :math:n_j
    are the population in county :math:i and :math:j respectively, and :math:s_{ij} is the total population
    in the circle centered at :math:i and touching :math:j excluding the source and the destination population.

    References
    ----------
    .. [1] Simini, Filippo, Gonzalez, Marta C., Maritan, Amos and Barabasi, Albert-Laszlo.
    "A universal model for mobility and migration patterns."
    Nature 484 , no. 7392 (2012): 96--100.

    .. [2] https://en.wikipedia.org/wiki/Radiation_law_for_human_mobility

    Examples
    --------
    >>> locations_info = pd.read_csv('../datasets/city_fluxes_US.csv').rename(columns={'population': 'relevance'})
    >>> locations_info = locations_info.assign(id=locations_info.index)
    >>> radiation = Radiation("my first radiation")
    >>> radiation.start_simulation(locations_info, filename='radiation_OD_trips_cities.csv', filter_threshold=1.0)
    Processing location 385 of 385...
    Done.
    >>> radiation.flows_.head()
    origin  destination flows_average
    0       304         67327.0
    1       374         34105.0
    2       230         16413.0
    3       213         15943.0
    4       49          3560.0

    """

    def __init__(self, name='Radiation model'):
        self.name_ = name
        self._spatial_tessellation = None
        self._out_format = None

    def _get_flows(self, origin, total_relevance, distance_f=getDistanceByHaversine):
        """
        Compute the edges (flows or probabilities) from location `origin` to all other locations.

        Parameters
        ----------
        origin  :  int or str
            identifier of the origin location

        location2info : dict
            information of the locations

        total_relevance : float
            sum of all relevances

        distance_f  :  callable
            distance function
            default: getDistanceByHaversine

        Returns
        -------
        edges : numpy array
            the edges generated from `origin` to the other locations

        Notes
        ------
        `m`  :  relevance of origin
        `n`  :  relevance of destination
        `s`  :  relevance in the circle between origin and destination

        """
        edges = []
        origin_lat = self._spatial_tessellation[origin][constants.LATITUDE]
        origin_lng = self._spatial_tessellation[origin][constants.LONGITUDE]
        origin_relevance = float(self._spatial_tessellation[origin]['relevance'])

        try:
            origin_outflow = self._spatial_tessellation[origin]['outflow']
        except KeyError:
            origin_outflow = origin_relevance

        if origin_outflow > 0.0:

            # compute the normalization factor
            normalization_factor = 1.0 / (1.0 - origin_relevance / total_relevance)

            # calculate the distance to all other locations
            destinations_and_distances = []
            for destination in self._spatial_tessellation:
                dest_lat = self._spatial_tessellation[destination][constants.LATITUDE]
                dest_lng = self._spatial_tessellation[destination][constants.LONGITUDE]
                if destination != origin:
                    destinations_and_distances += \
                        [(destination, distance_f((origin_lat, origin_lng), (dest_lat, dest_lng)))]

            # sort the destinations by distance (from the closest to the farest)
            destinations_and_distances.sort(key=operator.itemgetter(1))

            sum_inside = 0.0
            for destination, _ in destinations_and_distances:
                destination_relevance = self._spatial_tessellation[destination]['relevance']
                prob_origin_destination = normalization_factor * \
                                          (origin_relevance * destination_relevance) / \
                                          ((origin_relevance + sum_inside) * (origin_relevance + sum_inside + destination_relevance))

                sum_inside += destination_relevance
                edges.append([origin, destination, prob_origin_destination])

            edges = np.array(edges)
            probs = edges[:, 2]

            if self._out_format == 'flows_average':
                quantities = np.rint(origin_outflow * probs)

            elif self._out_format == 'flows_sample':
                quantities = np.random.multinomial(origin_outflow, probs)

            else:
                quantities = probs

            edges[:, 2] = quantities

        return edges


    def generate(self, spatial_tessellation, out_format='flows_average'):
        """
        Start the simulation of the Radiation model.

        :param spatial_tessellation : pandas DataFrame
            a pandas DataFrame containing the information about the spatial tessellation
            to use. It must contain the following columns:
            - "id": str, name or identifier of the location
            - "lat": float, l_edges_atitude of the location's centroid
            - "lon": float, longitude of the location's centroid
            - "relevance": float, number of opportunities at the location (e.g., population or total number of visits).

        :param out_format: {"flows_sample", "flows_average", "probs"}
            the type of edges to be generated. Three possible values:
            - "flows_sample" : the number of migrations generation by a single execution of the model
            -    "flows_average" : the average number of migrations between two locations
            - "probs" : the probability of movement between two locations
            default : "flows_average"
        """
        self._spatial_tessellation = spatial_tessellation
        self._out_format = out_format

        # check if arguments are valid
        if out_format not in ['flows_average', 'flows_sample', 'probs']:
            raise ValueError(
                'Value of out_format "%s" is not valid. \nValid values: flows_average, flows_sample, probs.' % out_format)

        # compute the total relevance, i.e., the sum of relevances of all the locations
        total_relevance = sum([info['relevance'] for location, info in self._spatial_tessellation.items()])

        all_flows = []
        for origin in tqdm(self._spatial_tessellation):  # tqdm print a progress bar

            # get the edges for the current origin location
            flows_from_origin = self._get_flows(origin, total_relevance)

            if len(flows_from_origin) > 0:
                all_flows += list(flows_from_origin)
        
        return np.array(all_flows)

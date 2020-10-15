import numpy as np
#try:
#    import statsmodels.api as sm
#except ImportError:
#    import statsmodels as sm
import statsmodels as sm
from statsmodels.genmod.generalized_linear_model import GLM
from tqdm import tqdm
from ..utils import gislib, constants, utils
from ..core.flowdataframe import FlowDataFrame

distfunc = gislib.getDistance


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
    return np.exp(- x * R)


def powerlaw_deterrence_func(x, exponent):
    """
    Power law deterrence function
    """
    return np.power(x, exponent)


def compute_distance_matrix(spatial_tessellation, origins):
    """
    Compute the matrix of distances between origin locations and all other locations.

    Parameters
    ----------
    spatial_tessellation : GeoDataFrame
        the spatial tessellation.
    
    origins : list
        indexes of the locations of origin.
    
    Returns
    -------
    distance_matrix : numpy array with distances between locations in `origins`
        and all locations in `spatial_tessellation`
    """
    coordinates = spatial_tessellation.geometry.apply(utils.get_geom_centroid, args=[True]).values

    n = len(spatial_tessellation)
    distance_matrix = np.zeros((n, n))

    for id_i in tqdm(origins):
        lat_i, lng_i = coordinates[id_i]
        for id_j in range(id_i + 1, n):
            lat_j, lng_j = coordinates[id_j]
            distance = distfunc((lat_i, lng_i), (lat_j, lng_j))
            distance_matrix[id_i, id_j] = distance
            distance_matrix[id_j, id_i] = distance
    return distance_matrix


class Gravity:
    """Gravity model.
    
    The Gravity model of human migration. In its original formulation, the probability :math:`T_{ij}` of moving from a location :math:`i` to location :math:`j` is defined as [Z1946]_:
    
    .. math::
        T_{ij} \propto \\frac{P_i P_j}{r_{ij}} 

    where :math:`P_i` and :math:`P_j` are the population of location :math:`i` and :math:`j` and :math:`r_{ij}` is the distance between :math:`i` and :math:`j`. The basic assumptions of this model are that the number of trips leaving location :math:`i` is proportional to its population :math:`P_i`, the attractivity of location :math:`j` is also proportional to :math:`P_j`, and finally, that there is a cost effect in terms of distance traveled. These ideas can be generalized assuming a relation of the type [BBGJLLMRST2018]_:
    
    .. math::
        T_{ij} = K m_i m_j f(r_{ij})
        
    where :math:`K` is a constant, the masses :math:`m_i` and :math:`m_j` relate to the number of trips leaving location :math:`i` or the ones attracted by location :math:`j`, and :math:`f(r_{ij})`, called *deterrence function*, is a decreasing function of distance. The deterrence function :math:`f(r_{ij})` is commonly modeled with a powerlaw or an exponential form.

    **Constrained gravity models**. Some of the limitations of the gravity model can be resolved via constrained versions. For example, one may hold the number of people originating from a location :math:`i` to be a known quantity :math:`O_i`, and the gravity model is then used to estimate the destination, constituting a so-called *singly constrained* gravity model of the form:

    .. math::
        T_{ij} = K_i O_i m_j f(r_{ij}) = O_i \\frac{m_i f(r_{ij})}{\sum_k m_k f(r_{ik})}.
        
    In this formulation, the proportionality constants :math:`K_i` depend on the location of the origin and its distance to the other places considered. We can fix also the total number of travelers arriving at a destination :math:`j` as :math:`D_j = \sum_i T_{ij}`, leading to a *doubly-constrained* gravity model. For each origin-destination pair, the flow is calculated as:
    
    .. math::
        T_{ij} = K_i O_i L_j D_j f(r_{ij})
        
    where there are now two flavors of proportionality constants
    
    .. math::
        K_i = \\frac{1}{\sum_j L_j D_j f(r_{ij})}, L_j = \\frac{1}{\sum_i K_i O_i f(r_{ij})}.

    Parameters
    ----------
    deterrence_func_type : str, optional
        the deterrence function to use. The possible deterrence function are "power_law" and "exponential". The default is "power_law".

    deterrence_func_args : list, optional
        the arguments of the deterrence function. The default is [-2.0].

    origin_exp : float, optional 
        the exponent of the origin's relevance (only relevant to globally-constrained model). The default is `1.0`.

    destination_exp : float, optional 
        the explonent of the destination's relevance. The default is 1.0.

    gravity_type : str, optional
        the type of gravity model. Possible values are "singly constrained" and "globally constrained". The default is "singly constrained".
        
    name : str, optional
        the name of the instantiation of the Gravity model. The default is "Gravity model".
    
    Attributes
    ----------
    deterrence_func_type : str
        the deterrence function to use. The possible deterrence function are "power_law" and "exponential". 

    deterrence_func_args : list
        the arguments of the deterrence function. 

    origin_exp : float
        the exponent of the origin's relevance (only relevant to globally-constrained model). 

    destination_exp : float
        the explonent of the destination's relevance.

    gravity_type : str
        the type of gravity model. Possible values are "singly constrained" and "globally constrained". 
        
    name : str
        the name of the instantiation of the Gravity model. 
    
    Examples
    --------
    >>> import skmob
    >>> from skmob.utils import utils, constants
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from skmob.models import Gravity
    >>> # load a spatial tessellation
    >>> url_tess = >>> url = skmob.utils.constants.NY_COUNTIES_2011
    >>> tessellation = gpd.read_file(url_tess).rename(columns={'tile_id': 'tile_ID'})
    >>> print(tessellation.head())
      tile_ID  population                                           geometry
    0   36019       81716  POLYGON ((-74.006668 44.886017, -74.027389 44....
    1   36101       99145  POLYGON ((-77.099754 42.274215, -77.0996569999...
    2   36107       50872  POLYGON ((-76.25014899999999 42.296676, -76.24...
    3   36059     1346176  POLYGON ((-73.707662 40.727831, -73.700272 40....
    4   36011       79693  POLYGON ((-76.279067 42.785866, -76.2753479999...    
    >>> # load real flows into a FlowDataFrame
    >>> fdf = skmob.FlowDataFrame.from_file(skmob.utils.constants.NY_FLOWS_2011,
                                            tessellation=tessellation, 
                                            tile_id='tile_ID', 
                                            sep=",")
    >>> print(fdf.head())
         flow origin destination
    0  121606  36001       36001
    1       5  36001       36005
    2      29  36001       36007
    3      11  36001       36017
    4      30  36001       36019    
    >>> # compute the total outflows from each location of the tessellation (excluding self loops)
    >>> tot_outflows = fdf[fdf['origin'] != fdf['destination']].groupby(by='origin', axis=0)['flow'].sum().fillna(0).values
    >>> tessellation[constants.TOT_OUTFLOW] = tot_outflows
    >>> print(tessellation.head())
      tile_id  population                                           geometry  \
    0   36019       81716  POLYGON ((-74.006668 44.886017, -74.027389 44....   
    1   36101       99145  POLYGON ((-77.099754 42.274215, -77.0996569999...   
    2   36107       50872  POLYGON ((-76.25014899999999 42.296676, -76.24...   
    3   36059     1346176  POLYGON ((-73.707662 40.727831, -73.700272 40....   
    4   36011       79693  POLYGON ((-76.279067 42.785866, -76.2753479999...   
       tot_outflow  
    0        29981  
    1         5319  
    2       295916  
    3         8665  
    4         8871 
    >>> # instantiate a singly constrained Gravity model
    >>> gravity_singly = Gravity(gravity_type='singly constrained')
    >>> print(gravity_singly)
    Gravity(name="Gravity model", deterrence_func_type="power_law", deterrence_func_args=[-2.0], origin_exp=1.0, destination_exp=1.0, gravity_type="singly constrained")
    >>> np.random.seed(0)
    >>> synth_fdf = gravity_singly.generate(tessellation, 
                                       tile_id_column='tile_ID', 
                                       tot_outflows_column='tot_outflow', 
                                       relevance_column= 'population',
                                       out_format='flows')
    >>> print(synth_fdf.head())
      origin destination  flow
    0  36019       36101   101
    1  36019       36107    66
    2  36019       36059  1041
    3  36019       36011   151
    4  36019       36123    33
    >>> # fit the parameters of the Gravity model from real fluxes
    >>> gravity_singly_fitted = Gravity(gravity_type='singly constrained')
    >>> print(gravity_singly_fitted)
    Gravity(name="Gravity model", deterrence_func_type="power_law", deterrence_func_args=[-2.0], origin_exp=1.0, destination_exp=1.0, gravity_type="singly constrained")
    >>> gravity_singly_fitted.fit(fdf, relevance_column='population')
    >>> print(gravity_singly_fitted)
    Gravity(name="Gravity model", deterrence_func_type="power_law", deterrence_func_args=[-1.9947152031914186], origin_exp=1.0, destination_exp=0.6471759552223144, gravity_type="singly constrained") 
    >>> np.random.seed(0)
    >>> synth_fdf_fitted = gravity_singly_fitted.generate(tessellation, 
                                                            tile_id_column='tile_ID', 
                                                            tot_outflows_column='tot_outflow', 
                                                            relevance_column= 'population', 
                                                            out_format='flows')
    >>> print(synth_fdf_fitted.head())
      origin destination  flow
    0  36019       36101   102
    1  36019       36107    66
    2  36019       36059  1044
    3  36019       36011   152
    4  36019       36123    33
    
    References
    ----------
    .. [Z1946] Zipf, G. K. (1946) The P 1 P 2/D hypothesis: on the intercity movement of persons. American sociological review 11(6), 677-686, https://www.jstor.org/stable/2087063?seq=1#metadata_info_tab_contents
    .. [W1971] Wilson, A. G. (1971) A family of spatial interaction models, and associated developments. Environment and Planning A 3(1), 1-32, https://econpapers.repec.org/article/pioenvira/v_3a3_3ay_3a1971_3ai_3a1_3ap_3a1-32.htm
    .. [BBGJLLMRST2018] Barbosa, H., Barthelemy, M., Ghoshal, G., James, C. R., Lenormand, M., Louail, T., Menezes, R., Ramasco, J. J. , Simini, F. & Tomasini, M. (2018) Human mobility: Models and applications. Physics Reports 734, 1-74, https://www.sciencedirect.com/science/article/abs/pii/S037015731830022X
    
    See Also
    --------
    Radiation
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
            print('Deterrence function type "%s" not available. Power law will be used.\n'
                  'Available deterrence functions are [power_law, exponential]' % self._deterrence_func_type)
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
        return 'Gravity(name=\"%s\", deterrence_func_type=\"%s\", ' \
               'deterrence_func_args=%s, origin_exp=%s, destination_exp=%s, gravity_type=\"%s\")' % \
               (self._name, self._deterrence_func_type, self._deterrence_func_args, self._origin_exp,
                self._destination_exp, self._gravity_type)

    def _compute_gravity_score(self, distance_matrix, relevances_orig, relevances_dest):
        trip_probs_matrix = self._deterrence_func(distance_matrix, *self._deterrence_func_args)
        # trip_probs_matrix = np.transpose(
        #     trip_probs_matrix * relevances ** self.destination_exp) * relevances ** self._origin_exp
        trip_probs_matrix = trip_probs_matrix * relevances_dest ** self.destination_exp * \
                            np.expand_dims(relevances_orig ** self._origin_exp, axis=1)
        # put the NaN and Inf to 0.0
        np.putmask(trip_probs_matrix, np.isnan(trip_probs_matrix), 0.0)
        np.putmask(trip_probs_matrix, np.isinf(trip_probs_matrix), 0.0)

        # put diagonal elements to zero: i.e. exclude intra-location trips (self flows)
        np.fill_diagonal(trip_probs_matrix, 0.)

        return trip_probs_matrix

    def generate(self, spatial_tessellation, tile_id_column=constants.TILE_ID,
                 tot_outflows_column=constants.TOT_OUTFLOW, relevance_column=constants.RELEVANCE, out_format='flows'):
        """
        Start the simulation of the Gravity model.
        
        Parameters
        ----------
        spatial_tessellation : GeoDataFrame
            the spatial tessellation on which to run the simulation.
            
        tile_id_column : str, optional
            the column in `spatial_tessellation` of the location identifier. The default is `constants.TILE_ID`.
            
        tot_outflows_column : str, optional
            the column in `spatial_tessellation` with the outflow of the location. The default is `constants.TOT_OUTFLOW`.
            
        relevance_column : str, optional
            the column in `spatial_tessellation` with the relevance of the location. The default is `constants.RELEVANCE`.
            
        out_format : str, optional
            the format of the generated flows. Possible values are "flows" (average flow between two locations), "flows_sample" (random sample of flows), and "probabilities" (probability of a unit flow between two locations). The default is "flows".
            
        Returns
        -------
        FlowDataFrame
            the flows generated by the Gravity model.
        """
        n_locs = len(spatial_tessellation)
        relevances = spatial_tessellation[relevance_column].fillna(0).values
        self._tile_id_column = tile_id_column
        # self._spatial_tessellation = spatial_tessellation

        if out_format not in ['flows', 'flows_sample', 'probabilities']:
            print('Output format \"%s\" not available. Flows will be used.\n'
                  'Available output formats are [flows, flows_sample, probabilities]' % out_format)
            out_format = "flows"

        if 'flows' in out_format:
            if tot_outflows_column not in spatial_tessellation.columns:
                raise KeyError("The column 'tot_outflows' must be present in the tessellation.")
            tot_outflows = spatial_tessellation[tot_outflows_column].fillna(0).values

        # the origin locations are all locations
        origins = np.arange(n_locs)

        # compute the distances between all pairs of locations
        distance_matrix = compute_distance_matrix(spatial_tessellation, origins)

        # compute scores
        trip_probs_matrix = self._compute_gravity_score(distance_matrix, relevances, relevances)

        if self._gravity_type == 'globally constrained':  # globally constrained gravity model
            trip_probs_matrix /= np.sum(trip_probs_matrix)

            if out_format == 'flows':
                # return average flows
                od_matrix = trip_probs_matrix * np.sum(tot_outflows)
                return self._from_matrix_to_flowdf(od_matrix, origins, spatial_tessellation)
            elif out_format == 'flows_sample':
                # generate random fluxes according to trip probabilities
                od_matrix = np.reshape(np.random.multinomial(np.sum(tot_outflows), trip_probs_matrix.flatten()),
                                       (n_locs, n_locs))
                return self._from_matrix_to_flowdf(od_matrix, origins, spatial_tessellation)
            else:
                # return trip_probs_matrix
                return self._from_matrix_to_flowdf(trip_probs_matrix, origins, spatial_tessellation)

        else:  # singly constrained gravity model
            # trip_probs_matrix = np.transpose(trip_probs_matrix / np.sum(trip_probs_matrix, axis=0))
            trip_probs_matrix = (trip_probs_matrix.T / np.sum(trip_probs_matrix, axis=1)).T
            # put the NaN and Inf to 0.0
            np.putmask(trip_probs_matrix, np.isnan(trip_probs_matrix), 0.0)
            np.putmask(trip_probs_matrix, np.isinf(trip_probs_matrix), 0.0)

            if out_format == 'flows':
                od_matrix = (trip_probs_matrix.T * tot_outflows).T
                return self._from_matrix_to_flowdf(od_matrix, origins, spatial_tessellation)
            elif out_format == 'flows_sample':
                # generate random fluxes according to trip probabilities
                od_matrix = np.array([np.random.multinomial(tot_outflows[i], trip_probs_matrix[i]) for i in origins])
                return self._from_matrix_to_flowdf(od_matrix, origins, spatial_tessellation)
            else:
                # return trip_probs_matrix
                return self._from_matrix_to_flowdf(trip_probs_matrix, origins, spatial_tessellation)

    def _from_matrix_to_flowdf(self, flow_matrix, origins, spatial_tessellation):
        index2tileid = dict([(i, tileid) for i, tileid in enumerate(spatial_tessellation[self._tile_id_column].values)])
        output_list = [[index2tileid[i], index2tileid[j], flow]
                       for i in origins for j, flow in enumerate(flow_matrix[i]) if flow > 0.]
        return FlowDataFrame(output_list, origin=0, destination=1, flow=2,
                             tile_id=self._tile_id_column, tessellation=spatial_tessellation)

    def _update_training_set(self, flow_example):

        id_origin = flow_example[constants.ORIGIN]
        id_destination = flow_example[constants.DESTINATION]
        trips = flow_example[constants.FLOW]

        if id_origin == id_destination:
            return

        try:
            coords_origin = self.lats_lngs[self.tileid2index[id_origin]]
            weight_origin = self.weights[self.tileid2index[id_origin]]
        except KeyError:
            print('Missing information for location ' + \
                  '"%s" in spatial tessellation. Skipping ...' % id_origin)
            return

        try:
            coords_destination = self.lats_lngs[self.tileid2index[id_destination]]
            weight_destination = self.weights[self.tileid2index[id_destination]]
        except KeyError:
            print('Missing information for location ' + \
                  '"%s" in spatial tessellation. Skipping ...' % id_destination)
            return

        if weight_destination <= 0:
            return

        dist = distfunc(coords_origin, coords_destination)

        if self._gravity_type == 'globally constrained':
            sc_vars = [np.log(weight_origin)]
        else:  # if singly constrained
            sc_vars = ci(self.tileid2index[id_origin], len(self.tileid2index))

        if self._deterrence_func_type == 'exponential':
            # exponential deterrence function
            self.X += [[1.] + sc_vars + [np.log(weight_destination), - dist]]
        else:
            # power law deterrence function
            self.X += [[1.] + sc_vars + [np.log(weight_destination), np.log(dist)]]

        self.y += [float(trips)]

    def fit(self, flow_df, relevance_column=constants.RELEVANCE):
        """
        Fit the parameters of the Gravity model to the flows provided in input, using a Generalized Linear Model (GLM) with a Poisson regression [FM1982]_.

        Parameters
        ----------
        flow_df  :  FlowDataFrame 
            the real flows on the spatial tessellation. 
            
        relevance_column : str, optional
            the column in the spatial tessellation with the relevance of the location. The default is constants.RELEVANCE.

        References
        ----------
        .. [FM1982] Flowerdew, R. & Murray, A. (1982) A method of fitting the gravity model based on the Poisson distribution. Journal of regional science 22(2), 191-202, https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9787.1982.tb00744.x

        """
        self.lats_lngs = flow_df.tessellation.geometry.apply(utils.get_geom_centroid, args=[True]).values
        self.weights = flow_df.tessellation[relevance_column].fillna(0).values
        self.tileid2index = dict(
            [(tileid, i) for i, tileid in enumerate(flow_df.tessellation[constants.TILE_ID].values)])

        self.X, self.y = [], []  # independent (X) and dependent (y) variables

        # flow_df.progress_apply(lambda flow_example: self._update_training_set(flow_example),
        #                        axis=1)
        flow_df.apply(lambda flow_example: self._update_training_set(flow_example), axis=1)

        # Perform GLM fit
        poisson_model = GLM(self.y, self.X, family=sm.genmod.families.family.Poisson(link=sm.genmod.families.links.log))
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

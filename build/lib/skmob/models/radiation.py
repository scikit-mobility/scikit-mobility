import numpy as np
from tqdm import tqdm
import operator
import pandas as pd
from ..utils import gislib, constants, utils
from ..core.flowdataframe import FlowDataFrame

# from geopy.distance import distance
# distfunc = (lambda p0, p1: distance(p0, p1).km)
distfunc = gislib.getDistance


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

    def _get_flows(self, origin, total_relevance):
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
        probs = []

        origin_lat, origin_lng = self.lats_lngs[origin]
        origin_relevance = self.relevances[origin]

        try:
            origin_outflow = self.tot_outflows[origin]
        except AttributeError:
            origin_outflow = 1

        if origin_outflow > 0.0:

            # compute the normalization factor
            normalization_factor = 1.0 / (1.0 - origin_relevance / total_relevance)

            destinations_and_distances = []
            for destination, (dest_lat, dest_lng) in enumerate(self.lats_lngs):
                if destination != origin:
                    destinations_and_distances += \
                        [(destination, distfunc((origin_lat, origin_lng), (dest_lat, dest_lng)))]

            # sort the destinations by distance (from the closest to the farthest)
            destinations_and_distances.sort(key=operator.itemgetter(1))

            sum_inside = 0.0
            for destination, _ in destinations_and_distances:
                destination_relevance = self.relevances[destination]
                prob_origin_destination = normalization_factor * \
                                          (origin_relevance * destination_relevance) / \
                                          ((origin_relevance + sum_inside) * (
                                                      origin_relevance + sum_inside + destination_relevance))

                sum_inside += destination_relevance
                edges += [[origin, destination]]
                probs.append(prob_origin_destination)

            probs = np.array(probs)

            if self._out_format == 'flows_average':
                quantities = np.rint(origin_outflow * probs)
            elif self._out_format == 'flows_sample':
                quantities = np.random.multinomial(origin_outflow, probs)
            else:
                quantities = probs

            edges = [edges[i] + [od] for i, od in enumerate(quantities)]

        return edges

    def generate(self, spatial_tessellation, tile_id_column=constants.TILE_ID,
                 tot_outflows_column=constants.TOT_OUTFLOW,
                 relevance_column=constants.RELEVANCE, out_format='flows_average'):
        """
        Start the simulation of the Radiation model.

        :param spatial_tessellation : GeoDataFrame
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
        self._out_format = out_format
        self._tile_id_column = tile_id_column
        self.lats_lngs = spatial_tessellation.geometry.apply(utils.get_geom_centroid, args=[True]).values
        self.relevances = spatial_tessellation[relevance_column].fillna(0).values
        if 'flows' in out_format:
            if tot_outflows_column not in spatial_tessellation.columns:
                raise KeyError(
                    "The column %s for the 'tot_outflows' must be present in the tessellation." % tot_outflows_column)
            self.tot_outflows = spatial_tessellation[tot_outflows_column].fillna(0).values

        # check if arguments are valid
        if out_format not in ['flows_average', 'flows_sample', 'probs']:
            raise ValueError(
                'Value of out_format "%s" is not valid. \nValid values: flows_average, flows_sample, probs.' % out_format)

        # compute the total relevance, i.e., the sum of relevances of all the locations
        total_relevance = np.sum(self.relevances)

        all_flows = []
        for origin in tqdm(range(len(spatial_tessellation))):  # tqdm print a progress bar

            # get the edges for the current origin location
            flows_from_origin = self._get_flows(origin, total_relevance)

            if len(flows_from_origin) > 0:
                all_flows += list(flows_from_origin)

        # return np.array(all_flows)
        if 'flows' in out_format:
            return self._from_matrix_to_flowdf(all_flows, spatial_tessellation)
        else:
            return all_flows

    def _from_matrix_to_flowdf(self, all_flows, spatial_tessellation):
        index2tileid = dict([(i, tileid) for i, tileid in enumerate(spatial_tessellation[self._tile_id_column].values)])
        output_list = [[index2tileid[i], index2tileid[j], flow] for i, j, flow in all_flows if flow > 0.]
        return FlowDataFrame(output_list, origin=0, destination=1, flow=2,
                             tile_id=self._tile_id_column, tessellation=spatial_tessellation)

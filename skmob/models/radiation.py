import numpy as np
from tqdm import tqdm
import operator
import pandas as pd
from ..utils import gislib, constants, utils
from ..core.flowdataframe import FlowDataFrame

distfunc = gislib.getDistance


class Radiation:
    """Radiation model.
    
    The radiation model for human migration. The radiation model assumes that the choice of a traveler's destination consists of two steps. First, each opportunity in every location is assigned a fitness represented by a number :math:`z`, chosen from some distribution :math:`P(z)` whose value represents the quality of the opportunity for the traveler. Second, the traveler ranks all opportunities according to their distances from the origin location and chooses the closest opportunity with a fitness higher than the traveler's fitness threshold, which is another random number extracted from the fitness distribution :math:`P(z)`. As a result, the average number of travelers from location :math:`i` to location :math:`j` takes the form [SGMB2012]_:
    
    .. math:: 
        T_{ij} = O_i \\frac{1}{1 - \\frac{m_i}{M}}\\frac{m_i m_j}{(m_i + s_{ij})(m_i + m_j + s_{ij})}.
        
    The destination of the :math:`O_i` trips originating in :math:`i` is sampled from a distribution of probabilities that a trip originating in :math:`i` ends in location :math:`j`. This probability depends on the number of opportunities at the origin :math:`m_i`, at the destination :math:`m_j` and the number of opportunities :math:`s_{ij}` in a circle of radius :math:`r_{ij}` centered in :math:`i` (excluding the source and destination). This conditional probability needs to be normalized so that the probability that a trip originating in the region of interest ends in this region is equal to 1. In case of a finite system it is possible to show that this is equal to :math:`1 - \\frac{m_i}{M}` where :math:`M=\\sum_i m_i` is the total number of opportunities. In the original version of the radiation model, the number of opportunities is approximated by the population, but the total inflows :math:`D_j` to each destination can also be used.

    .. image:: https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fnature10856/MediaObjects/41586_2012_Article_BFnature10856_Fig1_HTML.jpg?as=webp
    *(a)* To demonstrate the limitations of the gravity law we highlight two pairs of counties, one in Utah (UT) and the other in Alabama (AL), with similar origin (:math:`m`, blue) and destination (:math:`n`, green) populations and comparable distance :math:`r` between them (see bottom left table). The US census 2000 reports a flux that is an order of magnitude greater between the Utah counties, a difference correctly captured by the radiation model *(b, c)*. *(b)* The definition of the radiation model: an individual (for example, living in Saratoga County, New York) applies for jobs in all counties and collects potential employment offers. The number of job opportunities in each county (:math:`j`) is :math:`n_j / n_{jobs}`, chosen to be proportional to the resident population :math:`n_j`. Each offer's attractiveness (benefit) is represented by a random variable with distribution :math:`P(z)`, the numbers placed in each county representing the best offer among the :math:`n_j / n_{jobs}` trials in that area. Each county is marked in green (red) if its best offer is better (lower) than the best offer in the home county (here :math:`z = 10`). *(c)* An individual accepts the closest job that offers better benefits than his home county. In the shown configuration the individual will commute to Oneida County, New York, the closest county whose benefit :math:`z = 13` exceeds the home county benefit :math:`z = 10`. This process is repeated for each potential commuter, choosing new benefit variables :math:`z` in each case. Figure from [SGMB2012]_.
    
    Parameters
    ----------
    name : str, optional
        the name of the instantiation of the radiation model. The default is 'Radiation model'. 

    Attributes
    ----------
    name : str
        the name of the instantiation of the model.
    
    Examples
    --------
    >>> import skmob
    >>> from skmob.utils import utils, constants
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from skmob.models import Radiation
    >>> # load a spatial tessellation
    >>> url_tess = skmob.utils.constants.NY_COUNTIES_2011
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
    >>> tessellation[skmob.constants.TOT_OUTFLOW] = tot_outflows
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
    >>> np.random.seed(0)
    >>> radiation = Radiation()
    >>> rad_flows = radiation.generate(tessellation, tile_id_column='tile_ID',  tot_outflows_column='tot_outflow', relevance_column='population', out_format='flows_sample')
    >>> print(rad_flows.head())
      origin destination   flow
    0  36019       36033  11648
    1  36019       36031   4232
    2  36019       36089   5598
    3  36019       36113   1596
    4  36019       36041    117
    
    References
    ----------
    .. [SGMB2012] Simini, F., Gonzàlez, M. C., Maritan, A. & Barabasi, A.-L. (2012) A universal model for mobility and migration patterns. Nature 484(7392), 96-100, https://www.nature.com/articles/nature10856
    .. [MSJB2013] Masucci, A. P., Serras, J., Johansson, A., & Batty, M. (2013). Gravity versus radiation models: On the importance of scale and heterogeneity in commuting flows. Physical Review E, 88(2), 022812.
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

            if self._out_format == 'flows':
                quantities = np.rint(origin_outflow * probs)
            elif self._out_format == 'flows_sample':
                quantities = np.random.multinomial(origin_outflow, probs)
            else:
                quantities = probs

            edges = [edges[i] + [od] for i, od in enumerate(quantities)]

        return edges

    def generate(self, spatial_tessellation, tile_id_column=constants.TILE_ID,
                 tot_outflows_column=constants.TOT_OUTFLOW,
                 relevance_column=constants.RELEVANCE, out_format='flows'):
        """
        Start the simulation of the Radiation model.
        
        Parameters
        ----------
        spatial_tessellation : GeoDataFrame
            the spatial tessellation on which to perform the simulation. 
        
        tile_id_column : str, optional
            the column in `spatial_tessellation` of the location identifier. The default is `constants.TILE_ID`.
            
        tot_outflows_column : str, optional
            the column in `spatial_tessellation` with the outflow of the location. The default is `constants.TOT_OUTFLOW`.
            
        relevance_column : str, optional
            the column in `spatial_tessellation` with the relevance of the location. The default is `constants.RELEVANCE`.
            
        out_format : str, optional
            the format of the generated flows. Possible values are: "flows" (average flow between two locations), "flows_sample" (random sample of flows), and "probabilities" (probability of a unit flow between two locations). The default is "flows".
            
        Returns
        -------
        FlowDataFrame
            the fluxes generated by the Radiation model.
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
        if out_format not in ['flows', 'flows_sample', 'probabilities']:
            raise ValueError(
                'Value of out_format "%s" is not valid. \nValid values: flows, flows_sample, probabilities.' % out_format)

        # compute the total relevance, i.e., the sum of relevances of all the locations
        total_relevance = np.sum(self.relevances)

        all_flows = []
        for origin in tqdm(range(len(spatial_tessellation))):  # tqdm print a progress bar

            # get the edges for the current origin location
            flows_from_origin = self._get_flows(origin, total_relevance)

            if len(flows_from_origin) > 0:
                all_flows += list(flows_from_origin)

        # Always return a FlowDataFrame
        if True:  # 'flows' in out_format:
            return self._from_matrix_to_flowdf(all_flows, spatial_tessellation)
        else:
            return all_flows

    def _from_matrix_to_flowdf(self, all_flows, spatial_tessellation):
        index2tileid = dict([(i, tileid) for i, tileid in enumerate(spatial_tessellation[self._tile_id_column].values)])
        output_list = [[index2tileid[i], index2tileid[j], flow] for i, j, flow in all_flows if flow > 0.]
        return FlowDataFrame(output_list, origin=0, destination=1, flow=2,
                             tile_id=self._tile_id_column, tessellation=spatial_tessellation)

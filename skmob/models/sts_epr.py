import pandas
import numpy
import scipy
import datetime
import logging
import geopandas
from tqdm import tqdm
from igraph import *
from math import sqrt, sin, cos, pi, asin
from ..utils import utils
from ..core.trajectorydataframe import TrajDataFrame
from ..models.markov_diary_generator import MarkovDiaryGenerator


'''
Implementation of STS-EPR

'''

class STS_epr():
    
    """STS-EPR model.
    
    The STS-EPR (Spatial, Temporal and Social EPR model) model of individual human mobility consists of the following mechanisms [CRP2020]_: 
    
    
    **Action selection**. With probability :math:`P_{exp}=\\rho S^{-\\gamma}`, where :math:`S` is the number of distinct locations previously visited by the agent, 
    the agent visits a new location (Exploration), otherwise with a complementary probability :math:`P_{ret}=1-P{exp}` it returns to a previously visited location (Return). 
    At that point, the agent determines whether or not the location’s choice will be affected by the other agents; with a probability :math:`\\alpha`, the agent’s social contacts influence its
    movement (Social). With a complementary probability of :math:`1-\\alpha`, the agent’s choice is not influenced by the other agents (Individual).

    Parameters :math:`\\rho`, :math:`\\gamma`, and :math:`\\alpha=` correspond to arguments `rho`, `gamma`, and `alpha` of the constructor, respectively.
    
    After the selection of the spatial mechanism (Exploration or Return) and the social mechanism (Individual or Social) 
    decides which location will be the destination of its next displacement during the **Location selection phase**.
    For an agent :math:`a`, we denote the sets containing the indices of the locations :math:`a` can explore or return, as :math:`exp_{a}` and :math:`ret_{a}`, respectively.

    **Individual Exploration**. If the agent :math:`a` is currently in location :math:`i`, and explores a new location without the influence of its social contacts, then the new location :math:`j \\neq i` is an unvisited location for the agent (:math:`i \\in exp_{a}`) 
    and it is selected according to the gravity model with probability proportional to :math:`p_{ij} = \\frac{r_i r_j}{dist_{ij}^2}`, where :math:`r_{i (j)}` is the location's relevance, that is, the probability of a population to visit location :math:`i(j)`, :math:`dist_{ij}` is the geographic distance between :math:`i` and :math:`j`, 
     The number of distinct locations visited, :math:`S`, is increased by 1.

    **Social Exploration**. If the agent :math:`a` is currently in location :math:`i`, and explores a new location with the influence of a social contact, it first selects a social contact :math:`c` 
    with probability :math:`p(c) \\propto mob_{sim}(a,c)` [THSG2015]_. At this point, the agent :math:`a` explores an unvisited location for agent :math:`a` that was visited by agent :math:`c`, i.e., the location :math:`j \\neq i` is selected
    from set :math:`A = exp_a \\cap ret_c`; the probability :math:`p(j)` for a location :math:`j \\in A`, to be selected is proportional to :math:`\Pi_j = f_j`, where :math:`f_j` is the visitation frequency of location :math:`j` for the agent :math:`c`. The number of distinct locations visited, :math:`S`, is increased by 1. 

    **Individual Return**. If the agent :math:`a`, currently at location :math:`i`, returns to a previously visited location :math:`j \\in ret_a`, it is chosen with probability 
    proportional to the number of time the agent visited :math:`j`, i.e., :math:`\Pi_j = f_j`, where :math:`f_j` is the visitation frequency of location :math:`j`.

    **Social Return**. If the agent :math:`a` is currently in location :math:`i`, and returns to a previously visited location with the influence of a social contact, it first selects a social contact :math:`c` 
    with probability :math:`p(c) \\propto mob_{sim}(a,c)` [THSG2015]_. At this point, the agent :math:`a` returns to a previously visited location for agent :math:`a` that was visited by agent :math:`c` too, i.e., the location :math:`j \\neq i` is selected
    from set :math:`A = ret_a \\cap ret_c`; the probability :math:`p(j)` for a location :math:`j \\in A`, to be selected is proportional to :math:`\Pi_j = f_j`, where :math:`f_j` is the visitation frequency of location :math:`j` for the agent :math:`c`.

    
    
    parameters
    ----------
    name : str, optional
        the name of the instantiation of the STS-EPR model. The default value is "STS-EPR".
    rho : float, optional
        it corresponds to the parameter :math:`\\rho \in (0, 1]` in the Action selection mechanism :math:`P_{exp} = \\rho S^{-\gamma}` and controls the agent's tendency to explore a new location during the next move versus returning to a previously visited location. The default value is :math:`\\rho = 0.6` [SKWB2010]_.
    gamma : float, optional
        it corresponds to the parameter :math:`\gamma` (:math:`\gamma \geq 0`) in the Action selection mechanism :math:`P_{exp} = \\rho S^{-\gamma}` and controls the agent's tendency to explore a new location during the next move versus returning to a previously visited location. The default value is :math:`\gamma=0.21` [SKWB2010]_.
    alpha : float, optional
        it corresponds to the parameter `\\alpha` in the Action selection mechanism and controls the influence of the social contacts for an agent during its location selection phase. The default value is :math:`\\alpha=0.2` [THSG2015]_.
    
    
    Attributes
    ----------
    name : str
        the name of the instantiation of the model.
    rho : float
        the input parameter :math:`\\rho`.
    gamma : float
        the input parameters :math:`\gamma`.
    alpha: float
        the input parameter :math:`\\alpha`.     
    
    References
    ----------
    .. [PSRPGB2015] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F. & Barabasi, A. L. (2015) Returners and Explorers dichotomy in human mobility. Nature Communications 6, https://www.nature.com/articles/ncomms9166
    .. [PSR2016] Pappalardo, L., Simini, F. Rinzivillo, S. (2016) Human Mobility Modelling: exploration and preferential return meet the gravity model. Procedia Computer Science 83, https://www.sciencedirect.com/science/article/pii/S1877050916302216
    .. [SKWB2010] Song, C., Koren, T., Wang, P. & Barabasi, A.L. (2010) Modelling the scaling properties of human mobility. Nature Physics 6, 818-823, https://www.nature.com/articles/nphys1760
    .. [THSG2015] Toole, Jameson & Herrera-Yague, Carlos & Schneider, Christian & Gonzalez, Marta C.. (2015). Coupling Human Mobility and Social Ties. Journal of the Royal Society, Interface / the Royal Society. 12. 10.1098/rsif.2014.1128. 
    .. [CRP2020] Cornacchia, Giuliano & Rossetti, Giulio & Pappalardo, Luca. (2020). Modelling Human Mobility considering Spatial,Temporal and Social Dimensions. 
    .. [PS2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4
    
    See Also
    --------
    EPR, SpatialEPR, Ditras
    """

    def __init__(self, name='STS-EPR', rho=0.6, gamma=0.21, alpha=0.2):
        
        self.name = name
        self.rho = rho
        self.gamma = gamma
        self.alpha = alpha
        self.agents = {}
        self.lats_lngs = []
        self.distance_matrix = None
        self.map_uid_gid = None
        
        #dicts for efficient access
        self.dict_uid_to_gid = {}
        self.dict_gid_to_uid = {}
        
        # dict_uid_to_gid and dict_gid_to_uid are used to map the user_id into a graph_id
        # where graph_id is an integer in [0, n_agents) and user_id is the id of the agent
        

    #return the graph_id (user_id) of an agent given its user_id (graph_id)
    def uid_2_gid(self, uid):
        return self.dict_uid_to_gid[uid]
        
    def gid_2_uid(self, gid):
        return self.dict_gid_to_uid[gid]
        
    
    '''
    Location selection methods
            - make_social_action
            - make_individual_return_action
            - make_individual_exploration_action
            
    Notation:
        - exp(x): set containing the indices of the locations x can explore 
        - ret(x): set containing the indices of the locations x can return 
    '''
    
    def make_social_action(self, agent, mode):
        
        '''       
        The agent A makes a social choice in the following way:

        1. The agent A selects a social contact C with probability proportional to the 
        mobility similarity between them
        
        2. The candidate location to visit or explore is selected from the set composed of 
        the locations visited by C (ret(C)), that are feasible according to A's action: 
            - exploration: exp(A) \intersect ret(C)
            - return: ret(A) \intersect ret(C)
        
        3. select one of the feasible locations (if any) with a probability proportional
        to C's visitation frequency
        '''
        
        contact_sim = []
        
        #check and update the mobility similarity of the agent's edges if 'expired'
        for ns in self.social_graph.neighbors(agent):
            eid = self.social_graph.get_eid(agent,ns)

            if self.social_graph.es(eid)['next_update'][0] <= self.current_date:
                #update
                lv1 = self.agents[agent]['location_vector']
                lv2 = self.agents[ns]['location_vector']
                self.social_graph.es(eid)['mobility_similarity'] = self.cosine_similarity(lv1,lv2)
                self.social_graph.es(eid)['next_update'] = self.current_date + datetime.timedelta(hours=self.dt_update_mobSim)

            contact_sim.append(self.social_graph.es(eid)['mobility_similarity'][0])

        contact_sim = numpy.array(contact_sim)

        if len(contact_sim)!=0:
            if numpy.sum(contact_sim)!=0:
                contact_pick = self.random_weighted_choice(contact_sim)                
            else:
                contact_pick = numpy.random.randint(0, len(contact_sim))

            contact = [i for i in self.social_graph.neighbors(agent)][contact_pick]

        else:
            #no contact in the social network, can not make a social choice
            return -1

        # get the location vectors of the agent and contact
        location_vector_agent = self.agents[agent]['location_vector']
        location_vector_contact = self.agents[contact]['location_vector']

        # id_locs_feasible, a vector of indices containing all the agent's feasible location (depend on the mode)
        if mode == 'exploration':
            id_locs_feasible = numpy.where(location_vector_agent==0)[0]
        if mode == 'return':
            id_locs_feasible = numpy.where(location_vector_agent>=1)[0]
        
        # the constraint set is of the form {current_location, starting_location}
        id_locs_constrain_diary = [self.agents[agent]['current_location']]+[self.agents[agent]['home_location']]
        id_locs_feasible = [loc_id for loc_id in id_locs_feasible if loc_id not in id_locs_constrain_diary]
    
        #no location selectable for the agent in the current mode
        if len(id_locs_feasible) == 0:
            return -1

        id_locs_valid = id_locs_feasible
        
        #project v_location with the indices in id_locs_valid
        v_location_proj = [location_vector_contact[i] for i in id_locs_valid]

        if numpy.sum(v_location_proj) != 0:
            #weighted choice
            idx = self.random_weighted_choice(v_location_proj)
            location_id = id_locs_valid[idx]
        else:
            location_id = -1
            
        return location_id
     
        

    def make_individual_return_action(self, agent):    
        
        ''' 
            The agent A makes a preferential choice selecting a VISITED location 
            (i.e., in ret(A)) with probability proportional to the number of visits 
            to that location.   
        '''
        
        v_location = self.agents[agent]['location_vector']
        
        # compute the indices of all the feasible locations for the agent A (the visited ones)
        id_locs_feasible = numpy.where(v_location>=1)[0]
        
        # the constraint set is of the form {current_location, starting_location}
        id_locs_constrain_diary = [self.agents[agent]['current_location']]+[self.agents[agent]['home_location']]
                                                              
        id_locs_feasible = [loc_id for loc_id in id_locs_feasible if loc_id not in id_locs_constrain_diary ]
        #id_locs_valid = id_locs_feasible
        
        if len(id_locs_feasible)==0:
            #no location selectable for the agent in the current mode
            return -1

        #project v_location with the indices in id_locs_valid
        v_location_proj = [v_location[i] for i in id_locs_feasible]
        
        idx = self.random_weighted_choice(v_location_proj)
        location_id = id_locs_feasible[idx]
        
        return location_id
    


    def make_individual_exploration_action(self, agent):      
        '''
            The agent A, current at location i selects an UNVISITED location (i.e., in exp(A))
            j with probability proportional to (r_i * r_j)/ d_ij^2

        '''
        
        v_location = self.agents[agent]['location_vector']   
        
        # compute the indices of all the feasible locations for the agent A (the unvisited ones)
        id_locs_feasible = numpy.where(v_location==0)[0]
        
        id_locs_constrain_diary = [self.agents[agent]['current_location']]+[self.agents[agent]['home_location']]
            
        id_locs_feasible = [loc_id for loc_id in id_locs_feasible if loc_id not in id_locs_constrain_diary ]
                       
        if len(id_locs_feasible) == 0:
            return -1
        
            
        src = self.agents[agent]['current_location']
        self.compute_od_row(src)
        distance_row = numpy.array((self.distance_matrix[src].todense())[0])[0]
        id_locs_valid = id_locs_feasible 
        
        #this is made to avoid d/0 
        distance_row[src]=1
              
        relevance_src = self.relevances[src]
        distance_row_score = numpy.array(1/distance_row**2)
        distance_row_score = distance_row_score * self.relevances * relevance_src
        
        #avoid self return
        distance_row[src]=0

        v_location_proj = numpy.array([distance_row_score[i] for i in id_locs_valid])

        #weighted choice
        idx = self.random_weighted_choice(v_location_proj)
        location_id = id_locs_valid[idx]

        return location_id
            
    

    def random_weighted_choice(self, weights):
        
        probabilities = weights/numpy.sum(weights)
        t =  numpy.random.multinomial(1, probabilities)
        pos_choice = numpy.where(t==1)[0][0]
        
        return pos_choice

        
    '''
        Initialization methods
    '''
    
    
    def init_agents(self):
        self.agents = {}
        
        for i in range(self.n_agents):
            agent = {
                'ID':i,
                'current_location':-1,
                'home_location':-1,
                'location_vector':numpy.array([0]*self.n_locations),
                'S':0,
                'alpha':self.alpha,
                'rho':self.rho,
                'gamma':self.gamma,   
                'time_next_move':self.start_date,
                'dt':0,
                'mobility_diary':None,
                'index_mobility_diary':None
            }
            self.agents[i] = agent
    


    def init_social_graph(self, mode = 'random'):
        
        #generate a random graph
        if isinstance(mode, str):
            if mode == 'random':
                self.social_graph = (Graph.GRG(self.n_agents, 0.5).simplify())
        
        #edge list (src,dest):
        elif isinstance(mode, list):          
            #assuming mode is a list of couple (src,dest)
            user_ids = []
            for edge in mode:
                user_ids.append(edge[0])
                user_ids.append(edge[1])
            
            user_ids = list(set(user_ids))
            graph_ids = numpy.arange(0,len(user_ids))
            
            #update the number of agents n_agents
            self.n_agents = len(user_ids)
            
            #create dicts for efficient access
            self.dict_uid_to_gid = {}
            self.dict_gid_to_uid = {}
            for j in range(len(user_ids)):
                self.dict_uid_to_gid[user_ids[j]]=graph_ids[j]
                self.dict_gid_to_uid[graph_ids[j]]=user_ids[j]
            
            #create an empty Graph and add the vertices
            self.social_graph = Graph()
            self.social_graph.add_vertices(len(user_ids))
            
            #add the edges to the graph          
            for edge in mode:
                uid_src = edge[0]
                uid_dest = edge[1]                      
                gid_src = self.uid_2_gid(uid_src)
                gid_dest = self.uid_2_gid(uid_dest)      
                e = (gid_src,gid_dest)         
                self.social_graph.add_edges([e])
     

              
    def assign_starting_location(self, mode='uniform'):

        #For each agent
        for i in range(self.n_agents):
            if mode == 'uniform':                
                #compute a random location
                rand_location = numpy.random.randint(0, self.n_locations)
            if mode == 'relevance':
                #random choice proportional to relevance
                p_location = self.relevances / numpy.sum(self.relevances)
                t = numpy.random.multinomial(1, p_location)
                rand_location = numpy.where(t==1)[0][0]

                         
            #update the location vector of the user
            self.agents[i]['location_vector'][rand_location] = 1
            #set the number of unique location visited to 1 (home)
            self.agents[i]['S'] = 1
            #update currentLocation
            self.agents[i]['current_location'] = rand_location
            #set the home location
            self.agents[i]['home_location'] = rand_location
            
            #update timeNextMove             
            self.agents[i]['time_next_move'] = self.agents[i]['mobility_diary'].loc[1]['datetime']
            self.agents[i]['index_mobility_diary']= 1
            self.agents[i]['dt'] = 1

            if self.map_ids:
                i = self.gid_2_uid(i)

            lat = self.lats_lngs[rand_location][0]
            lng = self.lats_lngs[rand_location][1]
            self.trajectories.append((i, lat, lng, self.current_date))

       

    def compute_mobility_similarity(self):
        #compute the mobility similarity from every connected pair of agents
        
        for edge in self.social_graph.es:
            lv1 = self.agents[edge.source]['location_vector']
            lv2 = self.agents[edge.target]['location_vector']
            self.social_graph.es(edge.index)['mobility_similarity'] = self.cosine_similarity(lv1,lv2)
            self.social_graph.es(edge.index)['next_update'] = self.current_date + datetime.timedelta(hours=self.dt_update_mobSim)


                          
    def cosine_similarity(self,x,y):
        '''Cosine Similarity (x,y) = <x,y>/(||x||*||y||)'''
        num = numpy.dot(x,y)
        den = numpy.linalg.norm(x)*numpy.linalg.norm(y)
        return num/den
         
            
            
    def store_tmp_movement(self, t, agent, loc, dT):                                                  
        self.tmp_upd.append({'agent':agent, 'timestamp':t, 'location':loc, 'dT':dT})
        

           
    def update_agent_movement_window(self, to):
                                                        
        # take each tuple in tmp_upd and if timestamp <= to update the agent info, namely:
        # S, locationVector, current location, and trajectory
        toRemove=[]
        i=0

        for el in self.tmp_upd:
            if el['timestamp'] <= to:
                agent=int(el['agent'])

                if self.agents[agent]['location_vector'][el['location']] == 0:
                    self.agents[agent]['S']+=1

                self.agents[agent]['location_vector'][el['location']] += 1
                #current location       
                self.agents[agent]['current_location'] = el['location']
                
                if self.map_ids:
                    agent = self.gid_2_uid(agent)
                
                lat = self.lats_lngs[el['location']][0]
                lng = self.lats_lngs[el['location']][1]
                self.trajectories.append((agent, lat, lng, el['timestamp']))
                toRemove.append(i)
                
            i+=1      
        #remove the updated tuples
        toRemove.reverse()

        for ind in toRemove:
            self.tmp_upd.pop(ind)
        
    

    def compute_distance_matrix(self):
        
        self.distance_matrix = numpy.zeros((len(self.spatial_tessellation),len(self.spatial_tessellation)))
        
        for i in range(0,len(self.spatial_tessellation)):
            for j in range(0,len(self.spatial_tessellation)):
                if i != j:
                    d = self.distance_earth_km({'lat':self.lats_lngs[i][0],'lon':self.lats_lngs[i][1]},
                                             {'lat':self.lats_lngs[j][0],'lon':self.lats_lngs[j][1]})
                    self.distance_matrix[i,j] = d
    

    
    def compute_od_row(self, row):
         
        ## if the "row" is already computed do nothing
        ## I test two column, say column 1 and 0: if they are both zero i'am sure that the row has to be computed
        if self.distance_matrix[row,0] != 0 or self.distance_matrix[row,1] != 0:
            return
            
        for i in range(0,len(self.spatial_tessellation)):
                if i != row:
                    d = self.distance_earth_km({'lat':self.lats_lngs[i][0],'lon':self.lats_lngs[i][1]},
                                             {'lat':self.lats_lngs[row][0],'lon':self.lats_lngs[row][1]})
                    self.distance_matrix[row,i] = d

    
             
    def distance_earth_km(self, src, dest):
                
        lat1, lat2 = src['lat']*pi/180, dest['lat']*pi/180
        lon1, lon2 = src['lon']*pi/180, dest['lon']*pi/180
        dlat, dlon = lat1-lat2, lon1-lon2

        ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2.0) ** 2))
        return 6371.01 * ds
    

        
    def init_mobility_diaries(self, hours, start_date):
        #For each agent generate a mobility diary
        for i in range(self.n_agents):          
            diary = self.diary_generator.generate(hours, start_date)            
            #ensure mobility (at least two checkins)
            while len(diary) < 2:
                diary = self.diary_generator.generate(hours, start_date)
                             
            self.agents[i]['mobility_diary'] = diary
         

            
    def get_current_abstract_location_from_diary(self, agent):
            row = self.agents[agent]['index_mobility_diary']
            return self.agents[agent]['mobility_diary'].loc[row]['abstract_location']
    
    

    def confirm_action(self, agent, location_id):
        
        from_ = self.agents[agent]['current_location']
             
        self.agents[agent]['current_location'] = location_id          
        self.agents[agent]['index_mobility_diary']+=1
        
        row_diary = self.agents[agent]['index_mobility_diary'] 
        
        if row_diary < len(self.agents[agent]['mobility_diary']):
            self.agents[agent]['time_next_move'] = self.agents[agent]['mobility_diary'].loc[row_diary]['datetime']
            delta_T = self.agents[agent]['time_next_move']-self.current_date
            dT = delta_T.components[0]*24 + delta_T.components[1]
            next_move = str(self.agents[agent]['time_next_move'])
        else:
            self.agents[agent]['time_next_move'] = self.end_date + datetime.timedelta(hours=1)
            dT = 1
            next_move = "None"
       
        self.agents[agent]['dt'] = dT    
        self.store_tmp_movement(self.current_date, agent, location_id, dT)
        
        return {'from':from_, 'to': location_id, 'next_move':next_move}

        
        
    def action_correction_diary(self, agent, choice):
        
        '''  
        The implementation of the action-correction phase, executed by an agent if
        the location selection phase does not allow movements in any location     
        '''
        corrections=[]
        
        if choice == 'social_return':
            location_id = self.make_individual_return_action(agent)
            corrections.append('individual_return')
            if location_id < 0:
                choice  = 'individual_return'

        elif choice == 'social_exploration':
            location_id = self.make_individual_exploration_action(agent)
            corrections.append('individual_exploration')
            if location_id < 0:
                choice  = 'individual_exploration'

        if choice == 'individual_return':
            location_id = self.make_individual_exploration_action(agent)
            corrections.append('individual_exploration')

        elif choice == 'individual_exploration':
            location_id = self.make_individual_return_action(agent)
            corrections.append('individual_return')
           
        return location_id, corrections
    
        

    def init_spatial_tessellation(self, spatial_tessellation, relevance_column, min_relevance):

        if type(spatial_tessellation) == pandas.DataFrame:
            if len(spatial_tessellation)<3:
                raise ValueError("Argument `spatial_tessellation` must contain at least 3 tiles.")
            self.n_locations = len(spatial_tessellation)
            self.spatial_tessellation = spatial_tessellation
            g=[]
            for i in range(len(spatial_tessellation)):
                lat_ = spatial_tessellation.iloc[i].latitude
                lng_ = spatial_tessellation.iloc[i].longitude
                g.append([lat_,lng_])
            self.lats_lngs = numpy.array(g)

        elif type(spatial_tessellation) == geopandas.GeoDataFrame:
            if len(spatial_tessellation)<3:
                raise ValueError("Argument `spatial_tessellation` must contain at least 3 tiles.")
            self.n_locations = len(spatial_tessellation)
            self.spatial_tessellation = spatial_tessellation
            self.lats_lngs = self.spatial_tessellation.geometry.apply(utils.get_geom_centroid, args=[True]).values
        else:
            raise TypeError("Argument `spatial_tessellation` should be of type pandas.DataFrame or geopandas.GeoDataFrame.")

        if list(self.spatial_tessellation.columns).count(relevance_column) == 0:
            raise IndexError("the column `relevance_columns` is invalid")

        self.relevances = numpy.array(self.spatial_tessellation[relevance_column])

        #map relevance 0 in min_rel               
        self.relevances = numpy.where(self.relevances == 0, min_relevance, self.relevances)



    def init_agents_and_graph(self, social_graph):

        if isinstance(social_graph, str):
            if social_graph == 'random':  
                self.map_ids = False
                self.init_agents()
                self.init_mobility_diaries(self.total_h, self.start_date)
                self.assign_starting_location(mode = self.starting_locations_mode)
                self.init_social_graph(mode = social_graph)
                self.compute_mobility_similarity()
            else:
                raise ValueError("When the argument `social_graph` is a str it must be 'random'.")

        #in this case the parameter n_agents is inferred from the edge list        
        elif isinstance(social_graph, list):
            if len(social_graph)>0:
                self.map_ids = True
                self.init_social_graph(mode = social_graph)         
                self.init_agents()
                self.init_mobility_diaries(self.total_h, self.start_date)
                self.assign_starting_location(mode = self.starting_locations_mode)
                self.compute_mobility_similarity()
            else:
                 raise ValueError("The argument `social_graph` cannot be an empty list.")
        else:
            raise TypeError("Argument `social_graph` should be a string or a list.")
               

      
    def generate(self, start_date, end_date, spatial_tessellation, diary_generator,
                 social_graph='random', n_agents=500, rsl=False, distance_matrix=None, 
                 relevance_column=None, min_relevance = 0.1, dt_update_mobSim = 24*7, 
                 indipendency_window = 0.5, random_state=None, log_file=None, verbose=0,
                 show_progress=False): 

        """
        Start the simulation of a set of agents at time `start_date` till time `end_date`.
        
        Parameters
        ----------
        start_date : datetime
            the starting date of the simulation, in "YYY/mm/dd HH:MM:SS" format.
        end_date : datetime
            the ending date of the simulation, in "YYY/mm/dd HH:MM:SS" format.
        spatial_tessellation : pandas DataFrame or geopandas GeoDataFrame
            the spatial tessellation, i.e., a division of the territory in locations.
        diary_generator : MarkovDiaryGenerator
            the diary generator to use for generating the mobility diary [PS2018]_.
        social_graph : "random" or an edge list
            the social graph describing the sociality of the agents. The default is "random". 
        n_agents : int, optional
            the number of agents to generate. If `social_graph` is "random", `n_agents` are initialized and connected, otherwise the number of agents is inferred from the edge list. The default is 500. 
        rsl: bool, optional
            if Truen the probability :math:`p(i)` for an agent of being assigned to a starting physical location :math:`i` is proportional to the relevance of location :math:`i`; otherwise, if False, it is selected uniformly at random. The defailt is False.
        distance_matrix: numpy array or None, optional
            the origin destination matrix to use for deciding the movements of the agent. If None, it is computed “on the fly” during the simulation. The default is None.
        relevance_column: str, optional
            the name of the column in spatial_tessellation to use as relevance variable. The default is “relevance”.
        min_relevance: float, optional
            the value in which to map the null relevance. The default is 0.1.
        random_state : int or None, optional
            if int, it is the seed used by the random number generator; if None, the random number generator is the RandomState instance used by np.random and random.random. The default is None.  
        dt_update_mobSim: float, optional
            the time interval (in hours) that specifies how often to update the weights of the social graph. The default is 24*7=168 (one week).
        indipendency_window: float, optional
            the time window (in hours) that must elapse before an agent's movements can affect the movements of other agents in the simulation. The default is 0.5.
        log_file : str or None, optional
            the name of the file where to write a log of the execution of the model. The logfile will contain all decisions made by the model. The default is None.
        verbose: int, optional
            the verbosity level of the model relative to the standard output. If `verbose` is equal to 2 the initialization info and the decisions made by the model are printed, if `verbose` is equal to 1 only the initialization info are reported. The default is 0.
        show_progress : boolean, optional
            if True, show a progress bar. The default is False.
        
        Returns
        -------
        TrajDataFrame
            the synthetic trajectories generated by the model
        """

        
        
        # check arguments
        if n_agents<=0:
            raise ValueError("Argument 'n_agents' must be > 0.")
        if start_date > end_date :
            raise ValueError("Argument 'start_date' must be prior to 'end_date'.")
        if type(rsl) != bool:
            raise TypeError("Argument `rsl` must be a bool.")

        
 
        # init data structures and parameters   
        self.n_agents = n_agents
        self.tmp_upd = []
        self.trajectories = []
        self.dt_update_mobSim = dt_update_mobSim
        self.indipendency_window = indipendency_window
        self.verbose = verbose
        self.log_file = log_file

        if rsl:            
            self.starting_locations_mode = 'relevance'
        else:
            self.starting_locations_mode = 'uniform'
        
        self.start_date, self.current_date, self.end_date = start_date, start_date, end_date

        
        # INITIALIZATION
           
        #if specified, fix the random seeds to guarantee the reproducibility of the simulation
        if random_state is not None:
            numpy.random.seed(random_state)
            
        #log_file
        if log_file is not None:
            self._log_file = log_file
            logging.basicConfig(format='%(message)s', filename=log_file, filemode='w', level=logging.INFO)
        
        #Mobility diary generator
        if type(diary_generator) == MarkovDiaryGenerator:
            self.diary_generator = diary_generator
        else:
            raise TypeError("Argument `diary_generator` should be of type skmob.models.markov_diary_generator.MarkovDiaryGenerator.")
        
        #time interval of the simulation 
        delta_T = (self.end_date - self.start_date)
        self.total_h = delta_T.components[0]*24 + delta_T.components[1]
        #init. a progress bar with hourly precision
        if show_progress:
            last_t = self.start_date
            pbar = tqdm(total=self.total_h)        
            elapsed_h = 0
          
        #init. the spatial tessellation
        self.init_spatial_tessellation(spatial_tessellation, relevance_column, min_relevance)    
        
        #distance matrix    
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
            print("Pre-computed matrix")
        else:
            self.distance_matrix = scipy.sparse.lil_matrix((len(self.spatial_tessellation),len(self.spatial_tessellation)))
        
        
        #init. the agents and social graph    
        self.init_agents_and_graph(social_graph)
        
        
        #log init. info
        if self.log_file is not None:
            logging.info("model:\t"+self.name)
            logging.info("time interval:\t["+str(self.start_date)+" - "+str(self.end_date)+"]")
            logging.info("#agents:\t"+str(self.n_agents))
            logging.info("#locations:\t"+str(len(self.spatial_tessellation)))
            logging.info("starting locations:\t"+self.starting_locations_mode)
            if self.map_ids:
                logging.info("social graph:\t argument")
            else:
                logging.info("social graph:\t random")
            logging.info("#edges:\t"+str(len(self.social_graph.es)))
            logging.info("random state:\t"+str(random_state)+"\n\n")
            
        if self.verbose>0:
            print("Model:\t"+self.name)
            print("time interval:\t["+str(self.start_date)+" - "+str(self.end_date)+"]")
            print("#agents:\t"+str(self.n_agents))
            print("#locations:\t"+str(len(self.spatial_tessellation)))
            print("starting locations:\t"+self.starting_locations_mode)
            if self.map_ids:
                print("social graph:\t argument")
            else:
                print("social graph:\t random")
            print("#edges:\t"+str(len(self.social_graph.es)))
            print("random state:\t"+str(random_state)+"\n\n")
        
 
        while self.current_date < self.end_date:
                      
            # we can update all the trajectories made OUTSIDE the indipendence window.       
            sup_indipendency_win = self.current_date - datetime.timedelta(hours=self.indipendency_window)  
            self.update_agent_movement_window(sup_indipendency_win)
            
            min_time_next_move = self.end_date
            
            #for every agent
            #1. Select the Action it will execute (Action Selection phase)
            #2. Select the destination of its next displacement (Location Selection phase)
            #3. If the agent cannot move at any location, the action is corrected (Action Correction phase)
            
            for agent in range(self.n_agents):
                
                location_id = None
                
                #if the user is spending its visiting time do nothing 
                if self.current_date != self.agents[agent]['time_next_move']:
                    if self.agents[agent]['time_next_move'] < min_time_next_move:
                        min_time_next_move = self.agents[agent]['time_next_move']
                    continue
                
                #check the current abstract location, if it is 0 i can skip all the
                #location selection phase and return at the Home Location, otherwise 
                #the abstract location is mapped to a physical one through the standard procedure 
                abstract_location = self.get_current_abstract_location_from_diary(agent)
                
                #home location
                if abstract_location == 0:           
                    location_id = self.agents[agent]['home_location']
                    
                if location_id is None:                 
                    #compute p_exp, the probability that the agent will explore a new location           
                    p_exp = self.agents[agent]['rho'] * (self.agents[agent]['S'] ** -self.agents[agent]['gamma'])

                    #generate a random number for the choice: Explore or Return respectively with probability pS^-gamma and 1-pS^-gamma                
                    p_rand_exp = numpy.random.rand()  

                    #generate a random number for the social or solo choice (alpha, 1-alpha)
                    p_rand_soc = numpy.random.rand()                   
                    
                    p_action = ''
                else:
                    p_action = 'home_return'
                
                
                # ACTION CORRECTION PHASE
                if p_action == '':
                    # compute which action the agent will execute
                    if p_rand_exp < p_exp:
                        if p_rand_soc < self.agents[agent]['alpha']:
                            choice = 'social_exploration'
                        else:
                            choice = 'individual_exploration'
                    else:
                        if p_rand_soc < self.agents[agent]['alpha']:
                            choice = 'social_return'
                        else:
                            choice = 'individual_return'                                      
                else:
                    choice = p_action
                
                # LOCATION SELECTION PHASE
                if choice == 'social_exploration':               
                    location_id = self.make_social_action(agent, 'exploration')
 
                elif choice == 'individual_exploration':
                    location_id = self.make_individual_exploration_action(agent)

                elif choice == 'social_return':
                    location_id = self.make_social_action(agent, 'return')

                elif choice == 'individual_return':
                    location_id = self.make_individual_return_action(agent)
                        
                #ACTION CORRECTION PHASE
                # -1 means no selectable location
                
                corrections=None
                if location_id == -1:
                    location_id, corrections = self.action_correction_diary(agent, choice)                                                                    
                   
                if location_id >= 0:
                    info_move = self.confirm_action(agent, location_id)

                    if self.log_file is not None:
                        logging.info("Agent "+str(agent))
                        logging.info("Moved from loc. "+str(info_move['from'])+" to loc. " 
                              +str(info_move['to'])+" at timestamp "
                              +str(self.current_date))
                        logging.info("Action: "+choice)
                        if corrections is None:
                            logging.info("Corrections: None")
                        else:
                            str_corr = choice
                            for corr in corrections:
                                str_corr+=" -> "+corr
                            logging.info("Corrections: "+str_corr)
                        logging.info("Next move: "+str(info_move['next_move'])+"\n")
                    
                    if self.verbose>1:
                        print("Agent "+str(agent))
                        print("Moved from loc. "+str(info_move['from'])+" to loc. " 
                              +str(info_move['to'])+" at timestamp "
                              +str(self.current_date))
                        print("Action: "+choice)
                        if corrections is None:
                            print("Corrections: None")
                        else:
                            str_corr = choice
                            for corr in corrections:
                                str_corr+=" -> "+corr
                            print("Corrections: "+str_corr)
                        print("Next move: "+str(info_move['next_move'])+"\n")
                else:
                    #this should never happen, since n_loc>2
                    raise Exception("Fatal error, unable to correct the location") 
                        
                if self.agents[agent]['time_next_move']< min_time_next_move:
                        min_time_next_move = self.agents[agent]['time_next_move']
                        
                
            self.current_date = min_time_next_move
            
            if show_progress:                
                dT2 = self.current_date - last_t   
                if(dT2.components[0]!=0 or dT2.components[1]!=0):
                    pbar.update(dT2.components[0]*24 + dT2.components[1])
                    last_t = self.current_date
                    elapsed_h += dT2.components[0]*24 + dT2.components[1]
        
        if show_progress:
            pbar.update(self.total_h - elapsed_h)
            pbar.close()
        
        if self.log_file is not None:
            logging.shutdown()
        
        self.update_agent_movement_window(self.end_date)
        tdf = TrajDataFrame(self.trajectories, user_id=0, latitude=1, longitude=2, datetime=3)
        tdf = tdf.sort_by_uid_and_datetime() 
        return tdf
        
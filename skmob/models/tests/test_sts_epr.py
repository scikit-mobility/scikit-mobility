from ...utils import constants
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import pytest
from contextlib import ExitStack
from sklearn.metrics import mean_absolute_error
from ...preprocessing import detection, clustering
from ...models.sts_epr import STS_epr
from ...core.trajectorydataframe import TrajDataFrame
from ...models.markov_diary_generator import MarkovDiaryGenerator



def global_variables():
    # tessellation
    tess_polygons = [[[7.481, 45.184],
                      [7.481, 45.216],
                      [7.526, 45.216],
                      [7.526, 45.184],
                      [7.481, 45.184]],
                     [[7.481, 45.216],
                      [7.481, 45.247],
                      [7.526, 45.247],
                      [7.526, 45.216],
                      [7.481, 45.216]],
                     [[7.526, 45.184],
                      [7.526, 45.216],
                      [7.571, 45.216],
                      [7.571, 45.184],
                      [7.526, 45.184]],
                     [[7.526, 45.216],
                      [7.526, 45.247],
                      [7.571, 45.247],
                      [7.571, 45.216],
                      [7.526, 45.216]]]

    geom = [shapely.geometry.Polygon(p) for p in tess_polygons]
    tessellation = gpd.GeoDataFrame(geometry=geom, crs="EPSG:4326")
    tessellation = tessellation.reset_index().rename(columns={"index": constants.TILE_ID})
    relevance = np.random.randint(5, 10, size=len(tessellation))
    tessellation[constants.RELEVANCE] = relevance
    
    social_graph = [[0,1],[0,2],[0,3],[1,3],[2,4]]
    
    # mobility diary generator
    lats_lngs = np.array([[39.978253, 116.3272755],
                      [40.013819, 116.306532],
                      [39.878987, 116.1266865],
                      [40.013819, 116.306532],
                      [39.97958, 116.313649],
                      [39.978696, 116.3262205],
                      [39.98153775, 116.31079],
                      [39.978161, 116.3272425],
                      [38.978161, 115.3272425]])
    traj = pd.DataFrame(lats_lngs, columns=[constants.LATITUDE, constants.LONGITUDE])
    traj[constants.DATETIME] = pd.to_datetime([
        '20130101 8:34:04', '20130101 10:34:08', '20130105 10:34:08',
        '20130110 12:34:15', '20130101 1:34:28', '20130101 3:34:54',
        '20130101 4:34:55', '20130105 5:29:12', '20130115 00:29:12'])
    traj[constants.UID] = [1 for _ in range(5)] + [2 for _ in range(3)] + [3]
    tdf = TrajDataFrame(traj)
    ctdf = clustering.cluster(tdf)
    mdg = MarkovDiaryGenerator()
    mdg.fit(ctdf, 3, lid='cluster')
    
    return tessellation, social_graph, mdg

tessellation, social_graph, mdg = global_variables()



sts_epr = STS_epr()
    



@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', [social_graph, 'random'])
@pytest.mark.parametrize('n_agents', [1,5])
@pytest.mark.parametrize('rsl', [True, False])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])


# First test set: CORRECT arguments, no ERRORS expected (#test: 8)

def test_sts_generate_success(start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)



    assert isinstance(tdf, TrajDataFrame)




# Second test set: WRONG arguments, expected to FAIL

# test 2.1: wrong n_agents (#test: 3)

@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', [social_graph])
@pytest.mark.parametrize('n_agents', [-2,-1,0])
@pytest.mark.parametrize('rsl', [True])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=ValueError)

def test_sts_wrong_n_agents(start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)



    assert isinstance(tdf, TrajDataFrame)



# test 2.2: end_date prior to start_date (#test: 1)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', [social_graph])
@pytest.mark.parametrize('n_agents', [5])
@pytest.mark.parametrize('rsl', [True])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=ValueError)

def test_sts_wrong_dates(start_date, end_date, spatial_tessellation, diary_generator,
                          social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)



# test 2.3: wrong rsl type (#test: 3)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', [social_graph])
@pytest.mark.parametrize('n_agents', [5])
@pytest.mark.parametrize('rsl', [1, None, 'True'])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=TypeError)


def test_sts_wrong_rsl_type(start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)





# test 2.4: wrong type for the spatial_tessellation (#test: 5)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', ["", None, [], "tessellation", [1,2,3]])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', [social_graph])
@pytest.mark.parametrize('n_agents', [5])
@pytest.mark.parametrize('rsl', [True])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=TypeError)


def test_sts_wrong_tex_type(start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)




# test 2.5: # of tiles in spatial_tessellation < 3 (#test: 3)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [pd.DataFrame(),tessellation[:1],tessellation[:2]])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', [social_graph])
@pytest.mark.parametrize('n_agents', [5])
@pytest.mark.parametrize('rsl', [True])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=ValueError)


def test_sts_wrong_tiles_num(start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)



# test 2.6: wrong relevance's column name (#test: 1)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', [social_graph])
@pytest.mark.parametrize('n_agents', [5])
@pytest.mark.parametrize('rsl', [True,])
@pytest.mark.parametrize('relevance_column',['rel'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=IndexError)
 
def test_sts_wrong_relevance_col_name(start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)


# test 2.7: wrong type for the diary_generator (#test: 3)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [[],None,pd.DataFrame()])
@pytest.mark.parametrize('social_graph', [social_graph])
@pytest.mark.parametrize('n_agents', [5])
@pytest.mark.parametrize('rsl', [True,])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=TypeError)
 
def test_sts_wrong_relevance_col_name(start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)





# Third test set: assert the correctness of the model's functions

def all_equal(a, b, threshold=1e-3):
    return mean_absolute_error(a, b) <= threshold

def correcteness_set_exp(a,b):
    for i in range(len(b)):
        if a[i]>0 and b[i]>0:
            return False
    return True

def correcteness_set_ret(a, b):
    for i in range(len(b)):
        if b[i]>0 and a[i]==0:
            return False
    return True

def correcteness_set_exp_social(lva,lvc,choices):
    for i in range(len(choices)):
        if choices[i]>0:
            if not (lva[i]==0 and lvc[i]>0):
                return False
    return True

def correcteness_set_ret_social(lva,lvc,choices):
    for i in range(len(choices)):
        if choices[i]>0:
            if not (lva[i]>0 and lvc[i]>0):
                return False
    return True

# test 3.1: correct random_weighted_choice (#test: 1)

@pytest.mark.parametrize('size', [1000])
@pytest.mark.parametrize('n_picks', [int(1e4)])


def test_weighted_random_choice(size,n_picks):

    np.random.seed(24)
    sts_epr = STS_epr()

    weigths = np.random.randint(0, 10, size=size)

    theoretical  = weigths/np.sum(weigths)
    empirical = [0]*len(weigths)

    for j in range(n_picks):
      i = sts_epr.random_weighted_choice(weigths)
      empirical[i]+=1

    empirical = empirical/np.sum(empirical)

    assert(all_equal(theoretical,empirical))



# test 3.2: correct exploration choices (#test: 1)

# create a fake location vector of size n for the agent A (id=0)
# m elements = 0 and j elements > 0, m+j=n

# RETURN
@pytest.mark.parametrize('m', [3])
@pytest.mark.parametrize('j', [1])
@pytest.mark.parametrize('n_picks', [int(1e4)])
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/03/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', ['random'])
@pytest.mark.parametrize('n_agents', [2])
@pytest.mark.parametrize('rsl', [True])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])


def test_correctness_exp(m, j, n_picks, start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)

    np.random.seed(random_state)

    
    # create a fake location vector of size n for the agent A (id=0)
    # m elements = 0 and j elements > 0, m+j=n
    location_vector = [0]*m + list(np.random.randint(5, 10, size=j-1))

    np.random.shuffle(location_vector)

    #diary constraint = {home, last_location}, 
    #fix as an example both home and current loc to 0
    location_vector = [5]+location_vector

    choices = [0]*len(location_vector)

    #assign this location vector to agent with id=0
    sts_epr.agents[0]['location_vector'] = np.array(location_vector)
    #assign the home location 0
    sts_epr.agents[0]['home_location'] = 0
    #assign current location 1
    sts_epr.agents[0]['current_location'] = 0

    sts_epr.compute_od_row(0)
    v_dist = np.array((sts_epr.distance_matrix[0].todense())[0])[0]
    v_rel = sts_epr.relevances


    for j in range(n_picks):
        location_id = sts_epr.make_individual_exploration_action(0)
        choices[location_id]+=1
        

    #test 1 correctness of the choices; i.e., no location j s.t. lv[j]>0
    res_1 = correcteness_set_exp(location_vector,choices)

    #test 2 correct probabilities
    empirical = choices/np.sum(choices)
    #avoid division by 0
    v_dist[0]=1
    theoretical =  np.array(1/v_dist**2) * v_rel * v_rel[0]
    theoretical[0] = 0
    theoretical = theoretical/np.sum(theoretical)

    res_2 = all_equal(theoretical,empirical,threshold=1e-2)

    assert((res_1,res_2)==(True,True))



# test 3.3: correct return choices (#test: 1)

# create a fake location vector of size n for the agent A (id=0)
# m elements = 0 and j elements > 0, m+j=n

# RETURN
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('j', [500])
@pytest.mark.parametrize('n_picks', [int(1e4)])
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/03/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', ['random'])
@pytest.mark.parametrize('n_agents', [2])
@pytest.mark.parametrize('rsl', [True])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])


def test_correctness_ret(m, j, n_picks, start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)

    np.random.seed(random_state)

    
    # create a fake location vector of size n for the agent A (id=0)
    # m elements = 0 and j elements > 0, m+j=n
    location_vector = [0]*m + list(np.random.randint(5, 10, size=j-2))
    np.random.shuffle(location_vector)

    #diary constraint = {home, last_location}, fix as an example loc 0 and 1
    location_vector = [4,1]+location_vector

    choices = [0]*len(location_vector)

    #assign this location vector to agent with id=0
    sts_epr.agents[0]['location_vector'] = np.array(location_vector)
    #assign the home location 0
    sts_epr.agents[0]['home_location'] = 0
    #assign current location 1
    sts_epr.agents[0]['current_location'] = 1


    for j in range(n_picks):
      location_id = sts_epr.make_individual_return_action(0)
      choices[location_id]+=1

      
    #test 1 correctness of the choices; i.e., no location j s.t. lv[j]=0

    # we can consider the home and current location as unvisited to prove
    # the correctness of the choices
    location_vector[0]=0
    location_vector[1]=0

    res_1 = correcteness_set_ret(location_vector,choices)

    #test 2 correct probabilities
    empirical = choices/np.sum(choices)
    theoretical = location_vector/np.sum(location_vector)
    res_2 = all_equal(theoretical,empirical)


    assert((res_1,res_2)==(True,True))
    




# test 3.4: correct exp social choices (#test: 1)

# create a fake location vector of size n for the agent A (id=0) and agent C (id=1)
# agent A and C are connected in the social graph
# m elements = 0 and j elements > 0, m+j=n

# EXP SOCIAL
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('j', [500])
@pytest.mark.parametrize('n_picks', [int(1e4)])
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/02/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', [[(0,1)]])
@pytest.mark.parametrize('n_agents', [2])
@pytest.mark.parametrize('rsl', [True])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])


def test_correctness_exp_social(m, j, n_picks, start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)

    
    np.random.seed(random_state)
    
    # agent A (id=0)
    location_vector_a = [0]*(m-2) + list(np.random.randint(5, 10, size=j-2))
    np.random.shuffle(location_vector_a)
    location_vector_a = [5,2] + location_vector_a + [0]*2

    choices = [0]*len(location_vector_a)

    #assign the home location 0
    sts_epr.agents[0]['home_location'] = 0
    #assign current location 1
    sts_epr.agents[0]['current_location'] = 1
    #assign this location vector to agent with id=0
    sts_epr.agents[0]['location_vector']=np.array(location_vector_a)

    # agent C (id=1)
    location_vector_c = [0]*(m) + list(np.random.randint(5, 10, size=j-2))
    np.random.shuffle(location_vector_c)
    location_vector_c = location_vector_c + [5,3]

    #assign this location vector to agent with id=1
    sts_epr.agents[1]['location_vector']=np.array(location_vector_c)


    for j in range(n_picks):
        location_id = sts_epr.make_social_action(0, 'exploration')
        choices[location_id]+=1

     
    #test 1 correctness of the choices;
    res_1 = correcteness_set_exp_social(location_vector_a,location_vector_c,choices)

    #test 2 correct probabilities
    empirical = choices/np.sum(choices)
    set_c = [location_vector_c[i] if (location_vector_a[i]==0 and location_vector_c[i]>0) else 0 for
          i in range(len(location_vector_a))]
    theoretical = set_c/np.sum(set_c)
    res_2 = all_equal(theoretical,empirical)

    assert((res_1,res_2)==(True,True))
    




# test 3.5: correct ret social choices (#test: 1)

# create a fake location vector of size n for the agent A (id=0) and agent C (id=1)
# agent A and C are connected in the social graph
# m elements = 0 and j elements > 0, m+j=n

# RET SOCIAL
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('j', [500])
@pytest.mark.parametrize('n_picks', [int(1e4)])
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/02/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('diary_generator', [mdg])
@pytest.mark.parametrize('social_graph', [[(0,1)]])
@pytest.mark.parametrize('n_agents', [2])
@pytest.mark.parametrize('rsl', [True])
@pytest.mark.parametrize('relevance_column',['relevance'])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])


def test_correctness_ret_social(m, j, n_picks, start_date, end_date, spatial_tessellation, diary_generator,
                              social_graph, n_agents, rsl, relevance_column, random_state, show_progress):

    sts_epr = STS_epr()
    tdf = sts_epr.generate(start_date=start_date, end_date=end_date, spatial_tessellation=spatial_tessellation, 
                            social_graph=social_graph, diary_generator=diary_generator, n_agents= n_agents, rsl=rsl, 
                            relevance_column=relevance_column, random_state=random_state, show_progress=show_progress)

    np.random.seed(random_state)

    
    # agent A (id=0)
    location_vector_a = [0]*(m-2) + list(np.random.randint(5, 10, size=j-2))
    np.random.shuffle(location_vector_a)
    location_vector_a = [5,2] + location_vector_a + [0]*2

    choices = [0]*len(location_vector_a)

    #assign the home location 0
    sts_epr.agents[0]['home_location'] = 0
    #assign current location 1
    sts_epr.agents[0]['current_location'] = 1
    #assign this location vector to agent with id=0
    sts_epr.agents[0]['location_vector']=np.array(location_vector_a)

    # agent C (id=1)
    location_vector_c = [0]*(m) + list(np.random.randint(5, 10, size=j-2))
    np.random.shuffle(location_vector_c)
    location_vector_c = location_vector_c + [5,3]

    #assign this location vector to agent with id=1
    sts_epr.agents[1]['location_vector']=np.array(location_vector_c)


    for j in range(n_picks):
        location_id = sts_epr.make_social_action(0, 'return')
        choices[location_id]+=1

    # we can consider the home and current location as unvisited to prove
    # the correctness of the choices
    location_vector_a[0]=0
    location_vector_a[1]=0

    #test 1 correctness of the choices;
    res_1 = correcteness_set_ret_social(location_vector_a,location_vector_c,choices)

    #test 2 correct probabilities
    empirical = choices/np.sum(choices)
    set_c = [location_vector_c[i] if (location_vector_a[i]>0 and location_vector_c[i]>0) else 0 for
          i in range(len(location_vector_a))]
    theoretical = set_c/np.sum(set_c)
    res_2 = all_equal(theoretical,empirical)

    assert((res_1,res_2)==(True,True))
 
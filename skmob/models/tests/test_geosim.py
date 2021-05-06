from ...utils import constants
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import pytest
from contextlib import ExitStack
from sklearn.metrics import mean_absolute_error
from ...models.geosim import GeoSim
from ...core.trajectorydataframe import TrajDataFrame

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
    
    social_graph = [[0,1],[0,2],[0,3],[1,3],[2,4]]
    
    return tessellation, social_graph

tessellation, social_graph = global_variables()



@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('social_graph', [social_graph, 'random'])
@pytest.mark.parametrize('n_agents', [1,5])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])


# First test set: CORRECT arguments, no ERRORS expected (#test: 4)

def test_geosim_generate_success(start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)

    assert isinstance(tdf, TrajDataFrame)




# Second test set: WRONG arguments, expected to FAIL

# test 2.1: wrong n_agents (#test: 3)

@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('social_graph', ['random'])
@pytest.mark.parametrize('n_agents', [-2,-1,0])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=ValueError)

def test_geosim_wrong_n_agents(start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)


# test 2.2: end_date prior to start_date (#test: 1)

@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('social_graph', ['random'])
@pytest.mark.parametrize('n_agents', [5])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=ValueError)

def test_geosim_wrong_dates(start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)


# test 2.3: wrong type for the spatial_tessellation (#test: 5)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', ["", None, [], "tessellation", [1,2,3]])
@pytest.mark.parametrize('social_graph', ['random'])
@pytest.mark.parametrize('n_agents', [5])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=TypeError)

def test_geosim_wrong_tex_type(start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)


# test 2.4: #of tiles in spatial_tessellation < 2 (#test: 2)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [pd.DataFrame(),tessellation[:1]])
@pytest.mark.parametrize('social_graph', ['random'])
@pytest.mark.parametrize('n_agents', [5])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=ValueError)

def test_geosim_wrong_tiles_num(start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)


# test 2.5: wrong social_graph type (#test: 3)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('social_graph', [None, False, 24])
@pytest.mark.parametrize('n_agents', [1,5])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=TypeError)

def test_geosim_wrong_social_graph_type(start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)




# test 2.5: correct social_graph type with wrong value (#test: 2)
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('social_graph', ['xyz', []])
@pytest.mark.parametrize('n_agents', [1,5])
@pytest.mark.parametrize('random_state', [2])
@pytest.mark.parametrize('show_progress', [True])
@pytest.mark.xfail(raises=ValueError)

def test_geosim_wrong_social_graph_value(start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)



# Third test set: assert the correctness of the model's functions

def all_equal(a, b, threshold=1e-3):
    return mean_absolute_error(a, b) <= threshold


#a is the location_vector, and b represents che generated choices
def correcteness_set_exp(a,b):
    for i in range(len(b)):
        if a[i]>0 and b[i]>0:
            return False
    return True

def correcteness_set_ret(a,b):
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
    geosim = GeoSim()

    weigths = np.random.randint(0, 10, size=size)

    theoretical  = weigths/np.sum(weigths)
    empirical = [0]*len(weigths)

    for j in range(n_picks):
      i = geosim.random_weighted_choice(weigths)
      empirical[i]+=1

    empirical = empirical/np.sum(empirical)

    assert(all_equal(theoretical,empirical))


# test 3.2: correct exploration choices (#test: 1)

# create a fake location vector of size n for the agent A (id=0)
# m elements = 0 and j elements > 0, m+j=n

# EXPLORATION (in GeoSim uniformly at random)
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('j', [500])
@pytest.mark.parametrize('n_picks', [int(1e4)])
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('social_graph', ['random'])
@pytest.mark.parametrize('n_agents', [2])
@pytest.mark.parametrize('random_state', [24])
@pytest.mark.parametrize('show_progress', [True])


def test_correctness_exp(m, j, n_picks, start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)

    np.random.seed(random_state)

    # create a fake location vector of size n for the agent A (id=0)
    # m elements = 0 and j elements > 0, m+j=n
    location_vector = [0]*m + list(np.random.randint(5, 10, size=j))
    choices = [0]*len(location_vector)
    np.random.shuffle(location_vector)

    #assign this location vector to agent with id=0
    geosim.agents[0]['location_vector']=np.array(location_vector)

    for j in range(n_picks):
        location_id = geosim.make_individual_exploration_action(0)
        choices[location_id]+=1

    #test 1 correctness of the choices; i.e., no location j s.t. lv[j]>0
    res_1 = correcteness_set_exp(location_vector,choices)

    #test 2 correct probabilities
    empirical = choices/np.sum(choices)
    theoretical = [1/m if location_vector[i]==0 else 0 for i in range(len(location_vector))]
    res_2 = all_equal(theoretical,empirical)

    assert((res_1,res_2)==(True,True))


# test 3.3: correct return choices (#test: 1)

# create a fake location vector of size n for the agent A (id=0)
# m elements = 0 and j elements > 0, m+j=n

# RETURN (prop. to number of visits)
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('j', [500])
@pytest.mark.parametrize('n_picks', [int(1e4)])
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('social_graph', ['random'])
@pytest.mark.parametrize('n_agents', [2])
@pytest.mark.parametrize('random_state', [24])
@pytest.mark.parametrize('show_progress', [True])


def test_correctness_ret(m, j, n_picks, start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)

    np.random.seed(random_state)

    # create a fake location vector of size n for the agent A (id=0)
    # m elements = 0 and j elements > 0, m+j=n
    location_vector = [0]*m + list(np.random.randint(5, 10, size=j))
    choices = [0]*len(location_vector)
    np.random.shuffle(location_vector)

    #assign this location vector to agent with id=0
    geosim.agents[0]['location_vector']=np.array(location_vector)

    for j in range(n_picks):
        location_id = geosim.make_individual_return_action(0)
        choices[location_id]+=1

    #test 1 correctness of the choices; i.e., no location j s.t. lv[j]=0
    res_1 = correcteness_set_ret(location_vector,choices)

    #test 2 correct probabilities
    empirical = choices/np.sum(choices)
    theoretical = location_vector/np.sum(location_vector)
    res_2 = all_equal(theoretical,empirical)

    assert((res_1,res_2)==(True,True))





# test 3.4: correct social exploration choices (#test: 1)

# create a fake location vector of size n for the agent A (id=0) and agent C (id=1)
# agent A and C are connected in the social graph
# m elements = 0 and j elements > 0, m+j=n

# SOCIAL EXPLORATION (prop. to number of visits of a social contact)
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('j', [500])
@pytest.mark.parametrize('n_picks', [int(1e4)])
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('social_graph', [[(1,2)]])
@pytest.mark.parametrize('n_agents', [2])
@pytest.mark.parametrize('random_state', [24])
@pytest.mark.parametrize('show_progress', [True])


def test_correctness_exp_social(m, j, n_picks, start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)


    # agent A (id=0)
    location_vector_a = [0]*(m-2) + list(np.random.randint(5, 10, size=j))
    np.random.shuffle(location_vector_a)
    location_vector_a = location_vector_a + [0]*2

    choices = [0]*len(location_vector_a)

    #assign this location vector to agent with id=0
    geosim.agents[0]['location_vector']=np.array(location_vector_a)

    # agent C (id=1)
    location_vector_c = [0]*(m) + list(np.random.randint(5, 10, size=j-2))
    np.random.shuffle(location_vector_c)
    location_vector_c = location_vector_c + [5,3]

    #assign this location vector to agent with id=1
    geosim.agents[1]['location_vector']=np.array(location_vector_c)


    for j in range(n_picks):
        location_id = geosim.make_social_action(0, 'exploration')
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






# test 3.5: correct social return choices (#test: 1)

# create a fake location vector of size n for the agent A (id=0) and agent C (id=1)
# agent A and C are connected in the social graph
# m elements = 0 and j elements > 0, m+j=n

# SOCIAL RETURN (prop. to number of visits of a social contact)
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('j', [500])
@pytest.mark.parametrize('n_picks', [int(1e4)])
@pytest.mark.parametrize('start_date', [pd.to_datetime('2020/01/01 08:00:00')])
@pytest.mark.parametrize('end_date', [pd.to_datetime('2020/01/10 08:00:00')])
@pytest.mark.parametrize('spatial_tessellation', [tessellation])
@pytest.mark.parametrize('social_graph', [[(1,2)]])
@pytest.mark.parametrize('n_agents', [2])
@pytest.mark.parametrize('random_state', [24])
@pytest.mark.parametrize('show_progress', [True])


def test_correctness_ret_social(m, j, n_picks, start_date, end_date, spatial_tessellation,
                      social_graph, n_agents, random_state, show_progress):

    geosim = GeoSim()
    tdf = geosim.generate(start_date, end_date, social_graph=social_graph, 
                          spatial_tessellation=spatial_tessellation,
                            n_agents = n_agents, random_state=random_state, 
                            show_progress=show_progress)


    # agent A (id=0)
    location_vector_a = [0]*m + list(np.random.randint(5, 10, size=j-2))
    np.random.shuffle(location_vector_a)
    location_vector_a = location_vector_a + [1,1]

    choices = [0]*len(location_vector_a)

    #assign this location vector to agent with id=0
    geosim.agents[0]['location_vector']=np.array(location_vector_a)

    # agent C (id=1)
    location_vector_c = [0]*(m) + list(np.random.randint(5, 10, size=j-2))
    np.random.shuffle(location_vector_c)
    location_vector_c = location_vector_c + [5,3]

    #assign this location vector to agent with id=1
    geosim.agents[1]['location_vector']=np.array(location_vector_c)


    for j in range(n_picks):
        location_id = geosim.make_social_action(0, 'return')
        choices[location_id]+=1


    #test 1 correctness of the choices;
    res_1 = correcteness_set_ret_social(location_vector_a,location_vector_c,choices)

    #test 2 correct probabilities
    empirical = choices/np.sum(choices)
    set_c = [location_vector_c[i] if (location_vector_a[i]>0 and location_vector_c[i]>0) else 0 for
          i in range(len(location_vector_a))]
    theoretical = set_c/np.sum(set_c)
    res_2 = all_equal(theoretical,empirical)

    assert((res_1,res_2)==(True,True))
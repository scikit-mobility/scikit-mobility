from home_detection import homelocation_ma, homelocation_tc, homelocation_dd, homelocation_inactivity
import numpy as np
import pytest
import pandas as pd
import math

WEEK_DAYS = 'WK'
WEEKEND_DAYS = 'WE'


lats_lngs = np.array([
    [43.7206, 10.4026],
    [43.7166, 10.3989],
    [43.7166, 10.3989],
    [43.7441, 10.3724],
    [43.7206, 10.4026],
    [43.7206, 10.4026],
    [43.7206, 10.4026],
    [43.7106, 10.4121],
    [43.6932, 10.4003],
    [43.7064, 10.3858],
    [43.7197, 10.4005],
    [43.7189, 10.4191],
    [43.7083, 10.3984],
    [43.7185, 10.4031],
    [43.7166, 10.3989],
    [43.7166, 10.3989],
    [43.7174, 10.4203],
    [43.7185, 10.4031],
    [43.7206, 10.4026],
    [43.7441, 10.3724],
    [43.6979, 10.3884],
    [43.7083, 10.3984],
    [43.7045, 10.4331],
    [43.7045, 10.4331],
    [43.7155, 10.4017],
    [43.7113, 10.3986],
    [43.7254, 10.4499],
    [43.7006, 10.4078],
    [43.7033, 10.3992],
    [43.7112, 10.4277],
    [43.7131, 10.4078],
    [43.7062, 10.4433],
    [43.7212, 10.3899],
    [43.7112, 10.4277],
    [43.7149, 10.427],
    [43.6861, 10.4319]
])

traj = pd.DataFrame(lats_lngs, columns=['lat', 'lng'])

traj['datetime'] = pd.to_datetime([
    '20220103 03:00:00',
    '20221202 09:05:00',
    '20221201 17:11:00',
    '20220327 17:11:00',
    '20220322 23:55:00',
    '20220322 23:20:00',
    '20220322 06:32:00',
    '20220321 18:39:00',
    '20220317 06:20:00',
    '20220316 21:39:00',
    '20220316 16:38:00',
    '20220313 23:34:00',
    '20220306 05:40:00',
    '20220301 20:05:00',
    '20221203 14:36:00',
    '20221204 16:53:00',
    '20220120 01:11:00',
    '20220103 14:33:00',
    '20220106 22:32:00',
    '20220109 18:30:00',
    '20220126 07:00:00',
    '20220119 08:31:00',
    '20220104 09:19:00',
    '20220220 12:05:00',
    '20220316 15:24:00',
    '20220214 12:05:00',
    '20220214 16:13:00',
    '20220214 23:00:00',
    '20220120 08:27:00',
    '20220220 03:41:00',
    '20220311 05:12:00',
    '20220107 06:36:00',
    '20220324 07:08:00',
    '20220106 21:28:00',
    '20220120 16:00:00',
    '20220220 18:58:00'
])


traj['uid'] = [1 for _ in range(22)] + [2 for _ in range(6)] + [3 for _ in range(8)]


@pytest.mark.parametrize('traj', [traj])
def test_homelocation_ma(traj):
    ###################### Most Amount | No Radius ######################
    output = homelocation_ma(traj, week_period=None, radius=None, show_progress=False)

    assert(len(output) == 3)
    assert(isinstance(output, pd.core.frame.DataFrame))

    assert(output[output.uid == 1]['lat'].values[0] == 43.7206)
    assert(output[output.uid == 2]['lat'].values[0] == 43.7045)
    assert(output[output.uid == 3]['lat'].values[0] == 43.7112)

    assert(output[output.uid == 1]['lng'].values[0] == 10.4026)
    assert(output[output.uid == 2]['lng'].values[0] == 10.4331)
    assert(output[output.uid == 3]['lng'].values[0] == 10.4277)
    
    
    ###################### Most Amount | Period: Week Days | No Radius ######################
    output = homelocation_ma(traj, week_period=WEEK_DAYS, radius=None, show_progress=False)

    assert(len(output) == 3)
    assert(isinstance(output, pd.core.frame.DataFrame))

    assert(output[output.uid == 1]['lat'].values[0] == 43.7206)
    assert(output[output.uid == 2]['lat'].values[0] == 43.7006)
    assert(output[output.uid == 3]['lat'].values[0] == 43.7033)

    assert(output[output.uid == 1]['lng'].values[0] == 10.4026)
    assert(output[output.uid == 2]['lng'].values[0] == 10.4078)
    assert(output[output.uid == 3]['lng'].values[0] == 10.3992)


    ###################### Most Amount | Period: Weekend Days | No Radius ######################
    output = homelocation_ma(traj, week_period=WEEKEND_DAYS, radius=None, show_progress=False)

    assert(len(output) == 3)
    assert(isinstance(output, pd.core.frame.DataFrame))

    assert(output[output.uid == 1]['lat'].values[0] == 43.7166)
    assert(output[output.uid == 2]['lat'].values[0] == 43.7045)
    assert(output[output.uid == 3]['lat'].values[0] == 43.6861)

    assert(output[output.uid == 1]['lng'].values[0] == 10.3989)
    assert(output[output.uid == 2]['lng'].values[0] == 10.4331)
    assert(output[output.uid == 3]['lng'].values[0] == 10.4319)


    ###################### Most Amount | Radius: 1 | Matrix Distance ######################
    matrix_output = homelocation_ma(traj, week_period=None, radius=1, mode='distance_matrix', show_progress=False)

    assert(len(matrix_output) == 3)
    assert(isinstance(matrix_output, pd.core.frame.DataFrame))

    assert(matrix_output[matrix_output.uid == 1]['lat'].values[0] == 43.7166)
    assert(matrix_output[matrix_output.uid == 2]['lat'].values[0] == 43.7045)
    assert(matrix_output[matrix_output.uid == 3]['lat'].values[0] == 43.7112)

    assert(matrix_output[matrix_output.uid == 1]['lng'].values[0] == 10.3989)
    assert(matrix_output[matrix_output.uid == 2]['lng'].values[0] == 10.4331)
    assert(matrix_output[matrix_output.uid == 3]['lng'].values[0] == 10.4277)


    ###################### Most Amount | Radius: 1.3 | Spatial Join ######################
    sjoin_output = homelocation_ma(traj, week_period=None, radius=1.3, mode='sjoin', show_progress=False)

    assert(len(sjoin_output) == 3)
    assert(isinstance(sjoin_output, pd.core.frame.DataFrame))

    assert(sjoin_output.equals(matrix_output))


@pytest.mark.parametrize('traj', [traj])
def test_homelocation_tc(traj):
    ###################### Time Constraint | 19:00 - 05:00 ######################
    output = homelocation_tc(traj, start_time='19:00', end_time='5:00', show_progress=False)

    assert(len(output) == 3)
    assert(isinstance(output, pd.core.frame.DataFrame))

    assert(output[output.uid == 1]['lat'].values[0] == 43.7206)
    assert(output[output.uid == 2]['lat'].values[0] == 43.7006)
    assert(output[output.uid == 3]['lat'].values[0] == 43.7112)

    assert(output[output.uid == 1]['lng'].values[0] == 10.4026)
    assert(output[output.uid == 2]['lng'].values[0] == 10.4078)
    assert(output[output.uid == 3]['lng'].values[0] == 10.4277)
    
    
    ###################### Time Constraint | 19:00 - 05:00 ######################
    output = homelocation_tc(traj, start_time='08:00', end_time='14:00', show_progress=False)

    assert(len(output) == 3)
    assert(isinstance(output, pd.core.frame.DataFrame))

    assert(output[output.uid == 1]['lat'].values[0] == 43.7083)
    assert(output[output.uid == 2]['lat'].values[0] == 43.7045)
    assert(output[output.uid == 3]['lat'].values[0] == 43.7033)

    assert(output[output.uid == 1]['lng'].values[0] == 10.3984)
    assert(output[output.uid == 2]['lng'].values[0] == 10.4331)
    assert(output[output.uid == 3]['lng'].values[0] == 10.3992)


    ###################### Time Constraint | 06:00 - 08:00 ######################
    output = homelocation_tc(traj, start_time='06:00', end_time='08:00', radius=None, show_progress=False)

    assert(len(output) == 3)
    assert(isinstance(output, pd.core.frame.DataFrame))

    assert(output[output.uid == 1]['lat'].values[0] == 43.6932)
    assert(math.isnan(output[output.uid == 2]['lat'].values[0]))
    assert(output[output.uid == 3]['lat'].values[0] == 43.7062)

    assert(output[output.uid == 1]['lng'].values[0] == 10.4003)
    assert(math.isnan(output[output.uid == 2]['lng'].values[0]))
    assert(output[output.uid == 3]['lng'].values[0] == 10.4433)


    ###################### Time Constraint | 22:00 - 04:00 | Radius: 1 - Matrix distance ######################
    matrix_output = homelocation_tc(traj, start_time='22:00', end_time='4:00', radius=1, mode='distance_matrix', show_progress=False)

    assert(len(matrix_output) == 3)
    assert(isinstance(matrix_output, pd.core.frame.DataFrame))

    assert(matrix_output[matrix_output.uid == 1]['lat'].values[0] == 43.7206)
    assert(matrix_output[matrix_output.uid == 2]['lat'].values[0] == 43.7006)
    assert(matrix_output[matrix_output.uid == 3]['lat'].values[0] == 43.7112)

    assert(matrix_output[matrix_output.uid == 1]['lng'].values[0] == 10.4026)
    assert(matrix_output[matrix_output.uid == 2]['lng'].values[0] == 10.4078)
    assert(matrix_output[matrix_output.uid == 3]['lng'].values[0] == 10.4277)


    ###################### Time Constraint | 22:00 - 05:00 | Radius: 1.3 - Spatial join ######################
    sjoin_output = homelocation_tc(traj, start_time='22:00', end_time='5:00', radius=1.3, mode='sjoin', show_progress=False)

    assert(len(sjoin_output) == 3)
    assert(isinstance(sjoin_output, pd.core.frame.DataFrame))

    assert(sjoin_output.equals(matrix_output))


@pytest.mark.parametrize('traj', [traj])
def test_homelocation_dd(traj):
    ###################### Distinct Days ######################
    output = homelocation_dd(traj, show_progress=False)

    assert(len(output) == 3)
    assert(isinstance(output, pd.core.frame.DataFrame))

    assert(output[output.uid == 1]['lat'].values[0] == 43.7166)
    assert(output[output.uid == 2]['lat'].values[0] == 43.7045)
    assert(output[output.uid == 3]['lat'].values[0] == 43.7112)

    assert(output[output.uid == 1]['lng'].values[0] == 10.3989)
    assert(output[output.uid == 2]['lng'].values[0] == 10.4331)
    assert(output[output.uid == 3]['lng'].values[0] == 10.4277)


@pytest.mark.parametrize('traj', [traj])
def test_homelocation_inactivity(traj):
    ###################### Inactivity heuristic ######################
    output = homelocation_inactivity(traj, threshold=300, show_progress=False)

    assert(len(output) == 3)
    assert(isinstance(output, pd.core.frame.DataFrame))

    assert(output[output.uid == 1]['lat'].values[0] == 43.7206)
    assert(output[output.uid == 2]['lat'].values[0] == 43.7045)
    assert(output[output.uid == 3]['lat'].values[0] == 43.7112)

    assert(output[output.uid == 1]['lng'].values[0] == 10.4026)
    assert(output[output.uid == 2]['lng'].values[0] == 10.4331)
    assert(output[output.uid == 3]['lng'].values[0] == 10.4277)
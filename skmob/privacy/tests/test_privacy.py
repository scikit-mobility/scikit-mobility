import numpy as np
import pandas as pd
import pytest

from .. import attacks
from ...core.trajectorydataframe import TrajDataFrame
from ...utils import constants
from ...utils.utils import frequency_vector, probability_vector, date_time_precision#, TEMP

latitude = constants.LATITUDE
longitude = constants.LONGITUDE
date_time = constants.DATETIME
user_id = constants.UID

lat_lons = np.array([[43.8430139, 10.5079940],
                     [43.5442700, 10.3261500],
                     [43.7085300, 10.4036000],
                     [43.7792500, 11.2462600],
                     [43.8430139, 10.5079940],
                     [43.7085300, 10.4036000],
                     [43.8430139, 10.5079940],
                     [43.5442700, 10.3261500],
                     [43.5442700, 10.3261500],
                     [43.7085300, 10.4036000],
                     [43.8430139, 10.5079940],
                     [43.7792500, 11.2462600],
                     [43.7085300, 10.4036000],
                     [43.5442700, 10.3261500],
                     [43.7792500, 11.2462600],
                     [43.7085300, 10.4036000],
                     [43.7792500, 11.2462600],
                     [43.8430139, 10.5079940],
                     [43.8430139, 10.5079940],
                     [43.5442700, 10.3261500]])

traj = pd.DataFrame(lat_lons, columns=[latitude, longitude])

traj[date_time] = pd.to_datetime([
    '20110203 8:34:04', '20110203 9:34:04', '20110203 10:34:04', '20110204 10:34:04',
    '20110203 8:34:04', '20110203 9:34:04', '20110204 10:34:04', '20110204 11:34:04',
    '20110203 8:34:04', '20110203 9:34:04', '20110204 10:34:04', '20110204 11:34:04',
    '20110204 10:34:04', '20110204 11:34:04', '20110204 12:34:04',
    '20110204 10:34:04', '20110204 11:34:04', '20110205 12:34:04',
    '20110204 10:34:04', '20110204 11:34:04'])

traj[user_id] = [1 for _ in range(4)] + [2 for _ in range(4)] + \
                [3 for _ in range(4)] + [4 for _ in range(3)] + \
                [5 for _ in range(3)] + [6 for _ in range(2)]

traj = traj.sort_values([user_id, date_time])
trjdat = TrajDataFrame(traj, user_id=user_id)
trjfre = frequency_vector(trjdat)
trjpro = probability_vector(trjdat)

first_instance = trjdat[:2].values
second_instance = pd.concat([trjdat[0:1], trjdat[3:4]]).values
third_instance = pd.concat([trjfre[4:5], trjfre[6:7]]).values
fourth_instance = pd.concat([trjpro[4:5], trjpro[6:7]]).values


@pytest.mark.parametrize('traj,instance,output',
                         [(trjdat, first_instance, 1.0 / 4.0), (trjdat, second_instance, 1.0 / 3.0)])
def test_location_match(traj, instance, output):
    at = attacks.LocationAttack(knowledge_length=1)
    results = []
    for i in range(1, 7):
        results.append(at._match(single_traj=traj[traj[user_id] == i], instance=instance))
    assert 1.0 / sum(results) == output


@pytest.mark.parametrize('traj,instance,output',
                         [(trjdat, first_instance, 1.0 / 3.0), (trjdat, second_instance, 1.0 / 2.0)])
def test_location_sequence_match(traj, instance, output):
    at = attacks.LocationSequenceAttack(knowledge_length=1)
    results = []
    for i in range(1, 7):
        results.append(at._match(single_traj=traj[traj[user_id] == i], instance=instance))
    assert 1.0 / sum(results) == output

# TODO: check test_location_time_match

# @pytest.mark.parametrize('traj,prec,output', [(trjdat, "day", 1.0), (trjdat, "month", 1.0 / 3.0)])
# def test_location_time_match(traj, prec, output):
#     at = attacks.LocationTimeAttack(knowledge_length=1, time_precision=prec)
#     results = []
#     trjdat[TEMP] = trjdat[date_time].apply(lambda x: date_time_precision(x, prec))
#     first_instance = trjdat[:2].values
#     for i in range(1, 7):
#         results.append(at._match(single_traj=trjdat[trjdat[user_id] == i], instance=first_instance))
#     assert 1.0 / sum(results) == output


@pytest.mark.parametrize('traj,tolerance,output', [(trjfre, 0.0, 1.0), (trjfre, 1.0, 1.0 / 4.0)])
def test_location_frequency_match(traj, tolerance, output):
    at = attacks.LocationFrequencyAttack(knowledge_length=1, tolerance=tolerance)
    results = []
    for i in range(1, 7):
        results.append(at._match(single_traj=trjfre[trjfre[user_id] == i], instance=third_instance))
    assert 1.0 / sum(results) == output


@pytest.mark.parametrize('traj,tolerance,output', [(trjfre, 0.0, 1.0), (trjfre, 1.0, 1.0 / 4.0)])
def test_location_probability_match(traj, tolerance, output):
    at = attacks.LocationProbabilityAttack(knowledge_length=1, tolerance=tolerance)
    results = []
    for i in range(1, 7):
        results.append(at._match(single_traj=trjpro[trjpro[user_id] == i], instance=fourth_instance))
    assert 1.0 / sum(results) == output

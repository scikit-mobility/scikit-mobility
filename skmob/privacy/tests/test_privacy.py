import numpy as np
import pandas as pd
import pytest

from skmob.privacy import attacks
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.utils import constants
from skmob.utils.utils import frequency_vector, probability_vector, date_time_precision

latitude = constants.LATITUDE
longitude = constants.LONGITUDE
date_time = constants.DATETIME
user_id = constants.UID
frequency = constants.FREQUENCY
probability = constants.PROBABILITY

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

lat_lons_2 = np.array([[43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.5442700, 10.3261500],
                       [43.5442700, 10.3261500],
                       [43.5442700, 10.3261500],
                       [43.5442700, 10.3261500],
                       [43.7085300, 10.4036000],
                       [43.7085300, 10.4036000],

                       [43.7792500, 11.2462600],
                       [43.8430139, 10.5079940],
                       [43.7792500, 11.2462600],
                       [43.8430139, 10.5079940],

                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.5442700, 10.3261500],
                       [43.5442700, 10.3261500],
                       [43.5442700, 10.3261500],
                       [43.5442700, 10.3261500],

                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.8430139, 10.5079940],
                       [43.7085300, 10.4036000],
                       [43.5442700, 10.3261500],
                       [43.7085300, 10.4036000],
                       [43.5442700, 10.3261500],
                       [43.7085300, 10.4036000],
                       [43.5442700, 10.3261500]])

trj2 = pd.DataFrame(lat_lons_2, columns=[latitude, longitude])

trj2[date_time] = pd.to_datetime(['20110203 8:34:04' for _ in range(36)])

trj2[user_id] = [1 for _ in range(11)] + [2 for _ in range(4)] + \
                [3 for _ in range(9)] + [4 for _ in range(12)]

traj = traj.sort_values([user_id, date_time])
trjdat = TrajDataFrame(traj, user_id=user_id)
trjdat_2 = TrajDataFrame(trj2, user_id=user_id)
trj_freq = frequency_vector(trjdat_2)
trj_prob = probability_vector(trjdat_2)

first_instance = trjdat[:2].values
second_instance = pd.concat([trjdat[0:1], trjdat[3:4]]).values
third_instance = trj_freq[1:3].values
fourth_instance = trj_prob[1:3].values


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


@pytest.mark.parametrize('traj,prec,output', [(trjdat, "day", 1.0), (trjdat, "month", 1.0 / 4.0)])
def test_location_time_match(traj, prec, output):
    at = attacks.LocationTimeAttack(knowledge_length=1, time_precision=prec)
    results = []
    trjdat[constants.TEMP] = trjdat[date_time].apply(lambda x: date_time_precision(x, prec))
    first_instance = trjdat[:2].values
    for i in range(1, 7):
        results.append(at._match(single_traj=trjdat[trjdat[user_id] == i], instance=first_instance))
    assert 1.0 / sum(results) == output


@pytest.mark.parametrize('traj,tolerance,output', [(trj_freq, 0.0, 1.0/2.0), (trj_freq, 0.5, 1.0 / 3.0)])
def test_location_frequency_match(traj, tolerance, output):
    at = attacks.LocationFrequencyAttack(knowledge_length=1, tolerance=tolerance)
    results = []
    for i in range(1, 5):
        results.append(at._match(single_traj=trj_freq[trj_freq[user_id] == i], instance=third_instance))
    assert 1.0 / sum(results) == output


@pytest.mark.parametrize('traj,tolerance,output', [(trj_prob, 0.0, 1.0), (trj_prob, 1.0, 1.0 / 3.0)])
def test_location_probability_match(traj, tolerance, output):
    at = attacks.LocationProbabilityAttack(knowledge_length=1, tolerance=tolerance)
    results = []
    for i in range(1, 5):
        results.append(at._match(single_traj=trj_prob[trj_prob[user_id] == i], instance=fourth_instance))
    assert 1.0 / sum(results) == output

@pytest.mark.parametrize('traj,tolerance,output', [(trj_freq, 0.0, 1.0/2.0), (trj_freq, 1.0, 1.0 / 3.0)])
def test_location_proportion_match(traj, tolerance, output):
    at = attacks.LocationProportionAttack(knowledge_length=1, tolerance=tolerance)
    results = []
    for i in range(1, 5):
        results.append(at._match(single_traj=trj_freq[trj_freq[user_id] == i], instance=third_instance))
    assert 1.0 / sum(results) == output
from skmob import TrajDataFrame
from skmob.measures import collective
from skmob.utils import constants
import numpy as np
import pandas as pd
import math


class TestCollectiveMetrics:
    def setup_method(self):
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

        self.unique_points = [(43.544270, 10.326150), (43.708530, 10.403600), (43.779250, 11.246260), (43.843014, 10.507994)]

        self.traj = traj.sort_values([user_id, date_time])
        self.trjdat = TrajDataFrame(traj, user_id=user_id)

    def test_random_location_entropy(self):
        output = collective.random_location_entropy(self.trjdat)
        assert (len(output) == 4)
        assert (isinstance(output, pd.core.frame.DataFrame))
        assert (math.isclose(output[(output.lat == 43.544270)]['random_location_entropy'].values[0], 2.321928094))
        assert (math.isclose(output[(output.lat == 43.708530)]['random_location_entropy'].values[0], 2.321928094))
        assert (math.isclose(output[(output.lat == 43.779250)]['random_location_entropy'].values[0], 2.000000000))
        assert (math.isclose(output[(output.lng == 10.507994)]['random_location_entropy'].values[0], 2.321928094))

    def test_uncorrelated_location_entropy(self):
        output_1 = collective.uncorrelated_location_entropy(self.trjdat)

        assert (len(output_1) == 4)
        assert (isinstance(output_1, pd.core.frame.DataFrame))
        assert (math.isclose(output_1[(output_1.lat == 43.544270)]['uncorrelated_location_entropy'].values[0],
                             1.6094379124))
        assert (math.isclose(output_1[(output_1.lat == 43.708530)]['uncorrelated_location_entropy'].values[0],
                             1.6094379124))
        assert (math.isclose(output_1[(output_1.lat == 43.779250)]['uncorrelated_location_entropy'].values[0],
                             1.3862943611))
        assert (math.isclose(output_1[(output_1.lng == 10.507994)]['uncorrelated_location_entropy'].values[0],
                             1.5607104090))

        output_2 = collective.uncorrelated_location_entropy(self.trjdat, normalize=True)

        assert (len(output_2) == 4)
        assert (isinstance(output_2, pd.core.frame.DataFrame))
        assert (math.isclose(output_2[(output_2.lat == 43.544270)]['norm_uncorrelated_location_entropy'].values[0],
                             0.6931471805))
        assert (math.isclose(output_2[(output_2.lat == 43.708530)]['norm_uncorrelated_location_entropy'].values[0],
                             0.6931471805))
        assert (math.isclose(output_2[(output_2.lat == 43.779250)]['norm_uncorrelated_location_entropy'].values[0],
                             0.6931471805))
        assert (math.isclose(output_2[(output_2.lng == 10.507994)]['norm_uncorrelated_location_entropy'].values[0],
                             0.6721613871))


    def test_mean_square_displacement(self):

        output_1 = collective.mean_square_displacement(self.trjdat)
        output_2 = collective.mean_square_displacement(self.trjdat, days=0, hours=1, minutes=0)

        assert (math.isclose(output_1, output_2))
        assert (math.isclose(output_1, 1386.2137565731))

        output_3 = collective.mean_square_displacement(self.trjdat, days=1, hours=0, minutes=0)

        assert (math.isclose(output_3, 1927.57775786))

        output_4 = collective.mean_square_displacement(self.trjdat, days=0, hours=0, minutes=15)

        assert (math.isclose(output_4, 0))

        output_5 = collective.mean_square_displacement(self.trjdat, days=0, hours=0, minutes=60)

        assert (math.isclose(output_5, output_2))

    def test_visits_per_location(self):
        output = collective.visits_per_location(self.trjdat)
        assert (len(output) == 4)
        assert (isinstance(output, pd.core.frame.DataFrame))
        assert (output[(output.lat == 43.544270)]['n_visits'].values[0] == 5)
        assert (output[(output.lat == 43.708530)]['n_visits'].values[0] == 5)
        assert (output[(output.lat == 43.779250)]['n_visits'].values[0] == 4)
        assert (output[(output.lng == 10.507994)]['n_visits'].values[0] == 6)

    def test_homes_per_location(self):
        output = collective.homes_per_location(self.trjdat, start_night='22:00', end_night='07:00')

        assert (len(output) == 3)
        assert (isinstance(output, pd.core.frame.DataFrame))
        assert (output[(output.lat == 43.544270)]['n_homes'].values[0] == 4)
        assert (output[(output.lat == 43.708530)]['n_homes'].values[0] == 1)
        assert (output[(output.lng == 10.507994)]['n_homes'].values[0] == 1)

        output_2 = collective.homes_per_location(self.trjdat, start_night='18:00', end_night='11:00')

        assert (len(output_2) == 3)
        assert (isinstance(output_2, pd.core.frame.DataFrame))
        assert (output_2[(output_2.lat == 43.544270)]['n_homes'].values[0] == 2)
        assert (output_2[(output_2.lat == 43.708530)]['n_homes'].values[0] == 2)
        assert (output_2[(output_2.lng == 10.507994)]['n_homes'].values[0] == 2)

    def test_visits_per_time_unit(self):

        output_1 = collective.visits_per_time_unit(self.trjdat)['n_visits'].values

        expected_out_1 = [3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 6, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 1]

        assert(len(output_1) == len(expected_out_1))

        for i in range(len(output_1)):
            assert (output_1[i] == expected_out_1[i])

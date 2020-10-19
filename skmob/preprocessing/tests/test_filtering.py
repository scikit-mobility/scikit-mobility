import pandas as pd
import numpy as np

from skmob import TrajDataFrame
from skmob.utils import constants

from skmob.preprocessing import filtering


class TestFiltering:

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

        self.unique_points = [(43.544270, 10.326150), (43.708530, 10.403600), (43.779250, 11.246260),
                              (43.843014, 10.507994)]

        self.traj = traj.sort_values([user_id, date_time])
        self.trjdat = TrajDataFrame(traj, user_id=user_id)

    def test_filter(self):
        output = filtering.filter(self.trjdat, max_speed_kmh=10.)

        expected = self.trjdat.drop([1, 5, 9, 13, 16])

        output.reset_index(inplace=True)
        output.drop(columns=['index'], inplace=True)

        expected.reset_index(inplace=True)
        expected.drop(columns=['index'], inplace=True)

        # assert
        pd.testing.assert_frame_equal(output, expected)

        output = filtering.filter(self.trjdat, max_speed_kmh=120.)
        expected = self.trjdat

        # assert
        pd.testing.assert_frame_equal(output, expected)

        output = filtering.filter(self.trjdat, max_speed_kmh=10., max_loop=1)

        expected = self.trjdat.drop([1, 5, 9, 13, 16])

        output.reset_index(inplace=True)
        output.drop(columns=['index'], inplace=True)

        expected.reset_index(inplace=True)
        expected.drop(columns=['index'], inplace=True)

        # assert
        pd.testing.assert_frame_equal(output, expected)

        output = filtering.filter(self.trjdat, max_speed_kmh=10., ratio_max=0.9)

        expected = self.trjdat.drop([1, 5, 9, 13, 16])

        output.reset_index(inplace=True)
        output.drop(columns=['index'], inplace=True)

        expected.reset_index(inplace=True)
        expected.drop(columns=['index'], inplace=True)

        # assert
        pd.testing.assert_frame_equal(output, expected)


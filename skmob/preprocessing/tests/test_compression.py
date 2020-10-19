import pandas as pd
import numpy as np

from skmob import TrajDataFrame
from skmob.utils import constants

from skmob.preprocessing import compression


class TestCompression:

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

    def test_compress(self):
        output = compression.compress(self.trjdat)

        expected = self.trjdat.copy()

        output.reset_index(inplace=True)
        output.drop(columns=['index'], inplace=True)

        expected.reset_index(inplace=True)
        expected.drop(columns=['index'], inplace=True)

        # assert
        pd.testing.assert_frame_equal(output, expected)

        output = compression.compress(self.trjdat, spatial_radius_km=20)

        expected = TrajDataFrame({"lat":{"0":43.8430139,"1":43.6264,"2":43.77925,"3":43.8430139,"4":43.54427,
                                         "5":43.6264,"6":43.8430139,"7":43.77925,"8":43.6264,"9":43.77925,
                                         "10":43.70853,"11":43.77925,"12":43.8430139,"13":43.8430139,"14":43.54427},
                                  "lng":{"0":10.507994,"1":10.364875,"2":11.24626,"3":10.507994,"4":10.32615,
                                         "5":10.364875,"6":10.507994,"7":11.24626,"8":10.364875,"9":11.24626,
                                         "10":10.4036,"11":11.24626,"12":10.507994,"13":10.507994,"14":10.32615},
                                  "datetime":{"0":1296722044,"1":1296725644,"2":1296815644,"3":1296722044,
                                              "4":1296819244,"5":1296722044,"6":1296815644,"7":1296819244,
                                              "8":1296815644,"9":1296822844,"10":1296815644,"11":1296819244,
                                              "12":1296909244,"13":1296815644,"14":1296819244},
                                  "uid":{"0":1,"1":1,"2":1,"3":2,"4":2,"5":3,"6":3,"7":3,"8":4,"9":4,"10":5,
                                         "11":5,"12":5,"13":6,"14":6}}, timestamp=True)

        output.reset_index(inplace=True)
        output.drop(columns=['index'], inplace=True)

        expected.reset_index(inplace=True)
        expected.drop(columns=['index'], inplace=True)

        # assert
        pd.testing.assert_frame_equal(output, expected, check_dtype=False)

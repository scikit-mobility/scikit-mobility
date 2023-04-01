import numpy as np
import pandas as pd

from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.preprocessing import clustering
from skmob.utils import constants


class TestClustering:
    def setup_method(self):
        latitude = constants.LATITUDE
        longitude = constants.LONGITUDE
        date_time = constants.DATETIME
        user_id = constants.UID

        lat_lons = np.array(
            [
                [43.8430139, 10.5079940],
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
                [43.5442700, 10.3261500],
            ]
        )

        traj = pd.DataFrame(lat_lons, columns=[latitude, longitude])

        traj[date_time] = pd.to_datetime(
            [
                "20110203 8:34:04",
                "20110203 9:34:04",
                "20110203 10:34:04",
                "20110204 10:34:04",
                "20110203 8:34:04",
                "20110203 9:34:04",
                "20110204 10:34:04",
                "20110204 11:34:04",
                "20110203 8:34:04",
                "20110203 9:34:04",
                "20110204 10:34:04",
                "20110204 11:34:04",
                "20110204 10:34:04",
                "20110204 11:34:04",
                "20110204 12:34:04",
                "20110204 10:34:04",
                "20110204 11:34:04",
                "20110205 12:34:04",
                "20110204 10:34:04",
                "20110204 11:34:04",
            ]
        )

        traj[user_id] = (
            [1 for _ in range(4)]
            + [2 for _ in range(4)]
            + [3 for _ in range(4)]
            + [4 for _ in range(3)]
            + [5 for _ in range(3)]
            + [6 for _ in range(2)]
        )

        self.unique_points = [
            (43.544270, 10.326150),
            (43.708530, 10.403600),
            (43.779250, 11.246260),
            (43.843014, 10.507994),
        ]

        self.traj = traj.sort_values([user_id, date_time])
        self.trjdat = TrajDataFrame(traj, user_id=user_id)

    def test_cluster(self):
        output = clustering.cluster(self.trjdat)

        expected_cluster = [3, 2, 1, 0, 0, 2, 0, 1, 3, 2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 0]
        expected = self.trjdat
        expected["cluster"] = expected_cluster

        # assert
        pd.testing.assert_frame_equal(output, expected)

        output = clustering.cluster(self.trjdat, cluster_radius_km=40)

        expected_cluster = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        expected = self.trjdat
        expected["cluster"] = expected_cluster

        # assert
        pd.testing.assert_frame_equal(output, expected)

    def test_dbscan(self):
        output = clustering.dbscan(self.trjdat)

        expected_cluster = [3, 2, 1, 0, 0, 2, 0, 1, 3, 2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 0]
        expected = self.trjdat
        expected["cluster"] = expected_cluster

        # assert
        pd.testing.assert_frame_equal(output, expected)

        output = clustering.dbscan(self.trjdat, cluster_radius_km=40)

        expected_cluster = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        expected = self.trjdat
        expected["cluster"] = expected_cluster

        # assert
        pd.testing.assert_frame_equal(output, expected)
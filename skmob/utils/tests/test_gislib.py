from skmob.utils import gislib
import math


class TestClustering:

    def setup_method(self):
        self.point_1 = (43.8430139, 10.5079940)
        self.point_2 = (43.5442700, 10.3261500)

        self.decimal = 43.8430139
        self.DMS = (43, 50, 34.85)

    def test_get_distance(self):
        output = gislib.getDistance(self.point_1, self.point_2)
        assert (math.isclose(output, 36.293701213))

        support = gislib.getDistanceByHaversine(self.point_1, self.point_2)
        assert (math.isclose(support, output))

        output = gislib.getDistance(self.point_1, self.point_1)
        assert (math.isclose(output, 0))

    def test_get_distance_by_haversine(self):
        output = gislib.getDistanceByHaversine(self.point_1, self.point_2)
        assert (math.isclose(output, 36.293701213))

        output = gislib.getDistanceByHaversine(self.point_1, self.point_1)
        assert (math.isclose(output, 0))

    # def test_decimal_to_DMS(self):
    #     output = gislib.DecimalToDMS(self.decimal)
    #     assert (output[0] == 43)
    #     assert (output[1] == 50)
    #     assert (math.isclose(output[2], 34.85))

    def test_DMS_to_decimal(self):
        output = gislib.DMSToDecimal(self.DMS[0], self.DMS[1], self.DMS[2])
        assert (math.isclose(output, 43.84301388888))

    def test_get_coordinates_for_distance(self):
        output = gislib.getCoordinatesForDistance(self.point_1[0], self.point_1[1], 15)
        assert (math.isclose(output[0], 0.134989200863))
        assert (math.isclose(output[1], 0.187162559305))

    # def test_is_within_distance(self):
    #     assert (gislib.isWithinDistance(self.point_1, self.point_2, 20))
    #     assert (gislib.isWithinDistance(self.point_1, self.point_2, 40) is False)

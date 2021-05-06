import skmob.measures.evaluation as sk
import math
import numpy as np

class TestEvaluationMetrics:

    def setup_method(self):
        self.ground = [1,2,3]

        self.prediction_a = [1,2,3]
        self.prediction_b = [2,2,2]
        self.prediction_c = [3,2,1]

    def test_r_squared(self):
        assert(sk.r_squared(self.ground, self.prediction_a) == 1)
        assert(sk.r_squared(self.ground, self.prediction_b) == 0)
        assert(sk.r_squared(self.ground, self.prediction_c) == -3)

    def test_mse(self):
        assert(sk.mse(self.ground, self.prediction_a) == 0)
        assert(math.isclose(sk.mse(self.ground, self.prediction_b), 0.6666666666))
        assert (math.isclose(sk.mse(self.ground, self.prediction_c), 2.6666666666))

    def test_rmse(self):
        assert (math.sqrt(sk.mse(self.ground, self.prediction_a)) == 0)
        assert (math.isclose(math.sqrt(sk.mse(self.ground, self.prediction_b)), 0.8164965809))
        assert (math.isclose(math.sqrt(sk.mse(self.ground, self.prediction_c)), 1.63299316185))

    def test_nrmse(self):
        assert (math.sqrt(sk.mse(self.ground, self.prediction_a))/np.sum(self.ground) == 0)
        assert (math.isclose(math.sqrt(sk.mse(self.ground, self.prediction_b))/np.sum(self.ground), 0.1360827635))
        assert (math.isclose(math.sqrt(sk.mse(self.ground, self.prediction_c))/np.sum(self.ground), 0.272165527))

    def test_max_error(self):
        assert (sk.max_error(self.ground, self.prediction_a) == 0)
        assert (sk.max_error(self.ground, self.prediction_b) == 1)
        assert (sk.max_error(self.ground, self.prediction_c) == 2)

    def test_information_gain(self):
        assert (math.isclose(sk.information_gain(self.ground, self.prediction_a), 0.0))
        assert (math.isclose(sk.information_gain(self.ground, self.prediction_b), 0.08720802396))
        assert (math.isclose(sk.information_gain(self.ground, self.prediction_c), 0.36620409622))

    def test_pearson_correlation(self):
        assert (math.isclose(sk.pearson_correlation(self.ground, self.prediction_a)[0], 0.999999999999))
        assert (math.isnan(sk.pearson_correlation(self.ground, self.prediction_b)[0]) == True)
        assert (math.isclose(sk.pearson_correlation(self.ground, self.prediction_c)[0], -0.999999999999))

    def test_spearman_correlation(self):
        assert (math.isclose(sk.spearman_correlation(self.ground, self.prediction_a)[0], 0.999999999999))
        assert (math.isnan(sk.spearman_correlation(self.ground, self.prediction_b)[0]) == True)
        assert (math.isclose(sk.spearman_correlation(self.ground, self.prediction_c)[0], -0.999999999999))


    def test_kl_divergence(self):
        assert (math.isclose(sk.kullback_leibler_divergence(self.ground, self.prediction_a), 0))
        assert (math.isclose(sk.kullback_leibler_divergence(self.ground, self.prediction_b), 0.0872080239607))
        assert (math.isclose(sk.kullback_leibler_divergence(self.ground, self.prediction_c), 0.3662040962227))

    def test_common_part_of_commuters(self):
        assert (math.isclose(sk.common_part_of_commuters(self.ground, self.prediction_a), 1))
        assert (math.isclose(sk.common_part_of_commuters(self.ground, self.prediction_b), 0.83333333333))
        assert (math.isclose(sk.common_part_of_commuters(self.ground, self.prediction_c), 0.66666666666))

    def test_common_part_of_commuters_distance(self):
        assert (math.isclose(sk.common_part_of_commuters_distance(self.ground, self.prediction_a), 0.333333333333))
        assert (math.isclose(sk.common_part_of_commuters_distance(self.ground, self.prediction_b), 0.333333333333))
        assert (math.isclose(sk.common_part_of_commuters_distance(self.ground, self.prediction_c), 0.333333333333))
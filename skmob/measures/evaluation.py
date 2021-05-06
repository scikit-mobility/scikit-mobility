import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error


def common_part_of_commuters(values1, values2):
    """
    Compute the common part of commuters for two pairs of fluxes.

    :param values1: the values for the first array
    :type values1: numpy array

    :param values2: the values for the second array
    :type values1: numpy array

    :return: float
        the common part of commuters
    """
    return 2.0 * np.sum(np.minimum(values1, values2)) / (np.sum(values1) + np.sum(values2))


def common_part_of_links(values1, values2):
    """
    Compute the common part of commuters for two pairs of fluxes.

    :param values1: the values for the first array
    :type values1: numpy array

    :param values2: the values for the second array
    :type values2: numpy array

    :return: float
        the common part of commuters
    """

    def check_condition(value):
        if value > 0:
            return 1
        return 0

    cpl_num = 0.0
    cpl_den = 0.0
    for val1, val2 in zip(values1, values2):
        cpl_num += check_condition(val1) * check_condition(val2, 1)
        cpl_den += check_condition(val1) + check_condition(val2)
    return 2.0 * cpl_num / cpl_den


def common_part_of_commuters_distance(values1, values2):
    """
    Compute the common part of commuters according to the distance.

    :param values1: the values for the first array
    :type values1: numpy array

    :param values2: the values for the second array
    :type values2: numpy array

    :return: float
        the common part of commuters according to the distance
    """
    max_val = max(max(values1), max(values2))
    bins = np.arange(0, max_val, 2)
    hist1, bin_edges1 = np.histogram(values1, bins)
    hist2, bin_edges2 = np.histogram(values2, bins)

    N, cpcd = sum(values1), 0.0
    for k1, k2 in zip(hist1, hist2):
        cpcd += min(k1, k2)
    cpcd /= N
    return cpcd


def r_squared(true, pred):
    """
    R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative
    (because the model can be arbitrarily worse).
    A constant model that always predicts the expected value of y,
    disregarding the input features, would get a R^2 score of 0.0.

    :param true: Ground truth (correct) target values.
    :type true: numpy array array-like of shape = (n_samples) or (n_samples, n_outputs)

    :param pred: Estimated target values.
    :type pred: numpy array-like of shape = (n_samples) or (n_samples, n_outputs)

    :return: float
    """
    return r2_score(true, pred)


def mse(true, pred):
    """
    Mean squared error regression loss

    :param true: Ground truth (correct) target values.
    :type true: numpy array array-like of shape = (n_samples) or (n_samples, n_outputs)

    :param pred: Estimated target values.
    :type pred: numpy array-like of shape = (n_samples) or (n_samples, n_outputs)

    :return: float
        A non-negative floating point value (the best value is 0.0)
    """
    return mean_squared_error(true, pred)


def rmse(true, pred):
    """
    Root mean squared error regression loss

    :param true: Ground truth (correct) target values.
    :type true: numpy array array-like of shape = (n_samples) or (n_samples, n_outputs)

    :param pred: Estimated target values.
    :type pred: numpy array-like of shape = (n_samples) or (n_samples, n_outputs)

    :return: float
        A non-negative floating point value (the best value is 0.0)
    """
    return np.sqrt(mse(true, pred))


def nrmse(true, pred):
    """
    Normalized mean squared error regression loss

    :param true: Ground truth (correct) target values.
    :type true: numpy array array-like of shape = (n_samples) or (n_samples, n_outputs)

    :param pred: Estimated target values.
    :type pred: numpy array-like of shape = (n_samples) or (n_samples, n_outputs)

    :return: float
        A non-negative floating point value (the best value is 0.0)
    """
    return rmse(true, pred) / np.sum(true)


def information_gain(true, pred):
    """
    The information gain

    :param true: Ground truth (correct) target values.
    :type true: numpy array array-like of shape = (n_samples) or (n_samples, n_outputs)

    :param pred: Estimated target values.
    :type pred: numpy array-like of shape = (n_samples) or (n_samples, n_outputs)

    :return: float
        A non-negative floating point value (the best value is 0.0)
    """
    N = np.sum(true)
    information_gain = 0.0
    for true_value, pred_value in zip(true, pred):
        information_gain += (1.0 * true_value / N) * np.log(true_value / pred_value)
    return information_gain


def pearson_correlation(true, pred):
    """
    Calculates a Pearson correlation coefficient and the p-value for testing non-correlation.
    The Pearson correlation coefficient measures the linear relationship between two datasets.
    Strictly speaking, Pearson’s correlation requires that each dataset be normally distributed. Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply an exact linear relationship. Positive correlations imply that as x increases, so does y.
    Negative correlations imply that as x increases, y decreases.
    The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a Pearson correlation at least as extreme as the one computed from these datasets. The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.

    :param true: Ground truth (correct) target values.
    :type true: numpy array array-like of shape = (n_samples) or (n_samples, n_outputs)

    :param pred: Estimated target values.
    :type pred: numpy array-like of shape = (n_samples) or (n_samples, n_outputs)

    :return: tuple
        (Pearson’s correlation coefficient, 2-tailed p-value)
    """
    return stats.pearsonr(true, pred)


def spearman_correlation(true, pred):
    """
    Calculates a Spearman rank-order correlation coefficient and the p-value to test for non-correlation.
    The Spearman correlation is a nonparametric measure of the monotonicity of the relationship between two datasets.
    Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed.
    Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation.
    Correlations of -1 or +1 imply an exact monotonic relationship. Positive correlations imply that as x increases, so does y.
    Negative correlations imply that as x increases, y decreases. The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Spearman correlation at least as extreme as the one computed from these datasets.
    The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.

    :param true: Ground truth (correct) target values.
    :type true: numpy array array-like of shape = (n_samples) or (n_samples, n_outputs)

    :param pred: Estimated target values.
    :type pred: numpy array-like of shape = (n_samples) or (n_samples, n_outputs)

    :return: tuple
        (Spearman’s correlation coefficient, 2-tailed p-value)
    """
    return stats.spearmanr(true, pred)


def kullback_leibler_divergence(true, pred):
    """
    Compute the Kullback-Leibler divergence S = sum(pk * log(pk / qk), axis=0).

    :param true: Ground truth (correct) target values.
    :type true: numpy array array-like of shape = (n_samples) or (n_samples, n_outputs)

    :param pred: Estimated target values.
    :type pred: numpy array-like of shape = (n_samples) or (n_samples, n_outputs)

    :return: float
        the calculated Kullback-Leibler divergence
    """
    return stats.entropy(true, pred)


def max_error(true, pred):
    """
    The maximum error between the two arrays

    :param true: Ground truth (correct) target values.
    :type true: numpy array array-like of shape = (n_samples) or (n_samples, n_outputs)

    :param pred: Estimated target values.
    :type pred: numpy array-like of shape = (n_samples) or (n_samples, n_outputs)

    :return: float
        max error between the two samples
    """
    return np.max(np.subtract(true, pred))

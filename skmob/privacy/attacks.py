from itertools import combinations
from abc import ABCMeta, abstractmethod
from skmob.utils import constants
from skmob.core.trajectorydataframe import TrajDataFrame
from tqdm import tqdm
import pandas as pd
from ..utils.utils import frequency_vector, probability_vector, date_time_precision


class Attack(object):

    """Privacy Attack

    Abstract class for a generic attack. Defines a series of functions common to all attacks.
    Provides basic functions to compute risk for all users in a trajectory dataframe.
    Requires the implementation of both a matching function and an assessment function, which are attack dependant.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.
    Attributes
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate.

    References
    ----------
    .. [TISTPELL] Roberto Pellungrini, Luca Pappalardo, Francesca Pratesi, and Anna Monreale. 2017. A Data Mining Approach to Assess Privacy Risk in Human Mobility Data. ACM Trans. Intell. Syst. Technol. 9, 3, Article 31 (December 2017), 27 pages. DOI: https://doi.org/10.1145/3106774
    """
    __metaclass__ = ABCMeta

    def __init__(self, knowledge_length):
        self.knowledge_length = knowledge_length

    @property
    def knowledge_length(self):
        return self._knowledge_length

    @knowledge_length.setter
    def knowledge_length(self, val):
        if val < 1:
            raise ValueError("Parameter knowledge_length should not be less than 1")
        self._knowledge_length = val

    def _all_risks(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Computes risk for all the users in the data. It applies the risk function to every individual in the data.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the calculation on.

        Parameters
        ----------
        traj: TrajectoryDataFrame
            the dataframe against which to calculate risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The default is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The default is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The default is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk)
        """
        if targets is None:
            targets = traj
        else:
            if isinstance(targets, list):
                targets = traj[traj[constants.UID].isin(targets)]
            if isinstance(targets, TrajDataFrame) or isinstance(targets, pd.DataFrame):
                targets = traj[traj[constants.UID].isin(targets[constants.UID])]
        if show_progress:
            tqdm.pandas(desc="computing risk")
            risks = targets.groupby(constants.UID).progress_apply(lambda x: self._risk(x, traj, force_instances))
        else:
            risks = targets.groupby(constants.UID).apply(lambda x: self._risk(x, traj, force_instances))
        if force_instances:
            risks = risks.droplevel(1)
            risks = risks.reset_index(drop=True)
        else:
            risks = risks.reset_index(name=constants.PRIVACY_RISK)
        return risks

    def _generate_instances(self, single_traj):
        """
        Return a generator to all the possible background knowledge of length k for a single user_id.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        Yields
        ------
        generator
            a generator to all the possible instances of length k. Instances are tuples with the values of the actual
            records in the combination.
        """
        size = len(single_traj.index)
        if self.knowledge_length > size:
            return combinations(single_traj.values, size)
        else:
            return combinations(single_traj.values, self.knowledge_length)

    def _risk(self, single_traj, traj, force_instances=False):
        """
        Computes the risk of reidentification of an individual with respect to the entire population in the data.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        traj : TrajectoryDataFrame
            the dataframe with the complete data.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The default is `False`.

        Returns
        -------
        float
            the risk for the individual, expressed as a float between 0 and 1
        """
        instances = self._generate_instances(single_traj)
        risk = 0
        if force_instances:
            inst_data = {constants.LATITUDE: list(), constants.LONGITUDE: list(),
                         constants.DATETIME: list(), constants.UID: list(),
                         constants.INSTANCE: list(), constants.INSTANCE_ELEMENT: list(),
                         constants.PROBABILITY: list()}
            inst_id = 1
            for instance in instances:
                prob = 1.0 / traj.groupby(constants.UID).apply(lambda x: self._match(x, instance)).sum()
                elem_count = 1
                for elem in instance:
                    inst_data[constants.LATITUDE].append(elem[0])
                    inst_data[constants.LONGITUDE].append(elem[1])
                    inst_data[constants.DATETIME].append(elem[2])
                    inst_data[constants.UID].append(elem[3])
                    inst_data[constants.INSTANCE].append(inst_id)
                    inst_data[constants.INSTANCE_ELEMENT].append(elem_count)
                    inst_data[constants.PROBABILITY].append(prob)
                    elem_count += 1
                inst_id += 1
            return pd.DataFrame(inst_data)
        else:
            for instance in instances:
                prob = 1.0 / traj.groupby(constants.UID).apply(lambda x: self._match(x, instance)).sum()
                if prob > risk:
                    risk = prob
                if risk == 1.0:
                    break
            return risk

    @abstractmethod
    def assess_risk(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Abstract function to assess privacy risk for a TrajectoryDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        traj : TrajectoryDataFrame
            the dataframe on which to assess privacy risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The defaul is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The defaul is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The defaul is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        pass

    @abstractmethod
    def _match(self, single_traj, instance):
        """
        Matching function for the attack. It is used to decide if an instance of background knowledge matches a certain
        trajectory. The internal logic of an attack is represented by this function, therefore, it must be implemented
        depending in the kind of the attack.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        instance : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the trajectory, 0 otherwise.
        """
        pass


class LocationAttack(Attack):
    """Location Attack

    In a location attack the adversary knows the coordinates of the locations visited by an individual and matches them
    against trajectories.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.
    Attributes
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate.

    See Also
    --------
    Attack
    """

    def __init__(self, knowledge_length):
        super(LocationAttack, self).__init__(knowledge_length)

    def assess_risk(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Assess privacy risk for a TrajectoryDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        traj : TrajectoryDataFrame
            the dataframe on which to assess privacy risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The defaul is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The defaul is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The defaul is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        traj = traj.sort_values(by=[constants.UID, constants.DATETIME])
        return self._all_risks(traj, targets, force_instances, show_progress)

    def _match(self, single_traj, instance):
        """
        Matching function for the attack.
        For a location attack, only the coordinates are used in the matching.
        If a trajectory presents the same locations as the ones in the instance, a match is found.
        Multiple visits to the same location are also handled.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        instance : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the trajectory, 0 otherwise.
        """
        locs = single_traj.groupby([constants.LATITUDE, constants.LONGITUDE]).size().reset_index(name=constants.COUNT)
        inst = pd.DataFrame(data=instance, columns=single_traj.columns)
        inst = inst.astype(dtype=dict(single_traj.dtypes))
        inst = inst.groupby([constants.LATITUDE, constants.LONGITUDE]).size().reset_index(name=constants.COUNT + "inst")
        locs_inst = pd.merge(locs, inst, left_on=[constants.LATITUDE, constants.LONGITUDE],
                             right_on=[constants.LATITUDE, constants.LONGITUDE])
        if len(locs_inst.index) != len(inst.index):
            return 0
        else:
            condition = locs_inst[constants.COUNT] >= locs_inst[constants.COUNT + "inst"]
            if len(locs_inst[condition].index) != len(inst.index):
                return 0
            else:
                return 1


class LocationSequenceAttack(Attack):
    """Location Sequence Attack
    In a location sequence attack the adversary knows the coordinates of locations visited by an individual and
    the order in which they were visited and matches them against trajectories.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.
    Attributes
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate.

    See Also
    --------
    Attack
    """

    def __init__(self, knowledge_length):
        super(LocationSequenceAttack, self).__init__(knowledge_length)

    def assess_risk(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Assess privacy risk for a TrajectoryDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        traj : TrajectoryDataFrame
            the dataframe on which to assess privacy risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The defaul is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The defaul is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The defaul is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        traj = traj.sort_values(by=[constants.UID, constants.DATETIME])
        return self._all_risks(traj, targets, force_instances, show_progress)

    def _match(self, single_traj, instance):
        """
        Matching function for the attack.
        For a location sequence attack, both the coordinates and the order of visit are used in the matching.
        If a trajectory presents the same locations in the same order as the ones in the instance, a match is found.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        instance : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the trajectory, 0 otherwise.
        """
        inst = pd.DataFrame(data=instance, columns=single_traj.columns)
        inst_iterator = inst.iterrows()
        inst_line = next(inst_iterator)[1]
        count = 0
        for index, row in single_traj.iterrows():
            if inst_line[constants.LATITUDE] == row[constants.LATITUDE] and inst_line[constants.LONGITUDE] == row[
                constants.LONGITUDE]:
                count += 1
                try:
                    inst_line = next(inst_iterator)[1]
                except StopIteration:
                    break
        if len(inst.index) == count:
            return 1
        else:
            return 0


class LocationTimeAttack(Attack):
    """Location Time Attack

    In a location time attack the adversary knows the coordinates of locations visited by an individual and the time
    in which they were visited and matches them against trajectories. The precision at which to consider the temporal
    information can also be specified.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.

    time_precision : string, optional
        the precision at which to consider the timestamps for the visits.
        The possible precisions are: Year, Month, Day, Hour, Minute, Second. The default is `Hour`

    Attributes
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate.

    time_precision : string
        the precision at which to consider the timestamps for the visits.

    See Also
    --------
    Attack
    """

    def __init__(self, knowledge_length, time_precision="Hour"):
        self.time_precision = time_precision
        super(LocationTimeAttack, self).__init__(knowledge_length)

    @property
    def time_precision(self):
        return self._time_precision

    @time_precision.setter
    def time_precision(self, val):
        if val not in constants.PRECISION_LEVELS:
            raise ValueError("Possible time precisions are: Year, Month, Day, Hour, Minute, Second")
        self._time_precision = val

    def assess_risk(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Assess privacy risk for a TrajectoryDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        traj : TrajectoryDataFrame
            the dataframe on which to assess privacy risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The defaul is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The defaul is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The defaul is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        traj = traj.sort_values(by=[constants.UID, constants.DATETIME])
        traj[constants.TEMP] = traj[constants.DATETIME].apply(lambda x: date_time_precision(x, self.time_precision))
        return self._all_risks(traj, targets, force_instances, show_progress)

    def _match(self, single_traj, instance):
        """
        Matching function for the attack.
        For a location time attack, both the coordinates and the order of visit are used in the matching.
        If a trajectory presents the same locations with the same temporal information as in the instance,
        a match is found.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        instance : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the trajectory, 0 otherwise.
        """
        inst = pd.DataFrame(data=instance, columns=single_traj.columns)
        locs_inst = pd.merge(single_traj, inst, left_on=[constants.LATITUDE, constants.LONGITUDE, constants.TEMP],
                             right_on=[constants.LATITUDE, constants.LONGITUDE, constants.TEMP])
        if len(locs_inst.index) == len(inst.index):
            return 1
        else:
            return 0


class UniqueLocationAttack(Attack):
    """Unique Location Attack

    In a unique location attack the adversary knows the coordinates of unique locations visited by an individual,
    and matches them against frequency vectors. A frequency vector, is an aggregation on trajectory
    data showing the unique locations visited by an individual and the frequency with which he visited those locations.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.
    Attributes
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate.

    See Also
    --------
    Attack
    """

    def __init__(self, knowledge_length):
        super(UniqueLocationAttack, self).__init__(knowledge_length)

    def assess_risk(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Assess privacy risk for a TrajectoryDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        traj : TrajectoryDataFrame
            the dataframe on which to assess privacy risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The defaul is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The defaul is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The defaul is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        freq = frequency_vector(traj)
        return self._all_risks(freq, targets, force_instances, show_progress)

    def _match(self, single_traj, instance):
        """
        Matching function for the attack.
        For a unique location attack, the coordinates of unique locations are used in the matching.
        If a frequency vector presents the same locations as in the instance, a match is found.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        instance : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the trajectory, 0 otherwise.
        """
        inst = pd.DataFrame(data=instance, columns=single_traj.columns)
        locs_inst = pd.merge(single_traj, inst, left_on=[constants.LATITUDE, constants.LONGITUDE],
                             right_on=[constants.LATITUDE, constants.LONGITUDE])
        if len(locs_inst.index) == len(inst.index):
            return 1
        else:
            return 0


class LocationFrequencyAttack(Attack):
    """Location Frequency Attack

    In a location frequency attack the adversary knows the coordinates of the unique locations visited by an individual
    and the frequency with which he visited them, and matches them against frequency vectors. A frequency vector,
    is an aggregation on trajectory data showing the unique locations visited by an individual and the frequency
    with which he visited those locations. It is possible to specify a tolerance level for the matching of the frequency.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.

    tolerance : float, optional
        the tolarance with which to match the frequency. It can assume values between 0 and 1. The defaul is `0`.

    Attributes
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate.

    tolerance : float
        the tolarance with which to match the frequency.

    See Also
    --------
    Attack
    """

    def __init__(self, knowledge_length, tolerance=0.0):
        self.tolerance = tolerance
        super(LocationFrequencyAttack, self).__init__(knowledge_length)

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val):
        if val > 1.0 or val < 0.0:
            raise ValueError("Tolerance should be in the interval [0.0,1.0]")
        self._tolerance = val

    def assess_risk(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Assess privacy risk for a TrajectoryDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        traj : TrajectoryDataFrame
            the dataframe on which to assess privacy risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The defaul is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The defaul is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The defaul is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        freq = frequency_vector(traj)
        return self._all_risks(freq, targets, force_instances, show_progress)

    def _match(self, single_traj, instance):
        """
        Matching function for the attack.
        For a frequency location attack, the coordinates of unique locations and their frequency of visit are used
        in the matching. If a frequency vector presents the same locations with the same frequency as in the instance,
        a match is found. The tolerance level specified at construction is used to construct and interval of frequency
        and allow for less precise matching.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        instance : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the trajectory, 0 otherwise.
        """
        inst = pd.DataFrame(data=instance, columns=single_traj.columns)
        inst.rename(columns={constants.FREQUENCY: constants.FREQUENCY + "inst"}, inplace=True)
        locs_inst = pd.merge(single_traj, inst, left_on=[constants.LATITUDE, constants.LONGITUDE],
                             right_on=[constants.LATITUDE, constants.LONGITUDE])
        if len(locs_inst.index) != len(inst.index):
            return 0
        else:
            condition1 = locs_inst[constants.FREQUENCY + "inst"] >= locs_inst[constants.FREQUENCY] - (
                    locs_inst[constants.FREQUENCY] * self.tolerance)
            condition2 = locs_inst[constants.FREQUENCY + "inst"] <= locs_inst[constants.FREQUENCY] + (
                    locs_inst[constants.FREQUENCY] * self.tolerance)
            if len(locs_inst[condition1 & condition2].index) != len(inst.index):
                return 0
            else:
                return 1


class LocationProbabilityAttack(Attack):
    """Location Probability Attack

    In a location probability attack the adversary knows the coordinates of
    the unique locations visited by an individual and the probability with which he visited them,
    and matches them against probability vectors.
    A probability vector, is an aggregation on trajectory data showing the unique locations visited by an individual
    and the probability with which he visited those locations.
    It is possible to specify a tolerance level for the matching of the probability.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.

    tolerance : float, optional
        the tolarance with which to match the probability. It can assume values between 0 and 1. The defaul is `0`.

    Attributes
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate.

    tolerance : float
        the tolarance with which to match the probability.

    See Also
    --------
    Attack
    """

    def __init__(self, knowledge_length, tolerance=0.0):
        self.tolerance = tolerance
        super(LocationProbabilityAttack, self).__init__(knowledge_length)

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val):
        if val > 1.0 or val < 0.0:
            raise ValueError("Tolerance should be in the interval [0.0,1.0]")
        self._tolerance = val

    def assess_risk(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Assess privacy risk for a TrajectoryDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        traj : TrajectoryDataFrame
            the dataframe on which to assess privacy risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The defaul is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The defaul is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The defaul is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        prob = probability_vector(traj)
        return self._all_risks(prob, targets, force_instances, show_progress)

    def _match(self, single_traj, instance):
        """
        Matching function for the attack.
        For a probability location attack, the coordinates of unique locations and their probability of visit are used
        in the matching.
        If a probability vector presents the same locations with the same probability as in the instance,
        a match is found.
        The tolerance level specified at construction is used to build and interval of probability and allow
        for less precise matching.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        instance : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the trajectory, 0 otherwise.
        """
        inst = pd.DataFrame(data=instance, columns=single_traj.columns)
        inst.rename(columns={constants.PROBABILITY: constants.PROBABILITY + "inst"}, inplace=True)
        locs_inst = pd.merge(single_traj, inst, left_on=[constants.LATITUDE, constants.LONGITUDE],
                             right_on=[constants.LATITUDE, constants.LONGITUDE])
        if len(locs_inst.index) != len(inst.index):
            return 0
        else:
            condition1 = locs_inst[constants.PROBABILITY + "inst"] >= locs_inst[constants.PROBABILITY] - (
                    locs_inst[constants.PROBABILITY] * self.tolerance)
            condition2 = locs_inst[constants.PROBABILITY + "inst"] <= locs_inst[constants.PROBABILITY] + (
                    locs_inst[constants.PROBABILITY] * self.tolerance)
            if len(locs_inst[condition1 & condition2].index) != len(inst.index):
                return 0
            else:
                return 1


class LocationProportionAttack(Attack):
    """Location Proportion Attack

    In a location proportion attack the adversary knows the coordinates of the unique locations visited
    by an individual and the relative proportions between their frequencies of visit,
    and matches them against frequency vectors.
    A frequency vector is an aggregation on trajectory data showing the unique locations visited by an individual
    and the frequency with which he visited those locations.
    It is possible to specify a tolerance level for the matching of the proportion.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.

    tolerance : float, optional
        the tolarance with which to match the frequency. It can assume values between 0 and 1. The defaul is `0`.

    Attributes
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate.

    tolerance : float
        the tolarance with which to match the frequency.

    See Also
    --------
    Attack
    """

    def __init__(self, knowledge_length, tolerance=0.0):
        self.tolerance = tolerance
        super(LocationProportionAttack, self).__init__(knowledge_length)

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val):
        if val > 1.0 or val < 0.0:
            raise ValueError("Tolerance should be in the interval [0.0,1.0]")
        self._tolerance = val

    def assess_risk(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Assess privacy risk for a TrajectoryDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        traj : TrajectoryDataFrame
            the dataframe on which to assess privacy risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The defaul is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The defaul is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The defaul is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        freq = frequency_vector(traj)
        return self._all_risks(freq, targets, force_instances, show_progress)

    def _match(self, single_traj, instance):
        """
        Matching function for the attack. For a proportion location attack,
        the coordinates of unique locations and their relative proportion of frequency of visit
        are used in the matching.
        The proportion of visit are calculated with respect to the most frequent location found in the instance.
        If a frequency vector presents the same locations with the same proportions of frequency of
        visit as in the instance, a match is found.
        The tolerance level specified at construction is used to build an interval of proportion
        and allow for less precise matching.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        instance : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the trajectory, 0 otherwise.
        """
        inst = pd.DataFrame(data=instance, columns=single_traj.columns)
        inst.rename(columns={constants.FREQUENCY: constants.FREQUENCY + "inst"}, inplace=True)
        locs_inst = pd.merge(single_traj, inst, left_on=[constants.LATITUDE, constants.LONGITUDE],
                             right_on=[constants.LATITUDE, constants.LONGITUDE])
        if len(locs_inst.index) != len(inst.index):
            return 0
        else:
            locs_inst[constants.PROPORTION + "inst"] = locs_inst[constants.FREQUENCY + "inst"] / locs_inst[
                constants.FREQUENCY + "inst"].max()
            locs_inst[constants.PROPORTION] = locs_inst[constants.FREQUENCY] / locs_inst[constants.FREQUENCY].max()
            condition1 = locs_inst[constants.PROPORTION + "inst"] >= locs_inst[constants.PROPORTION] - (
                    locs_inst[constants.PROPORTION] * self.tolerance)
            condition2 = locs_inst[constants.PROPORTION + "inst"] <= locs_inst[constants.PROPORTION] + (
                    locs_inst[constants.PROPORTION] * self.tolerance)
            if len(locs_inst[condition1 & condition2].index) != len(inst.index):
                return 0
            else:
                return 1


class HomeWorkAttack(Attack):
    """Home And Work Attack

    In a home and work attack the adversary knows the coordinates of
    the two locations most frequently visited by an individual, and matches them against frequency vectors.
    A frequency vector is an aggregation on trajectory data showing the unique
    locations visited by an individual and the frequency with which he visited those locations.
    This attack does not require the generation of combinations to build the possible instances of background knowledge.

    Parameters
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate. The length of the background knowledge
        specifies the amount of knowledge that the adversary will use for her attack. For each individual all the
        combinations of points of length k will be evaluated.
    Attributes
    ----------
    knowledge_length : int
        the length of the background knowledge that we want to simulate.

    See Also
    --------
    Attack

    """

    def __init__(self, knowledge_length=0):
        super(HomeWorkAttack, self).__init__(knowledge_length)

    def _generate_instances(self, single_traj):
        """
        Returns the two most frequently visited locations by an individual.
        This is an ovverride of the _generate_instances method of the Attack absttract class.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        Returns
        -------
        list
            a list with the records with the two most frequently visited locations.
        """
        return [single_traj[:2].values]

    def assess_risk(self, traj, targets=None, force_instances=False, show_progress=False):
        """
        Assess privacy risk for a TrajectoryDataFrame.
        An attack must implement an assessing strategy. This could involve some preprocessing, for example
        transforming the original data, and calls to the risk function.
        If it is not required to compute the risk for the entire data, the targets parameter can be used to select
        a portion of users to perform the assessment on.

        Parameters
        ----------
        traj : TrajectoryDataFrame
            the dataframe on which to assess privacy risk.

        targets : TrajectoryDataFrame or list, optional
            the users_id target of the attack.  They must be compatible with the trajectory data. Default values is None
            in which case risk is computed on all users in traj. The defaul is `None`.

        force_instances : boolean, optional
            if True, returns all possible instances of background knowledge
            with their respective probability of reidentification. The defaul is `False`.

        show_progress : boolean, optional
            if True, shows the progress of the computation. The defaul is `False`.

        Returns
        -------
        DataFrame
            a DataFrame with the privacy risk for each user, in the form (user_id, risk).
        """
        freq = frequency_vector(traj)
        return self._all_risks(freq, targets, force_instances, show_progress)

    def _match(self, single_traj, instance):
        """
        Matching function for the attack.
        For a home and work attack, the coordinates of the two locations are used in the matching.
        If a frequency vector presents the same locations as in the instance, a match is found.

        Parameters
        ----------
        single_traj : TrajectoryDataFrame
            the dataframe of the trajectory of a single individual.

        instance : tuple
            an instance of background knowledge.

        Returns
        -------
        int
            1 if the instance matches the trajectory, 0 otherwise.
        """
        inst = pd.DataFrame(data=instance, columns=single_traj.columns)
        locs_inst = pd.merge(single_traj[:2], inst, left_on=[constants.LATITUDE, constants.LONGITUDE],
                             right_on=[constants.LATITUDE, constants.LONGITUDE])
        if len(locs_inst.index) == len(inst.index):
            return 1
        else:
            return 0

import datetime
from tqdm import tqdm
import operator
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from ..utils import constants
import random

latitude = constants.LATITUDE
longitude = constants.LONGITUDE
date_time = constants.DATETIME
user_id = constants.UID


class MarkovDiaryGenerator:
    """
    The Markov model which describes the probability that an individual follows her routine and visits a typical location
    at the usual time, or she breaks the routine and visits another location.

    :param name: str
        name of the object

    :ivar: markov_chain_: dict
        the trained markov chain

    :ivar time_slot_length: str
        length of the time slot (1h)

    References:
        .. [pappalardo2017data] Pappalardo, L., Simini, F. "Data-driven generation of spatio-temporal routines in human mobility.", Data Mining and Knowledge Discovery, 32:3, 2017.

    """
    def __init__(self, name='Markov diary'):
        self._markov_chain_ = None
        self._time_slot_length = '1h'
        self._name = name

    @property
    def markov_chain_(self):
        return self._markov_chain_

    @property
    def time_slot_length(self):
        return self._time_slot_length

    @property
    def name(self):
        return self._name

    def _create_empty_markov_chain(self):
        """
        Create an empty Markov chain, i.e., a matrix 48 * 48 where an element M(i,j) is a pair of pairs
        ((h_i, b_i), (h_j, b_j)), h_i, h_j \in {0, ..., 23} and b_i, b_j \in {0, 1}
        """
        self._markov_chain_ = defaultdict(lambda: defaultdict(float))
        for h1 in range(0, 24):
            for r1 in [0, 1]:
                for h2 in range(0, 24):
                    for r2 in [0, 1]:
                        self._markov_chain_[(h1, r1)][(h2, r2)] = 0.0

    @staticmethod
    def _select_loc(individual_df, location2frequency, location_column='location'):

        if isinstance(individual_df[location_column], str): # if there is at least a location in the time slot
            locations = individual_df[location_column].split(',')

            if len(locations) == 1:  # if there just one location, then assign that location to the time slot
                return locations[0]

            elif len(set(Counter(locations).values())) == 1:  # if there are multiple locations, with the same frequency
                return sorted({k: location2frequency[k] for k in locations}.items(),
                              key=operator.itemgetter(1),
                              reverse=True)[0][0]  # return the locations with the highest overall frequency

            else: # if there multiple location in the same slot, with different frequencies
                l = sorted(Counter(locations).items(), key=operator.itemgetter(1), reverse=True)
                return l[0][0]

        # if there is no location in the time slot, return NaN
        return np.nan

    @staticmethod
    def _get_location2frequency(traj, location_column='location'):
        """
        Compute the visitation frequency and rank of each location of an individual

        :param traj: the trajectories of the individuals
        :type traj: pandas DataFrame

        :return: tuple
            a tuple of two dictionaries of locations to the corresponding visitation frequency and rank, respectively
        """
        location2frequency, location2rank = defaultdict(int), defaultdict(int)
        for i, row in traj.iterrows():
            if isinstance(row[location_column], str):  # if it is not NaN
                for location in row[location_column].split(','):
                    # we can have more than one location in a time slot
                    # so, every slot has a comma separated list of locations
                    location2frequency[location] += 1

        # compute location2rank
        rank = 1
        for loc in sorted(location2frequency.items(), key=operator.itemgetter(1), reverse=True):
            location, frequency = loc
            location2rank[location] = rank
            rank += 1
        return location2frequency, location2rank

    def _create_time_series(self, traj, lid='location'):#start_date, end_date, lid='location'):
        """
        Parameters
        ----------
        :param traj: the trips of an individual
        :type traj: pandas DataFrame

        :param start_date: datetime
            the initial date of the simulation

        :param end_date: datetime
            the end date of the simulation

        :return: the time series of the abstract locations visited by the individual
        :rtype: pandas DataFrame
        """
        # lat_lng_df = traj[['lat', 'lng']].drop_duplicates(subset=['lat', 'lng'])
        # lat_lng_df[lid] = np.arange(len(lat_lng_df)).astype('str')
        # traj = pd.merge(traj, lat_lng_df, on=['lat', 'lng'])

        shift = traj[constants.DATETIME].min().hour
        traj = traj[[constants.DATETIME, lid]].set_index(date_time)
        traj[lid] = traj[lid].astype('str')
        # enlarge (eventually) the time series with the specified freq (replace empty time slots with NaN)
        traj = traj.groupby(pd.Grouper(freq=self._time_slot_length, closed='left')).aggregate(lambda x: ','.join(x)).replace('', np.nan)


        # compute the frequency of every location visited by the individual
        location2frequency, location2rank = self._get_location2frequency(traj, location_column=lid)

        # select the location for every slot
        # ix = pd.DatetimeIndex(start=start_date, end=end_date, freq=self._time_slot_length)
        #ix = pd.date_range(start=start_date, end=end_date, freq=self._time_slot_length)
        time_series = traj.apply(lambda x: self._select_loc(x, location2frequency, location_column=lid), axis=1)#.reindex(ix)

        # fill the slots with NaN with the previous element or the next element ###
        time_series.fillna(method='ffill', inplace=True)
        time_series.fillna(method='bfill', inplace=True)

        # you can use location2frequency to assign a number to every location
        time_series = time_series.apply(lambda x: location2rank[x])
        return time_series, shift

    def _update_markov_chain(self, time_series, shift=0):
        """
        Update the Markov Chain by including the behavior of an individual

        :param time_series: time series of abstract locations visisted by an individual
        :type traj: pandas DataFrame
        """
        HOME = 1
        TYPICAL, NON_TYPICAL = 1, 0

        n = len(time_series)  # n is the length of the time series of the individual
        slot = 0  # it starts from the first slot in the time series

        while slot < n - 1:  # scan the time series of the individual, time slot by time slot

            #h = (slot % 24)
            h = (slot  + shift) % 24  # h, the hour of the day
            next_h = (h + 1) % 24  # next_h, the next hour of the day

            loc_h = time_series[slot]  # loc_h  ,   abstract location at the current slot
            next_loc_h = time_series[slot + 1]  # d_{h+1},   abstract location at the next slot

            if loc_h == HOME:  # if \delta(loc_h, t_h) == 1, i.e., she stays at home

                # we have two cases
                if next_loc_h == HOME:  # if \delta(d_{h + 1}, t_{h + 1}) == 1

                    # we are in Type1: (h, 1) --> (h + 1, 1)
                    self._markov_chain_[(h, TYPICAL)][(next_h, TYPICAL)] += 1

                else:  # she will be not in the typical location

                    # we are in Type2: (h, 1) --> (h + tau, 0)
                    tau = 1
                    if slot + 2 < n:  # if slot is the second last in the time series

                        for j in range(slot + 2, n):  # in slot + 1 we do not have HOME so we start from slot + 2
                            loc_hh = time_series[j]
                            if loc_hh == next_loc_h:  # if \delta(d_{h + j}, d_{h + 1}) == 1
                                tau += 1
                            else:
                                break

                        h_tau = (h + tau) % 24
                        # update the state of edge (h, 1) --> (h + tau, 0)
                        self._markov_chain_[(h, TYPICAL)][(h_tau, NON_TYPICAL)] += 1
                        slot = j - 2 #1

                    else:  # terminate the while cycle
                        slot = n

            else:  # loc_h != HOME

                if next_loc_h == HOME:  # if \delta(d_{h + 1}, t_{h + 1}) == 1, i.e., she will stay at home

                    # we are in Type3: (h, 0) --> (h + 1, 1)
                    self._markov_chain_[(h, NON_TYPICAL)][(next_h, TYPICAL)] += 1

                else:

                    # we are in Type 4: (h, 0) --> (h + tau, 0)
                    tau = 1
                    if slot + 2 < n:

                        for j in range(slot + 2, n):
                            loc_hh = time_series[j]
                            if loc_hh == next_loc_h:  # if \delta(d_{h + j}, d_{h + 1}) == 1
                                tau += 1
                            else:
                                break

                        h_tau = (h + tau) % 24

                        # update the state of edge (h, 0) --> (h + tau, 0)
                        self._markov_chain_[(h, NON_TYPICAL)][(h_tau, NON_TYPICAL)] += 1
                        slot = j - 2 #1

                    else:
                        slot = n

            slot += 1

    def _normalize_markov_chain(self):
        """
        Transform the dictionary into a proper Markov chain, i.e., normalize by row in order
        to obtain transition probabilities.
        """
        # compute the probabilities of the Markov chain, i.e. normalize by row
        for state1 in self._markov_chain_:
            tot = sum([prob for prob in self._markov_chain_[state1].values()])
            for state2 in self._markov_chain_[state1]:
                if tot != 0.0:
                    self._markov_chain_[state1][state2] /= tot

    def fit(self, traj, n_individuals, lid='location'): #start_date, end_date
        """
        Fit a Markov chain diary based on the trajectories in input

        :param traj: the trajectories of the individuals
        :type traj: pandas DataFrame

        :param n_individuals: int
            the number of individuals to consider

        :param start_date: datetime
            the starting date

        :param end_date: datetime
            the ending date
        """
        self._create_empty_markov_chain()  # initialize the markov chain

        individuals = traj.uid.unique()  # list of individuals' identifiers
        with tqdm(total=n_individuals) as pbar:

            for individual in individuals[:n_individuals]:

                # create the time series of the individual
                time_series, shift = self._create_time_series(traj[traj.uid == individual], lid=lid) # start_date, end_date,

                # update the markov chain according to the individual's time series
                self._update_markov_chain(time_series, shift)

                pbar.update(1)

        # normalize the markov chain, i.e., normalize the transitions by row
        self._normalize_markov_chain()

    @staticmethod
    def _weighted_random_selection(weights):
        """
        Choose an index from the list of weights according to the numbers in the list

        Parameters
        ----------
        weights: list
            a list of weights (e.g., probabilities)

        Returns
        -------
        index: int
            the index of element chosen from the list
        """
        # totals = []
        # running_total = 0
        #
        # for w in weights:
        #     running_total += w
        #     totals.append(running_total)
        #
        # rnd = random.random() * running_total
        # for index, total in enumerate(totals):
        #     if rnd < total:
        #         return index
        # return len(totals) - 1
        return np.searchsorted(np.cumsum(weights), random.random())


    def generate(self, diary_length, start_date):
        """
        Start the simulation of the mobility diary given the diary_length

        :param diary_length: int
            the length of the diary in the time unit of the MD Markov chain

        :param start_date: datetime
            the starting date

        :return: the generated diarySeries
        :rtype: pandas DataFrame
        """
        current_date = start_date
        V, i = [], 0
        prev_state = (i, 1)  # it starts from the typical location at midnight
        V.append(prev_state)

        while i < diary_length:

            h = i % 24  # the hour of the day

            # select the next state in the Markov chain
            p = list(self._markov_chain_[prev_state].values())
            if sum(p) == 0.:
                hh, rr = prev_state
                next_state = ((hh + 1) % 24, rr)
            else:
                index = self._weighted_random_selection(p)
                next_state = list(self._markov_chain_[prev_state].keys())[index]
            V.append(next_state)

            j = next_state[0]
            if j > h:  # we are in the same day
                i += j - h
            else:  # we are in the next day
                i += 24 - h + j

            prev_state = next_state

        # now we translate the temporal diary into the the mobility diary
        prev, diary, other_count = V[0], [], 1
        diary.append([current_date, 0])

        for v in V[1:]:  # scan all the states obtained and create the synthetic time series
            h, s = v
            h_prev, s_prev = prev

            if s == 1:  # if in that hour she visits home
                current_date += datetime.timedelta(hours=1)
                diary.append([current_date, 0])
                other_count = 1
            else:  # if in that hour she does NOT visit home

                if h > h_prev:  # we are in the same day
                    j = h - h_prev
                else:  # we are in the next day
                    j = 24 - h_prev + h

                for i in range(0, j):
                    current_date += datetime.timedelta(hours=1)
                    diary.append([current_date, other_count])
                other_count += 1

            prev = v

        short_diary = []
        prev_location = -1
        for visit_date, abstract_location in diary[0: diary_length]:
            if abstract_location != prev_location:
                short_diary.append([visit_date, abstract_location])
            prev_location = abstract_location

        diary_df = pd.DataFrame(short_diary, columns=[date_time, 'abstract_location'])
        return diary_df

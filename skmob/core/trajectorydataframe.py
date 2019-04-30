import pandas as pd
from ..utils import constants, plot
import numpy as np
from warnings import warn


class TrajSeries(pd.Series):

    @property
    def _constructor(self):
        return TrajSeries

    @property
    def _constructor_expanddim(self):
        return TrajDataFrame


class TrajDataFrame(pd.DataFrame):

    _metadata = ['_crs', '_parameters']

    def __init__(self, data, latitude=constants.LATITUDE, longitude=constants.LONGITUDE, datetime=constants.DATETIME,
                 user_id=constants.UID, trajectory_id=constants.TID,
                 timestamp=False, crs=constants.DEFAULT_CRS, parameters=None):

        original2default = {latitude: constants.LATITUDE,
                            longitude: constants.LONGITUDE,
                            datetime: constants.DATETIME,
                            user_id: constants.UID,
                            trajectory_id: constants.TID}

        columns = None

        if isinstance(data, pd.DataFrame):
            tdf = data.rename(columns=original2default)
            columns = tdf.columns

        # Dictionary
        elif isinstance(data, dict):
            tdf = pd.DataFrame.from_dict(data).rename(columns=original2default)
            columns = tdf.columns

        # List
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            tdf = data
            columns = []
            num_columns = len(data[0])
            for i in range(num_columns):
                try:
                    columns += [original2default[i]]
                except KeyError:
                    columns += [i]

        elif isinstance(data, pd.core.internals.BlockManager):
            tdf = data

        else:
            raise TypeError('DataFrame constructor called with incompatible data and dtype: {e}'.format(e=type(data)))

        super(TrajDataFrame, self).__init__(tdf, columns=columns)

        if parameters is None:
            # Init empty prop dictionary
            self._parameters = {}
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise AttributeError("Parameters must be a dictionary.")

        # Check crs consistency
        if crs is None:
            warn("crs will be set to the default crs WGS84 (EPSG:4326).")

        if not isinstance(crs, dict):
            raise TypeError('crs must be a dict type.')

        self._crs = crs

        if not isinstance(data, pd.core.internals.BlockManager):
            if timestamp:
                self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME], unit='s')
            if not pd.core.dtypes.common.is_datetime64_any_dtype(self[constants.DATETIME].dtype):
                self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME])
            if not pd.core.dtypes.common.is_float_dtype(self[constants.LONGITUDE].dtype):
                self[constants.LONGITUDE] = self[constants.LONGITUDE].astype('float')
            if not pd.core.dtypes.common.is_float_dtype(self[constants.LATITUDE].dtype):
                self[constants.LATITUDE] = self[constants.LATITUDE].astype('float')

    def _valid_trajdataframe(self):

        if (constants.LATITUDE not in self) or (constants.LONGITUDE not in self) or (constants.DATETIME not in self):
            raise AttributeError("No point data set yet (expected in column '%s, %s, % s')."
                                 % (constants.LATITUDE, constants.LONGITUDE, constants.DATETIME))

    @classmethod
    def from_file(cls, filename, latitude=constants.LATITUDE, longitude=constants.LONGITUDE, datetime=constants.DATETIME,
                  user_id=constants.UID, trajectory_id=constants.TID,
                  usecols=None, header='infer', timestamp=False, crs=constants.DEFAULT_CRS, sep="\t", parameters=None):

        df = pd.read_csv(filename, sep=sep, header=header, usecols=usecols)

        if parameters is None:
            # Init prop dictionary
            parameters = {'from_file': filename}

        return cls(df, latitude=latitude, longitude=longitude, datetime=datetime, user_id=user_id,
                   trajectory_id=trajectory_id, parameters=parameters, crs=crs, timestamp=timestamp)

    @property
    def lat(self):
        if constants.LATITUDE not in self:
            raise AttributeError("The TrajectoryDataFrame does not contain the column '%s.'" % constants.LATITUDE)
        # if not pd.core.dtypes.common.is_float_dtype(self[constants.LATITUDE].dtype):
        #     self[constants.LATITUDE] = self[constants.LATITUDE].astype('float')
        return self[constants.LATITUDE]

    @property
    def lng(self):
        if constants.LONGITUDE not in self:
            raise AttributeError("The TrajectoryDataFrame does not contain the column '%s.'"%constants.LONGITUDE)
        # if not pd.core.dtypes.common.is_float_dtype(self[constants.LONGITUDE].dtype):
        #     self[constants.LONGITUDE] = self[constants.LONGITUDE].astype('float')
        return self[constants.LONGITUDE]

    @property
    def datetime(self):
        if constants.DATETIME not in self:
            raise AttributeError("The TrajectoryDataFrame does not contain the column '%s.'"%constants.DATETIME)
        # if not pd.core.dtypes.common.is_datetime64_any_dtype(self[constants.DATETIME].dtype):
        #     self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME])
        return self[constants.DATETIME]

    @property
    def _constructor(self):
        return TrajDataFrame

    @property
    def _constructor_sliced(self):
        return TrajSeries

    @property
    def _constructor_expanddim(self):
        return TrajDataFrame

    def __finalize__(self, other, method=None, **kwargs):

        """propagate metadata from other to self """
        # merge operation: using metadata of the left object
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))

        # concat operation: using metadata of the first object
        elif method == 'concat':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))

        return self

    def set_parameter(self, key, param):

        self._parameters[key] = param

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, crs):
        self._crs = crs

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):

        # TODO: check if parameters are correct
        self._parameters = dict(parameters)

    # Sorting
    def sort_by_uid_and_datetime(self):
        self.sort_values(by=[constants.UID, constants.DATETIME], ascending=[True, True], inplace=True)

    # Plot methods
    def plot_trajectory(self, map_f=None, max_users=10, max_points=1000, imin=0, imax=-1, tiles='OpenStreetMap',
                        zoom=12, hex_color=-1, weight=2, opacity=0.75):
        return plot.plot_trajectory(self, map_f=map_f, max_users=max_users, max_points=max_points, imin=imin, imax=imax,
                                    tiles=tiles, zoom=zoom, hex_color=hex_color, weight=weight, opacity=opacity)

    def plot_stops(self, map_f=None, max_users=10, tiles='OpenStreetMap', zoom=12, hex_color=-1, opacity=0.3,
                   popup=True):
        return plot.plot_stops(self, map_f=map_f, max_users=max_users, tiles=tiles, zoom=zoom,
                               hex_color=hex_color, opacity=opacity, popup=popup)

    def plot_diary(self, user, start_datetime=None, end_datetime=None, ax=None):
        return plot.plot_diary(self, user, start_datetime=start_datetime, end_datetime=end_datetime, ax=ax)

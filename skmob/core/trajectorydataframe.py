import pandas as pd
from ..utils import constants, plot, utils
import numpy as np
from warnings import warn
from shapely.geometry import Polygon, Point
import geopandas as gpd
from .flowdataframe import FlowDataFrame
from skmob.preprocessing import routing


class TrajSeries(pd.Series):

    @property
    def _constructor(self):
        return TrajSeries

    @property
    def _constructor_expanddim(self):
        return TrajDataFrame


class TrajDataFrame(pd.DataFrame):

    _metadata = ['_parameters','_crs']

    def __init__(self, data, latitude=constants.LATITUDE, longitude=constants.LONGITUDE, datetime=constants.DATETIME,
                 user_id=constants.UID, trajectory_id=constants.TID,
                 timestamp=False, crs={"init": "epsg:4326"}, parameters={}):

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

        # Check crs consistency
        if crs is None:
            warn("crs will be set to the default crs WGS84 (EPSG:4326).")

        if not isinstance(crs, dict):
            raise TypeError('crs must be a dict type.')

        self._crs = crs

        if not isinstance(parameters, dict):
            raise AttributeError("parameters must be a dictionary.")

        self._parameters = parameters

        if self._is_trajdataframe():
            self._set_traj(timestamp=timestamp, inplace=True)

    def _is_trajdataframe(self):

        if (constants.DATETIME in self) and (constants.LATITUDE in self) and (constants.LONGITUDE in self):
            return True

        return False

    def _set_traj(self, timestamp=False, inplace=False):

        if not inplace:
            frame = self.copy()
        else:
            frame = self

        if timestamp:
            frame[constants.DATETIME] = pd.to_datetime(frame[constants.DATETIME], unit='s')

        if not pd.core.dtypes.common.is_datetime64_any_dtype(frame[constants.DATETIME].dtype):
            frame[constants.DATETIME] = pd.to_datetime(frame[constants.DATETIME])

        if not pd.core.dtypes.common.is_float_dtype(frame[constants.LONGITUDE].dtype):
            frame[constants.LONGITUDE] = frame[constants.LONGITUDE].astype('float')

        if not pd.core.dtypes.common.is_float_dtype(frame[constants.LATITUDE].dtype):
            frame[constants.LATITUDE] = frame[constants.LATITUDE].astype('float')

        frame.parameters = self._parameters
        frame.crs = self._crs

        if not inplace:
            return frame

    def to_flowdataframe(self, tessellation, remove_na=False, self_loops=True):
        """

        :param tessellation:
        :param remove_na:
        :param self_loop: if True, it counts self movements (default True)
        :return:
        """

        # Step 1: order the dataframe by user_id, traj_id, datetime
        self.sort_values(by=self.__operate_on(), ascending=True, inplace=True)

        # Step 2: map the trajectory onto the tessellation
        flow = self.mapping(tessellation, remove_na=remove_na)

        # Step 3: groupby tile_id and sum to obtain the flow
        flow.loc[:, 'destination'] = flow[constants.TILE_ID].shift(-1)
        flow = flow.groupby([constants.TILE_ID, 'destination']).size().reset_index(name=constants.FLOW)
        flow.rename(columns={constants.TILE_ID: constants.ORIGIN}, inplace=True)

        if not self_loops:
            flow = flow[flow[constants.ORIGIN] != flow[constants.DESTINATION]]

        return FlowDataFrame(flow, tessellation=tessellation)

    def to_geodataframe(self):

        gdf = gpd.GeoDataFrame(self, geometry=gpd.points_from_xy(self[constants.LONGITUDE], self[constants.LATITUDE]),
                               crs=self._crs)

        return gdf

    def mapping(self, tessellation, remove_na=False):
        """
        Method to assign to each point of the TrajDataFrame a corresponding tile_id of a given tessellation.
        :param tessellation: GeoDataFrame containing a tessellation (geometry of points or polygons).
        :param remove_na: (default False) it removes points that do not have a corresponding tile_id
        :return: TrajDataFrame with an additional column containing the tile_ids.
        """

        gdf = self.to_geodataframe()

        if all(isinstance(x, Polygon) for x in tessellation.geometry):

            if remove_na:
                how = 'inner'
            else:
                how = 'left'

            tile_ids = gpd.sjoin(gdf, tessellation, how=how, op='within')[[constants.TILE_ID]]

        elif all(isinstance(x, Point) for x in tessellation.geometry):

            tile_ids = utils.nearest(gdf, tessellation, constants.TILE_ID)

        new_data = self._constructor(self).__finalize__(self) #TrajDataFrame(self)
        new_data = new_data.merge(tile_ids, right_index=True, left_index=True)

        return new_data

    def __getitem__(self, key):
        """
        It the result contains lat, lng and datetime, return a TrajDataFrame, else a pandas DataFrame.
        """
        result = super(TrajDataFrame, self).__getitem__(key)

        if (isinstance(result, TrajDataFrame)) and result._is_trajdataframe():
            result.__class__ = TrajDataFrame
            result.crs = self._crs
            result.parameters = self._parameters

        elif isinstance(result, TrajDataFrame) and not result._is_trajdataframe():
            result.__class__ = pd.DataFrame

        return result

    @classmethod
    def from_file(cls, filename, latitude=constants.LATITUDE, longitude=constants.LONGITUDE, datetime=constants.DATETIME,
                  user_id=constants.UID, trajectory_id=constants.TID,
                  usecols=None, header='infer', timestamp=False, crs={"init": "epsg:4326"}, sep="\t", parameters=None):

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
        return self[constants.LATITUDE]

    @property
    def lng(self):
        if constants.LONGITUDE not in self:
            raise AttributeError("The TrajectoryDataFrame does not contain the column '%s.'"%constants.LONGITUDE)
        return self[constants.LONGITUDE]

    @property
    def datetime(self):
        if constants.DATETIME not in self:
            raise AttributeError("The TrajectoryDataFrame does not contain the column '%s.'"%constants.DATETIME)
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

        self._parameters = parameters #dict(parameters)

    def __operate_on(self):
        """
        Check which optional fields are present and return a list of them plus mandatory fields to which apply
        built-in pandas functions such as sort_values or groupby.
        :return: list
        """

        cols = []

        if constants.UID in self:
            cols.append(constants.UID)
        if constants.TID in self:
            cols.append(constants.TID)

        cols.append(constants.DATETIME)

        return cols

    # Sorting
    def sort_by_uid_and_datetime(self):
        return self.sort_values(by=[constants.UID, constants.DATETIME], ascending=[True, True]) #, inplace=True)

    # Plot methods
    def plot_trajectory(self, map_f=None, max_users=10, max_points=1000, imin=0, imax=-1, tiles='cartodbpositron',
                        zoom=12, hex_color=-1, weight=2, opacity=0.75):
        return plot.plot_trajectory(self, map_f=map_f, max_users=max_users, max_points=max_points, imin=imin, imax=imax,
                                    tiles=tiles, zoom=zoom, hex_color=hex_color, weight=weight, opacity=opacity)

    def plot_stops(self, map_f=None, max_users=10, tiles='OpenStreetMap', zoom=12, hex_color=-1, opacity=0.3,
                   popup=True):
        return plot.plot_stops(self, map_f=map_f, max_users=max_users, tiles=tiles, zoom=zoom,
                               hex_color=hex_color, opacity=opacity, popup=popup)

    def plot_diary(self, user, start_datetime=None, end_datetime=None, ax=None):
        return plot.plot_diary(self, user, start_datetime=start_datetime, end_datetime=end_datetime, ax=ax)

    def route(self, G=None, index_origin=0, index_destin=-1):
        return routing.route(self, G=G, index_origin=index_origin, index_destin=index_destin)


def nparray_to_trajdataframe(trajectory_array, columns, parameters={}):
    df = pd.DataFrame(trajectory_array, columns=columns)
    tdf = TrajDataFrame(df, parameters=parameters)
    return tdf

def _dataframe_set_geometry(self, col, timestampe=False, drop=False, inplace=False, crs=None):
    if inplace:
        raise ValueError("Can't do inplace setting when converting from"
                         " DataFrame to GeoDataFrame")
    gf = TrajDataFrame(self)

    # this will copy so that BlockManager gets copied
    return gf._set_traj() #.set_geometry(col, drop=drop, inplace=False, crs=crs)

pd.DataFrame._set_traj = _dataframe_set_geometry
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

    _metadata = ['_parameters', '_crs'] # All the metadata that should be accessible must be also in the metadata method

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

        if self._has_traj_columns():
            self._set_traj(timestamp=timestamp, inplace=True)

    def _has_traj_columns(self):

        if (constants.DATETIME in self) and (constants.LATITUDE in self) and (constants.LONGITUDE in self):
            return True

        return False

    def _is_trajdataframe(self):

        if ((constants.DATETIME in self) and pd.core.dtypes.common.is_datetime64_any_dtype(self[constants.DATETIME]))\
                and ((constants.LONGITUDE in self) and pd.core.dtypes.common.is_float_dtype(self[constants.LONGITUDE])) \
                and ((constants.LATITUDE in self) and pd.core.dtypes.common.is_float_dtype(self[constants.LATITUDE])):

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
        flow.loc[:, constants.DESTINATION] = flow[constants.TILE_ID].shift(-1)
        flow = flow.groupby([constants.TILE_ID, constants.DESTINATION]).size().reset_index(name=constants.FLOW)
        flow.rename(columns={constants.TILE_ID: constants.ORIGIN}, inplace=True)

        if not self_loops:
            flow = flow[flow[constants.ORIGIN] != flow[constants.DESTINATION]]

        return FlowDataFrame(flow, tessellation=tessellation)

    def to_geodataframe(self):

        gdf = gpd.GeoDataFrame(self.copy(), geometry=gpd.points_from_xy(self[constants.LONGITUDE],
                                                                        self[constants.LATITUDE]), crs=self._crs)

        return gdf

    def mapping(self, tessellation, remove_na=False):
        """
        Method to assign to each point of the TrajDataFrame a corresponding tile_id of a given tessellation.
        :param tessellation: GeoDataFrame containing a tessellation (geometry of points or polygons).
        :param remove_na: (default False) it removes points that do not have a corresponding tile_id
        :return: TrajDataFrame with an additional column containing the tile_ids.
        """

        gdf = self.to_geodataframe()

        if constants.TILE_ID not in tessellation.columns:
            tessellation[constants.TILE_ID] = tessellation.index

        if all(isinstance(x, Polygon) for x in tessellation.geometry):

            if remove_na:
                how = 'inner'
            else:
                how = 'left'

            tile_ids = gpd.sjoin(gdf, tessellation, how=how, op='within')[[constants.TILE_ID]]

        elif all(isinstance(x, Point) for x in tessellation.geometry):

            tile_ids = utils.nearest(gdf, tessellation, constants.TILE_ID)

        new_data = self._constructor(self).__finalize__(self)
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


    def settings_from(self, trajdataframe):
        """
        Method to copy attributes from another TrajDataFrame.
        :param trajdataframe: TrajDataFrame from which copy the attributes.
        """
        for k in trajdataframe.metadata:
            value = getattr(trajdataframe, k)
            setattr(self, k, value)

    @classmethod
    def from_file(cls, filename, latitude=constants.LATITUDE, longitude=constants.LONGITUDE, datetime=constants.DATETIME,
                  user_id=constants.UID, trajectory_id=constants.TID,
                  usecols=None, header='infer', timestamp=False, crs={"init": "epsg:4326"}, sep=",", parameters=None):

        df = pd.read_csv(filename, sep=sep, header=header, usecols=usecols)

        if parameters is None:
            # Init prop dictionary
            parameters = {'from_file': filename}

        return cls(df, latitude=latitude, longitude=longitude, datetime=datetime, user_id=user_id,
                   trajectory_id=trajectory_id, parameters=parameters, crs=crs, timestamp=timestamp)

    @property
    def lat(self):
        if constants.LATITUDE not in self:
            raise AttributeError("The TrajDataFrame does not contain the column '%s.'" % constants.LATITUDE)
        return self[constants.LATITUDE]

    @property
    def lng(self):
        if constants.LONGITUDE not in self:
            raise AttributeError("The TrajDataFrame does not contain the column '%s.'"%constants.LONGITUDE)
        return self[constants.LONGITUDE]

    @property
    def datetime(self):
        if constants.DATETIME not in self:
            raise AttributeError("The TrajDataFrame does not contain the column '%s.'"%constants.DATETIME)
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

    @property
    def metadata(self):

        md = ['crs', 'parameters']    # Add here all the metadata that are accessible from the object
        return md

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

        self._parameters = dict(parameters)

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
        if constants.UID in self.columns:
            return self.sort_values(by=[constants.UID, constants.DATETIME], ascending=[True, True])
        else:
            return self.sort_values(by=[constants.DATETIME], ascending=[True])

    # Plot methods
    def plot_trajectory(self, map_f=None, max_users=10, max_points=1000, style_function=plot.traj_style_function,
                        tiles='cartodbpositron', zoom=12, hex_color=-1, weight=2, opacity=0.75, dashArray='0, 0',
                        start_end_markers=True):
        """
        :param map_f: folium.Map
            `folium.Map` object where the trajectory will be plotted. If `None`, a new map will be created.

        :param max_users: int
            maximum number of users whose trajectories should be plotted.

        :param max_points: int
            maximum number of points per user to plot.
            If necessary, a user's trajectory will be down-sampled to have at most `max_points` points.

        :param style_function: lambda function
            function specifying the style (weight, color, opacity) of the GeoJson object.

        :param tiles: str
            folium's `tiles` parameter.

        :param zoom: int
            initial zoom.

        :param hex_color: str or int
            hex color of the trajectory line. If `-1` a random color will be generated for each trajectory.

        :param weight: float
            thickness of the trajectory line.

        :param opacity: float
            opacity (alpha level) of the trajectory line.

        :param dashArray: str
            style of the trajectory line: '0, 0' for a solid trajectory line, '5, 5' for a dashed line
            (where dashArray='size of segment, size of spacing').

        :param start_end_markers: bool
            add markers on the start and end points of the trajectory.

        :return: `folium.Map` object with the plotted trajectories.

        """
        return plot.plot_trajectory(self, map_f=map_f, max_users=max_users, max_points=max_points,
                                    style_function=style_function, tiles=tiles, zoom=zoom, hex_color=hex_color,
                                    weight=weight, opacity=opacity, dashArray=dashArray,
                                    start_end_markers=start_end_markers)

    def plot_stops(self, map_f=None, max_users=10, tiles='cartodbpositron', zoom=12, hex_color=-1, opacity=0.3,
                   radius=12, number_of_sides=4, popup=True):
        """
        Requires a TrajDataFrame with stops or clusters, output of `preprocessing.detection.stops`
        or `preprocessing.clustering.cluster`. The column `constants.LEAVING_DATETIME` must be present.

        :param map_f: folium.Map
            `folium.Map` object where the stops will be plotted. If `None`, a new map will be created.

        :param max_users: int
            maximum number of users whose stops should be plotted.

        :param tiles: str
            folium's `tiles` parameter.

        :param zoom: int
            initial zoom.

        :param hex_color: str or int
            hex color of the stop markers. If `-1` a random color will be generated for each user.

        :param opacity: float
            opacity (alpha level) of the stop makers.

        :param radius: float
            size of the markers.

        :param number_of_sides: int
            number of sides of the markers.

        :param popup: bool
            if `True`, when clicking on a marker a popup window displaying information on the stop will appear.

        :return: `folium.Map` object with the plotted stops.

        """
        return plot.plot_stops(self, map_f=map_f, max_users=max_users, tiles=tiles, zoom=zoom,
                               hex_color=hex_color, opacity=opacity, radius=radius, number_of_sides=number_of_sides,
                               popup=popup)

    def plot_diary(self, uid, start_datetime=None, end_datetime=None, ax=None):
        """
        Requires a TrajDataFrame with clusters, output of `preprocessing.clustering.cluster`.
        The column `constants.CLUSTER` must be present.

        :param uid: str or int
            user ID whose diary should be plotted.

        :param start_datetime: datetime.datetime
            Only stops made after this date will be plotted.
            If `None` the datetime of the oldest stop will be selected.

        :param end_datetime: datetime.datetime
            Only stops made before this date will be plotted.
            If `None` the datetime of the newest stop will be selected.

        :param ax: matplotlib.axes
            axes where the diary will be plotted.

        :return: `matplotlib.axes` of the plotted diary.

        """
        return plot.plot_diary(self, uid, start_datetime=start_datetime, end_datetime=end_datetime, ax=ax)

    def route(self, G=None, index_origin=0, index_destin=-1):
        return routing.route(self, G=G, index_origin=index_origin, index_destin=index_destin)

    def timezone_conversion(self, from_timezone, to_timezone):
        """
        :param from_timezone: str
            current timezone (e.g. 'GMT')

        :param to_timezone: str
            new timezone (e.g. 'Asia/Shanghai')
        """
        self.rename(columns={'datetime': 'original_datetime'}, inplace=True)
        self['datetime'] = self['original_datetime']. \
            dt.tz_localize(from_timezone). \
            dt.tz_convert(to_timezone). \
            dt.tz_localize(None)
        self.drop(columns=['original_datetime'], inplace=True)


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

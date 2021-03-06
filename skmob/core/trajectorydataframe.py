import pandas as pd
from ..utils import constants, plot, utils
import numpy as np
from warnings import warn
from shapely.geometry import Polygon, Point
import geopandas as gpd
from .flowdataframe import FlowDataFrame
# from skmob.preprocessing import routing


class TrajSeries(pd.Series):

    @property
    def _constructor(self):
        return TrajSeries

    @property
    def _constructor_expanddim(self):
        return TrajDataFrame


class TrajDataFrame(pd.DataFrame):
    """TrajDataFrame.
    
    A TrajDataFrame object is a pandas.DataFrame that has three columns latitude, longitude and datetime. TrajDataFrame accepts the following keyword arguments:
    
    Parameters
    ----------
    data : list or dict or pandas DataFrame
        the data that must be embedded into a TrajDataFrame.
        
    latitude : int or str, optional
        the position or the name of the column in `data` containing the latitude. The default is `constants.LATITUDE`.
        
    longitude : int or str, optional
        the position or the name of the column in `data` containing the longitude. The default is `constants.LONGITUDE`.
        
    datetime : int or str, optional
        the position or the name of the column in `data` containing the datetime. The default is `constants.DATETIME`.
        
    user_id : int or str, optional
        the position or the name of the column in `data`containing the user identifier. The default is `constants.UID`.
        
    trajectory_id : int or str, optional
        the position or the name of the column in `data` containing the trajectory identifier. The default is `constants.TID`.
        
    timestamp : boolean, optional
        it True, the datetime is a timestamp. The default is `False`.
        
    crs : dict, optional
        the coordinate reference system of the geographic points. The default is `{"init": "epsg:4326"}`.
        
    parameters : dict, optional
        parameters to add to the TrajDataFrame. The default is `{}` (no parameters).
        
    Examples
    --------
    >>> import skmob
    >>> # create a TrajDataFrame from a list
    >>> data_list = [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'], [1, 39.984198, 116.319322, '2008-10-23 13:53:06'], [1, 39.984224, 116.319402, '2008-10-23 13:53:11'], [1, 39.984211, 116.319389, '2008-10-23 13:53:16']]
    >>> tdf = skmob.TrajDataFrame(data_list, latitude=1, longitude=2, datetime=3)
    >>> print(tdf.head())
       0        lat         lng            datetime
    0  1  39.984094  116.319236 2008-10-23 13:53:05
    1  1  39.984198  116.319322 2008-10-23 13:53:06
    2  1  39.984224  116.319402 2008-10-23 13:53:11
    3  1  39.984211  116.319389 2008-10-23 13:53:16
    >>> print(type(tdf))
    <class 'skmob.core.trajectorydataframe.TrajDataFrame'>
    >>> 
    >>> # create a TrajDataFrame from a pandas DataFrame
    >>> import pandas as pd
    >>> # create a DataFrame from the previous list 
    >>> data_df = pd.DataFrame(data_list, columns=['user', 'latitude', 'lng', 'hour'])
    >>> print(type(data_df))
    <class 'pandas.core.frame.DataFrame'>
    >>> tdf = skmob.TrajDataFrame(data_df, latitude='latitude', datetime='hour', user_id='user')
    >>> print(type(tdf))
    <class 'skmob.core.trajectorydataframe.TrajDataFrame'>
    >>> print(tdf.head())
       uid        lat         lng            datetime
    0    1  39.984094  116.319236 2008-10-23 13:53:05
    1    1  39.984198  116.319322 2008-10-23 13:53:06
    2    1  39.984224  116.319402 2008-10-23 13:53:11
    3    1  39.984211  116.319389 2008-10-23 13:53:16
    """
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

    def to_flowdataframe(self, tessellation, self_loops=True):
        """
        Aggregate a TrajDataFrame into a FlowDataFrame.
        The points that do not have a corresponding tile in the spatial tessellation are removed.
        
        Parameters
        ----------
        tessellation : GeoDataFrame
            the spatial tessellation to use to aggregate the points.
            
        self_loops : boolean
            if True, it counts movements that start and end in the same tile. The default is `True`.
        
        Returns
        -------
        FlowDataFrame
            the FlowDataFrame obtained as an aggregation of the TrajDataFrame
        
        Examples
        --------
        >>> import skmob
        >>> from skmob.tessellation import tilers
        >>> import pandas as pd
        >>> from skmob.preprocessing import filtering
        >>> # read the trajectory data (GeoLife, Beijing, China)
        >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
        >>> df = pd.read_csv(url, sep=',', compression='gzip')
        >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
        >>> print(tdf.head())
                 lat         lng            datetime  uid
        0  39.984094  116.319236 2008-10-23 05:53:05    1
        1  39.984198  116.319322 2008-10-23 05:53:06    1
        2  39.984224  116.319402 2008-10-23 05:53:11    1
        3  39.984211  116.319389 2008-10-23 05:53:16    1
        4  39.984217  116.319422 2008-10-23 05:53:21    1
        >>> # build a tessellation over the city
        >>> tessellation = tilers.tiler.get("squared", base_shape="Beijing, China", meters=15000)
        >>> # counting movements that start and end in the same tile
        >>> fdf = tdf.to_flowdataframe(tessellation=tessellation, self_loops=True)
        >>> print(fdf.head())
          origin destination  flow
        0     49          49   788
        1     49          62     1
        2     50          50  4974
        3     50          63     1
        4     61          61   207
        
        See Also
        --------
        FlowDataFrame
        """

        # Step 1: order the dataframe by user_id, traj_id, datetime
        self.sort_values(by=self.__operate_on(), ascending=True, inplace=True)

        # Step 2: map the trajectory onto the tessellation
        flow = self.mapping(tessellation, remove_na=False)

        # Step 3: groupby tile_id and sum to obtain the flow
        flow.loc[:, constants.DESTINATION] = flow[constants.TILE_ID].shift(-1)
        # excluding rows with points of different users
        flow.loc[:, 'next_uid'] = flow[constants.UID].shift(-1)
        flow = flow.loc[flow['uid'] == flow['next_uid']]

        flow = flow.groupby([constants.TILE_ID, constants.DESTINATION], dropna=True).size().reset_index(name=constants.FLOW)
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
        Assign each point of the TrajDataFrame to the corresponding tile of a spatial tessellation.
        
        Parameters
        ----------
        tessellation : GeoDataFrame
            the spatial tessellation containing a geometry column with points or polygons.
            
        remove_na : boolean, optional
            if `True`, remove points that do not have a corresponding tile in the spatial tessellation. The default is `False`.
        
        Returns
        -------
        TrajDataFrame
            a TrajDataFrame with an additional column `tile_ID` indicating the tile identifiers.
        
        Examples
        --------
        >>> import skmob
        >>> from skmob.tessellation import tilers
        >>> import pandas as pd
        >>> from skmob.preprocessing import filtering
        >>> # read the trajectory data (GeoLife, Beijing, China)
        >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
        >>> df = pd.read_csv(url, sep=',', compression='gzip')
        >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
        >>> print(tdf.head())
                 lat         lng            datetime  uid
        0  39.984094  116.319236 2008-10-23 05:53:05    1
        1  39.984198  116.319322 2008-10-23 05:53:06    1
        2  39.984224  116.319402 2008-10-23 05:53:11    1
        3  39.984211  116.319389 2008-10-23 05:53:16    1
        4  39.984217  116.319422 2008-10-23 05:53:21    1
        >>> # build a tessellation over the city
        >>> tessellation = tilers.tiler.get("squared", base_shape="Beijing, China", meters=15000)
        >>> mtdf = tdf.mapping(tessellation)
        >>> print(mtdf.head())
                 lat         lng            datetime  uid tile_ID
        0  39.984094  116.319236 2008-10-23 05:53:05    1      63
        1  39.984198  116.319322 2008-10-23 05:53:06    1      63
        2  39.984224  116.319402 2008-10-23 05:53:11    1      63
        3  39.984211  116.319389 2008-10-23 05:53:16    1      63
        4  39.984217  116.319422 2008-10-23 05:53:21    1      63        
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

        new_data = self._constructor(self).__finalize__(self)
        new_data = new_data.merge(tile_ids, right_index=True, left_index=True)

        return new_data

    def __getitem__(self, key):
        """
        If the result contains lat, lng and datetime, return a TrajDataFrame, else a pandas DataFrame.
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
        Copy the attributes from another TrajDataFrame.
        
        Parameters
        ----------
        trajdataframe : TrajDataFrame 
            the TrajDataFrame from which to copy the attributes.
            
        Examples
        --------
        >>> import skmob
        >>> import pandas as pd
        >>> # read the trajectory data (GeoLife, Beijing, China)
        >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
        >>> df = pd.read_csv(url, sep=',', compression='gzip')
        >>> tdf1 = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
        >>> tdf1 = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
        >>> print(tdf1.parameters)
        {}
        >>> tdf2.parameters['hasProperty'] = True
        >>> print(tdf2.parameters)
        {'hasProperty': True}
        >>> tdf1.settings_from(tdf2)
        >>> print(tdf1.parameters)
        {'hasProperty': True}
        """
        for k in trajdataframe.metadata:
            value = getattr(trajdataframe, k)
            setattr(self, k, value)

    @classmethod
    def from_file(cls, filename, latitude=constants.LATITUDE, longitude=constants.LONGITUDE, datetime=constants.DATETIME,
                  user_id=constants.UID, trajectory_id=constants.TID, encoding=None,
                  usecols=None, header='infer', timestamp=False, crs={"init": "epsg:4326"}, sep=",", parameters=None):

        df = pd.read_csv(filename, sep=sep, header=header, usecols=usecols, encoding=encoding)

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
                        tiles='cartodbpositron', zoom=12, hex_color=None, weight=2, opacity=0.75, dashArray='0, 0',
                        start_end_markers=True, control_scale=True):

        """
        Plot the trajectories on a Folium map.
        
        Parameters
        ----------
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
    
        :param hex_color: str
            hex color of the trajectory line. If `None` a random color will be generated for each trajectory.
    
        :param weight: float
            thickness of the trajectory line.
    
        :param opacity: float
            opacity (alpha level) of the trajectory line.
    
        :param dashArray: str
            style of the trajectory line: '0, 0' for a solid trajectory line, '5, 5' for a dashed line
            (where dashArray='size of segment, size of spacing').
    
        :param start_end_markers: bool
            add markers on the start and end points of the trajectory.
    
        :param control_scale: bool
            if `True`, add scale information in the bottom left corner of the visualization. The default is `True`.

        Returns
        -------
            `folium.Map` object with the plotted trajectories.


        Examples
        --------
        >>> import skmob
        >>> import pandas as pd
        >>> # read the trajectory data (GeoLife, Beijing, China)
        >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
        >>> df = pd.read_csv(url, sep=',', compression='gzip')
        >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
        >>> print(tdf.head())
                 lat         lng            datetime  uid
        0  39.984094  116.319236 2008-10-23 05:53:05    1
        1  39.984198  116.319322 2008-10-23 05:53:06    1
        2  39.984224  116.319402 2008-10-23 05:53:11    1
        3  39.984211  116.319389 2008-10-23 05:53:16    1
        4  39.984217  116.319422 2008-10-23 05:53:21    1
        >>> m = tdf.plot_trajectory(zoom=12, weight=3, opacity=0.9, tiles='Stamen Toner')
        >>> m
        
        .. image:: https://raw.githubusercontent.com/scikit-mobility/scikit-mobility/master/examples/plot_trajectory_example.png
        """

        return plot.plot_trajectory(self, map_f=map_f, max_users=max_users, max_points=max_points, style_function=style_function,
                    tiles=tiles, zoom=zoom, hex_color=hex_color, weight=weight, opacity=opacity, dashArray=dashArray,
                    start_end_markers=start_end_markers, control_scale=control_scale)

    def plot_stops(self, map_f=None, max_users=10, tiles='cartodbpositron', zoom=12, hex_color=None, opacity=0.3,
                   radius=12, number_of_sides=4, popup=True, control_scale=True):

        """
        Plot the stops in the TrajDataFrame on a Folium map. This function requires a TrajDataFrame with stops or clusters, output of `preprocessing.detection.stops` or `preprocessing.clustering.cluster` functions. The column `constants.LEAVING_DATETIME` must be present.
        
        Parameters
        ----------
        :param map_f: folium.Map
            `folium.Map` object where the stops will be plotted. If `None`, a new map will be created.
    
        :param max_users: int
            maximum number of users whose stops should be plotted.
    
        :param tiles: str
            folium's `tiles` parameter.
    
        :param zoom: int
            initial zoom.
    
        :param hex_color: str
            hex color of the stop markers. If `None` a random color will be generated for each user.
    
        :param opacity: float
            opacity (alpha level) of the stop makers.
    
        :param radius: float
            size of the markers.
    
        :param number_of_sides: int
            number of sides of the markers.
    
        :param popup: bool
            if `True`, when clicking on a marker a popup window displaying information on the stop will appear.
            The default is `True`.
        
        :param control_scale: bool
            if `True`, add scale information in the bottom left corner of the visualization. The default is `True`.

        Returns
        -------
            `folium.Map` object with the plotted stops.

        Examples
        --------
        >>> import skmob
        >>> from skmob.preprocessing import detection
        >>> import pandas as pd
        >>> # read the trajectory data (GeoLife, Beijing, China)
        >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
        >>> df = pd.read_csv(url, sep=',', compression='gzip')
        >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
        >>> print(tdf.head())
                 lat         lng            datetime  uid
        0  39.984094  116.319236 2008-10-23 05:53:05    1
        1  39.984198  116.319322 2008-10-23 05:53:06    1
        2  39.984224  116.319402 2008-10-23 05:53:11    1
        3  39.984211  116.319389 2008-10-23 05:53:16    1
        4  39.984217  116.319422 2008-10-23 05:53:21    1
        >>> stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2, leaving_time=True)
        >>> print(stdf.head())
                 lat         lng            datetime  uid    leaving_datetime
        0  39.978030  116.327481 2008-10-23 06:01:37    1 2008-10-23 10:32:53
        1  40.013820  116.306532 2008-10-23 11:10:19    1 2008-10-23 23:45:27
        2  39.978419  116.326870 2008-10-24 00:21:52    1 2008-10-24 01:47:30
        3  39.981166  116.308475 2008-10-24 02:02:31    1 2008-10-24 02:30:29
        4  39.981431  116.309902 2008-10-24 02:30:29    1 2008-10-24 03:16:35 
        >>> map_f = tdf.plot_trajectory(max_points=1000, start_end_markers=False)
        >>> stdf.plot_stops(map_f=map_f)
        
        .. image:: https://raw.githubusercontent.com/scikit-mobility/scikit-mobility/master/examples/plot_stops_example.png
        """
        return plot.plot_stops(self, map_f=map_f, max_users=max_users, tiles=tiles, zoom=zoom,
                               hex_color=hex_color, opacity=opacity, radius=radius, number_of_sides=number_of_sides,
                               popup=popup, control_scale=control_scale)


    def plot_diary(self, user, start_datetime=None, end_datetime=None, ax=None, legend=False):
        """
        Plot a mobility diary of an individual in a TrajDataFrame. It requires a TrajDataFrame with clusters, output of `preprocessing.clustering.cluster`. The column `constants.CLUSTER` must be present.

        Parameters
        ----------
        user : str or int
            user identifier whose diary should be plotted.

        start_datetime : datetime.datetime, optional
            only stops made after this date will be plotted. If `None` the datetime of the oldest stop will be selected. The default is `None`.

        end_datetime : datetime.datetime, optional
            only stops made before this date will be plotted. If `None` the datetime of the newest stop will be selected. The default is `None`.

        ax : matplotlib.axes, optional
            axes where the diary will be plotted. If `None` a new ax is created. The default is `None`.

        legend : bool, optional
            If `True`, legend with cluster IDs is shown. The default is `False`.
        
        Returns
        -------
        matplotlib.axes
            the `matplotlib.axes` object of the plotted diary.
        
        Examples
        --------
        >>> import skmob
        >>> from skmob.preprocessing import detection, clustering
        >>> import pandas as pd
        >>> # read the trajectory data (GeoLife, Beijing, China)
        >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
        >>> df = pd.read_csv(url, sep=',', compression='gzip')
        >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
        >>> print(tdf.head())
                 lat         lng            datetime  uid
        0  39.984094  116.319236 2008-10-23 05:53:05    1
        1  39.984198  116.319322 2008-10-23 05:53:06    1
        2  39.984224  116.319402 2008-10-23 05:53:11    1
        3  39.984211  116.319389 2008-10-23 05:53:16    1
        4  39.984217  116.319422 2008-10-23 05:53:21    1
        >>> # detect stops
        >>> stdf = detection.stops(tdf, stop_radius_factor=0.5, minutes_for_a_stop=20.0, spatial_radius_km=0.2, leaving_time=True)
        >>> print(stdf.head())
                 lat         lng            datetime  uid    leaving_datetime
        0  39.978030  116.327481 2008-10-23 06:01:37    1 2008-10-23 10:32:53
        1  40.013820  116.306532 2008-10-23 11:10:19    1 2008-10-23 23:45:27
        2  39.978419  116.326870 2008-10-24 00:21:52    1 2008-10-24 01:47:30
        3  39.981166  116.308475 2008-10-24 02:02:31    1 2008-10-24 02:30:29
        4  39.981431  116.309902 2008-10-24 02:30:29    1 2008-10-24 03:16:35 
        >>> # cluster stops
        >>> cstdf = clustering.cluster(stdf, cluster_radius_km=0.1, min_samples=1)
        >>> print(cstdf.head())
                 lat         lng            datetime  uid    leaving_datetime  cluster
        0  39.978030  116.327481 2008-10-23 06:01:37    1 2008-10-23 10:32:53        0
        1  40.013820  116.306532 2008-10-23 11:10:19    1 2008-10-23 23:45:27        1
        2  39.978419  116.326870 2008-10-24 00:21:52    1 2008-10-24 01:47:30        0
        3  39.981166  116.308475 2008-10-24 02:02:31    1 2008-10-24 02:30:29       42
        4  39.981431  116.309902 2008-10-24 02:30:29    1 2008-10-24 03:16:35       41
        >>> # plot the diary of one individual
        >>> user = 1
        >>> start_datetime = pd.to_datetime('2008-10-23 030000')
        >>> end_datetime = pd.to_datetime('2008-10-30 030000')
        >>> ax = cstdf.plot_diary(user, start_datetime=start_datetime, end_datetime=end_datetime)
        
        .. image:: https://raw.githubusercontent.com/scikit-mobility/scikit-mobility/master/examples/plot_diary_example.png
        """
        return plot.plot_diary(self, user, start_datetime=start_datetime, end_datetime=end_datetime, ax=ax, legend=legend)

    # def route(self, G=None, index_origin=0, index_destin=-1):
    #     return routing.route(self, G=G, index_origin=index_origin, index_destin=index_destin)

    def timezone_conversion(self, from_timezone, to_timezone):
        """
        Convert the timezone of the datetime in the TrajDataFrame.
        
        Parameters
        ----------
        from_timezone : str
            the current timezone of the TrajDataFrame (e.g., 'GMT').

        to_timezone : str
            the new timezone of the TrajDataFrame (e.g., 'Asia/Shanghai').
            
        Examples
        --------
        >>> import skmob
        >>> import pandas as pd
        >>> # read the trajectory data (GeoLife, Beijing, China)
        >>> url = skmob.utils.constants.GEOLIFE_SAMPLE
        >>> df = pd.read_csv(url, sep=',', compression='gzip')
        >>> tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
        >>> print(tdf.head())
                 lat         lng            datetime  uid
        0  39.984094  116.319236 2008-10-23 05:53:05    1
        1  39.984198  116.319322 2008-10-23 05:53:06    1
        2  39.984224  116.319402 2008-10-23 05:53:11    1
        3  39.984211  116.319389 2008-10-23 05:53:16    1
        4  39.984217  116.319422 2008-10-23 05:53:21    1
        >>> tdf.timezone_conversion('GMT', 'Asia/Shanghai')
        >>> print(tdf.head())
                 lat         lng  uid            datetime
        0  39.984094  116.319236    1 2008-10-23 13:53:05
        1  39.984198  116.319322    1 2008-10-23 13:53:06
        2  39.984224  116.319402    1 2008-10-23 13:53:11
        3  39.984211  116.319389    1 2008-10-23 13:53:16
        4  39.984217  116.319422    1 2008-10-23 13:53:21
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

import pandas as pd
import geopandas as gpd
from ..utils import constants, utils, plot
import numpy as np
from warnings import warn
from ..tessellation.tilers import tiler
from shapely.geometry import Point, Polygon


class FlowSeries(pd.Series):

    @property
    def _constructor(self):
        return FlowSeries

    @property
    def _constructor_expanddim(self):
        return FlowDataFrame


class FlowDataFrame(pd.DataFrame):

    _metadata = ['_crs', '_tessellation', '_parameters']

    def __init__(self, data, origin=constants.ORIGIN, destination=constants.DESTINATION, flow=constants.FLOW,
                 datetime=constants.DATETIME, tile_id=constants.TILE_ID, timestamp=False, tessellation=None,
                 parameters={}):

        original2default = {origin: constants.ORIGIN,
                            destination: constants.DESTINATION,
                            flow: constants.FLOW,
                            datetime: constants.DATETIME}

        columns = None

        if isinstance(data, pd.DataFrame):
            fdf = data.rename(columns=original2default)
            columns = fdf.columns

        # Dictionary
        elif isinstance(data, dict):
            fdf = pd.DataFrame.from_dict(data).rename(columns=original2default)
            columns = fdf.columns

        # List
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            fdf = data
            columns = []
            num_columns = len(data[0])
            for i in range(num_columns):
                try:
                    columns += [original2default[i]]
                except KeyError:
                    columns += [i]

        elif isinstance(data, pd.core.internals.BlockManager):
            fdf = data

        else:
            raise TypeError('DataFrame constructor called with incompatible data and dtype: {e}'.format(e=type(data)))

        super(FlowDataFrame, self).__init__(fdf, columns=columns)

        if parameters is None:
            # Init empty prop dictionary
            self._parameters = {}
        elif isinstance(parameters, dict):
            self._parameters = parameters
        else:
            raise AttributeError("Parameters must be a dictionary.")

        if not isinstance(data, pd.core.internals.BlockManager):

            self[constants.ORIGIN] = self[constants.ORIGIN].astype('str')
            self[constants.DESTINATION] = self[constants.DESTINATION].astype('str')

            if tessellation is None:
                raise TypeError("tessellation must be a GeoDataFrame with tile_id and geometry.")

            elif isinstance(tessellation, gpd.GeoDataFrame):
                self._tessellation = tessellation.copy()
                self._tessellation.rename(columns={tile_id: constants.TILE_ID}, inplace=True)
                self._tessellation[constants.TILE_ID] = self._tessellation[constants.TILE_ID].astype('str')

                if tessellation.crs is None:
                    warn("The tessellation crs is None. It will be set to the default crs WGS84 (EPSG:4326).")

                # Check consistency
                origin = self[constants.ORIGIN]
                destination = self[constants.DESTINATION]

                if not all(origin.isin(self._tessellation[constants.TILE_ID])) or \
                        not all(destination.isin(self._tessellation[constants.TILE_ID])):
                    raise ValueError("Inconsistency - origin and destination IDs must be present in the tessellation.")

                # Cleaning the index to make sure it is incremental
                self._tessellation.reset_index(inplace=True, drop=True)

            else:
                raise TypeError("tessellation must be a GeoDataFrame with tile_id and geometry.")

            if datetime in columns:

                if timestamp:
                    self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME], unit='s')

                if not pd.core.dtypes.common.is_datetime64_any_dtype(self[constants.DATETIME].dtype):
                    self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME])

    def get_flow(self, origin_id, destination_id):
        """
        Get the flow between two locations. If there is no flow between two locations it returns 0.
        :param origin_id: the id of the origin tile
        :param destination_id: the id of the tessellation tile
        :return: The flow between the two locations
        :rtype: int
        """

        if (origin_id not in self._tessellation[constants.TILE_ID].values) or \
                (destination_id not in self._tessellation[constants.TILE_ID].values):
            raise ValueError("Both origin_id and destination_id must be present in the tessellation.")

        tmp = self[(self[constants.ORIGIN] == origin_id) & (self[constants.DESTINATION] == destination_id)]
        if len(tmp) == 0:
            return 0
        else:
            return tmp[constants.FLOW].item()

    def get_geometry(self, tile_id):
        if tile_id not in self._tessellation[constants.TILE_ID].values:
            raise ValueError("tile_id \"%s\" is not in the tessellation." % tile_id)

        return self.tessellation[self.tessellation[constants.TILE_ID] == tile_id].geometry.item()

    def to_matrix(self):

        m = np.zeros((len(self._tessellation), len(self._tessellation)))

        def _to_matrix(df, matrix, tessellation):
            o = tessellation.index[tessellation['tile_ID'] == df['origin']].item()
            d = tessellation.index[tessellation['tile_ID'] == df['destination']].item()

            matrix[o][d] = df['flow']

        self.apply(_to_matrix, args=(m, self._tessellation), axis=1)

        return m

    @classmethod
    def from_file(cls, filename, origin=None, destination=None, origin_lat=None, origin_lng=None, destination_lat=None,
                  destination_lng=None, flow=constants.FLOW, datetime=constants.DATETIME, timestamp=False, sep="\t",
                  tessellation=None, tile_id=constants.TILE_ID, usecols=None, header='infer', parameters=None,
                  remove_na=False):

        # Case 1: origin, destination, flow, [datetime]
        if (origin is not None) and (destination is not None):

            if not isinstance(tessellation, gpd.GeoDataFrame):
                raise AttributeError("tessellation must be a GeoDataFrame.")

        df = pd.read_csv(filename, sep=sep, header=header, usecols=usecols)

        # Case 2: origin_lat, origin_lng, destination_lat, destination_lng, flow, [datetime]
        if (origin_lat is not None) and (origin_lng is not None) and (destination_lat is not None) and \
                (destination_lng is not None):

            # Step 1: if tessellation is None infer it from data
            if tessellation is None:

                a = df[[origin_lat, origin_lng]].rename(columns={origin_lat: 'lat', origin_lng: 'lng'})

                b = df[[destination_lat, destination_lng]].rename(columns={destination_lat: 'lat',
                                                                           destination_lng: 'lng'})

                # DropDuplicates has to be applied now because Geopandas doesn't support removing duplicates in geometry
                points = pd.concat([a, b]).drop_duplicates(['lat', 'lng'])
                points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points['lng'], points['lat']),
                                          crs=constants.DEFAULT_CRS)

                tessellation = tiler.get('voronoi', points=points)

            # Step 2: map origin and destination points into the tessellation

            gdf_origin = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[origin_lng], df[origin_lat]),
                                          crs=tessellation.crs)
            gdf_destination = gpd.GeoDataFrame(df.copy(),
                                               geometry=gpd.points_from_xy(df[destination_lng], df[destination_lat]),
                                               crs=tessellation.crs)

            if all(isinstance(x, Polygon) for x in tessellation.geometry):

                if remove_na:
                    how = 'inner'
                else:
                    how = 'left'

                origin_join = gpd.sjoin(gdf_origin, tessellation, how=how, op='within').drop("geometry", axis=1)
                destination_join = gpd.sjoin(gdf_destination, tessellation, how=how, op='within').drop("geometry",
                                                                                                       axis=1)

                df = df.merge(origin_join[[constants.TILE_ID]], left_index=True, right_index=True)
                df.loc[:, constants.ORIGIN] = origin_join[constants.TILE_ID]
                df.drop([constants.ORIGIN_LAT, constants.ORIGIN_LNG, constants.TILE_ID], axis=1, inplace=True)

                df = df.merge(destination_join[[constants.TILE_ID]], left_index=True, right_index=True)
                df.loc[:, constants.DESTINATION] = destination_join[constants.TILE_ID]
                df.drop([constants.DESTINATION_LAT, constants.DESTINATION_LNG, constants.TILE_ID], axis=1, inplace=True)

            elif all(isinstance(x, Point) for x in tessellation.geometry):

                df.loc[:, constants.ORIGIN] = utils.nearest(gdf_origin, tessellation, constants.TILE_ID).values
                df.loc[:, constants.DESTINATION] = utils.nearest(gdf_destination, tessellation,
                                                                 constants.TILE_ID).values

                df.drop([origin_lat, origin_lng, destination_lat, destination_lng], inplace=True, axis=1)

            else:
                raise AttributeError("In case of expanded format (coordinates instead of ids), the tessellation must "
                                     "contains either all Polygon or all Point. Mixed types are not allowed.")

        # Step 3: call the constructor

        if parameters is None:
            parameters = {'from_file': filename}

        return cls(df, origin=constants.ORIGIN, destination=constants.DESTINATION, flow=flow, datetime=datetime,
                   timestamp=timestamp, tessellation=tessellation, parameters=parameters, tile_id=tile_id)

    @property
    def origin(self):
        if constants.ORIGIN not in self:
            raise AttributeError("The FlowDataFrame does not contain the column '%s.'" % constants.ORIGIN)

        return self[constants.ORIGIN]

    @property
    def destination(self):
        if constants.DESTINATION not in self:
            raise AttributeError("The FlowDataFrame does not contain the column '%s.'" % constants.DESTINATION)

        return self[constants.DESTINATION]

    @property
    def flow(self):
        if constants.FLOW not in self:
            raise AttributeError("The FlowDataFrame does not contain the column '%s.'" % constants.FLOW)

        return self[constants.FLOW]

    @property
    def datetime(self):
        if constants.DATETIME not in self:
            raise AttributeError("The FlowDataFrame does not contain the column '%s.'" % constants.DATETIME)

        return self[constants.DATETIME]

    @property
    def tessellation(self):
        return self._tessellation

    @property
    def _constructor(self):
        return FlowDataFrame

    @property
    def _constructor_sliced(self):
        return FlowSeries

    @property
    def _constructor_expanddim(self):
        return FlowDataFrame

    # Plot methods
    def plot_flows(self, map_f=None, min_flow=0, tiles='Stamen Toner', zoom=6, flow_color='red', flow_weight=5,
                   flow_exp=0.5, num_od_popup=5, style_function=plot.flow_style_function, flow_popup=False,
                   tile_popup=True, radius_origin_point=5, color_origin_point='#3186cc'):

        return plot.plot_flows(self, map_f=map_f, min_flow=min_flow,  tiles=tiles, zoom=zoom, flow_color=flow_color,
                               flow_weight=flow_weight, flow_exp=flow_exp, num_od_popup=num_od_popup,
                               style_function=style_function, flow_popup=flow_popup, tile_popup=tile_popup,
                               radius_origin_point=radius_origin_point, color_origin_point=color_origin_point)

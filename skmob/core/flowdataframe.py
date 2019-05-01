import pandas as pd
import geopandas as gpd
from ..utils import constants, utils
import numpy as np
from warnings import warn
from ..tessellation.tilers import tiler
import shapely


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
                 dates=constants.DATETIME, timestamp=False, tessellation=None, tile_id=constants.TILE_ID,
                 parameters={}):

        original2default = {origin: constants.ORIGIN,
                            destination: constants.DESTINATION,
                            flow: constants.FLOW,
                            dates: constants.DATETIME}

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

            if tessellation is None:
                raise TypeError("tessellation must be a GeoDataFrame with tile_id and geometry.")

            elif isinstance(tessellation, gpd.GeoDataFrame):
                self._tessellation = tessellation.copy()
                self._tessellation.rename(columns={tile_id: constants.TILE_ID}, inplace=True)

                if tessellation.crs is None:
                    warn("The tessellation crs is None. It will be set to the default crs WGS84 (EPSG:4326).")

                # Check consistency
                origin = self[constants.ORIGIN]
                destination = self[constants.DESTINATION]

                if not all(origin.isin(self._tessellation[constants.TILE_ID])) or \
                        not all(destination.isin(self._tessellation[constants.TILE_ID])):
                    raise ValueError("Inconsistency - origin and destination IDs must be present in the tessellation.")

            else:
                raise TypeError("tessellation must be a GeoDataFrame with tile_id and geometry.")

            if dates in columns:

                if timestamp:
                    self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME], unit='s')

                if not pd.core.dtypes.common.is_datetime64_any_dtype(self[constants.DATETIME].dtype):
                    self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME])

    @classmethod
    def from_file(cls, filename, origin=None, destination=None, origin_lat=None, origin_lng=None, destination_lat=None,
                  destination_lng=None, flow=constants.FLOW, dates=constants.DATETIME, timestamp=False, sep="\t",
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

            origin = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[origin_lng], df[origin_lat]),
                                      crs=tessellation.crs)
            destination = gpd.GeoDataFrame(df.copy(),
                                           geometry=gpd.points_from_xy(df[destination_lng], df[destination_lat]),
                                           crs=tessellation.crs)

            if all(isinstance(x, shapely.geometry.Polygon) for x in tessellation.geometry):

                if remove_na:
                    how = 'inner'
                else:
                    how = 'left'

                origin_join = gpd.sjoin(origin, tessellation, how=how, op='within').drop("geometry", axis=1)
                destination_join = gpd.sjoin(destination, tessellation, how=how, op='within').drop("geometry", axis=1)

                df = df.merge(origin_join[[constants.TILE_ID]], left_index=True, right_index=True)
                df.loc[:, constants.ORIGIN] = origin_join[constants.TILE_ID]
                df.drop([constants.ORIGIN_LAT, constants.ORIGIN_LNG, constants.TILE_ID], axis=1, inplace=True)

                df = df.merge(destination_join[[constants.TILE_ID]], left_index=True, right_index=True)
                df.loc[:, constants.DESTINATION] = destination_join[constants.TILE_ID]
                df.drop([constants.DESTINATION_LAT, constants.DESTINATION_LNG, constants.TILE_ID], axis=1, inplace=True)

            elif all(isinstance(x, shapely.geometry.Point) for x in tessellation.geometry):

                df.loc[:, constants.ORIGIN] = utils.ckdnearest(origin, tessellation, 'tile_ID')
                df.loc[:, constants.DESTINATION] = utils.ckdnearest(origin, tessellation, 'tile_ID')

                df.drop([origin_lat, origin_lng, destination_lat, destination_lng], inplace=True, axis=1)

        # Step 3: call the constructor

        if parameters is None:
            parameters = {'from_file': filename}

        return cls(df, origin=constants.ORIGIN, destination=constants.DESTINATION, flow=flow, dates=dates,
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

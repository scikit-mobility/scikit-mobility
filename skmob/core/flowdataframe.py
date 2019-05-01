import pandas as pd
import geopandas as gpd
from ..utils import constants, utils
import numpy as np
from warnings import warn
from ..tessellation.tilers import tiler
import shapely


# ------------------------------ FLOW DATA STRUCTURES ------------------------------


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
                 dates=constants.DATETIME, timestamp=False, tessellation=None, parameters={}):

        original2default = {origin: constants.LONGITUDE,
                            destination: constants.DATETIME,
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

        # CRS
        if (tessellation is None) and (tessellation.crs is None):
            warn("The tessellation crs is None. It will be set to the default crs WGS84 (EPSG:4326).")

        # Tessellation
        if isinstance(tessellation, gpd.GeoDataFrame):
            self._tessellation = tessellation
        else:
            raise AttributeError("tessellation must be a GeoDataFrame with tile_id and geometry.")

        # TODO: Check tessellation consistency: all the IDs in the flowdataframe must appear in the tessellation

        if not isinstance(data, pd.core.internals.BlockManager):

            if dates in columns:

                if timestamp:
                    self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME], unit='s')

                if not pd.core.dtypes.common.is_datetime64_any_dtype(self[constants.DATETIME].dtype):
                    self[constants.DATETIME] = pd.to_datetime(self[constants.DATETIME])

    @classmethod
    def from_file(cls, filename, origin=None, destination=None, origin_lat=None, origin_lng=None, destination_lat=None,
                  destination_lng=None, flow=constants.FLOW, dates=constants.DATETIME, timestamp=False, sep="\t",
                  crs=constants.DEFAULT_CRS, tessellation=None, usecols=None, header='infer', parameters=None):

        # Case 1: origin, destination, flow, [datetime]
        if (origin is not None) and (destination is not None) and (not isinstance(tessellation, gpd.GeoDataFrame)):
            raise AttributeError("tessellation must be a GeoDataFrame.")

        df = pd.read_csv(filename, sep=sep, header=header, usecols=usecols)

        # Case 2: origin_lat, origin_lng, destination_lat, destination_lng, flow, [datetime]
        if (origin_lat is not None) and (origin_lng is not None) and (destination_lat is not None) and (destination_lng is not None):

            # Step 1: if tessellation is None infer it from data
            if tessellation is None:
                a = df[df[constants.ORIGIN_LAT], df[constants.ORIGIN_LNG]].rename(columns={constants.ORIGIN_LAT: 'lat',
                                                                                       constants.ORIGIN_LNG: 'lng'})

                b = df[df[constants.DESTINATION_LAT], df[constants.DESTINATION_LNG]].rename(
                    columns={constants.DESTINATION_LAT: 'lat', constants.DESTINATION_LNG: 'lng'})

                # DropDuplicates has to be applied now because Geopandas doesn't support removing duplicates in geometry
                points = pd.concat([a, b]).drop_duplicates(['lat', 'lng'])
                points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(points['lng'], points['lat']))

                tessellation = tiler.get('voronoi', points)

            # Step 2: map origin and destination points into the tessellation

            origin = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[constants.ORIGIN_LNG],
                                                                      df[constants.DESTINATION_LAT]))

            destination = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[constants.DESTINATION_LNG],
                                                                           df[constants.DESTINATION_LAT]))

            if all(isinstance(x, shapely.geometry.Polygon) for x in points.geometry):

                origin_join = gpd.sjoin(tessellation, origin, how='left', op='within')

                destination_join = gpd.sjoin(tessellation, destination, how='left', op='within')

                df.loc[:, constants.ORIGIN] = origin_join[constants.TILE_ID]
                df.loc[:, constants.DESTINATION] = destination_join[constants.TILE_ID]
                df.drop([constants.ORIGIN_LAT, constants.ORIGIN_LNG, constants.DESTINATION_LAT,
                         constants.DESTINATION_LNG])

            elif all(isinstance(x, shapely.geometry.Points) for x in points.geometry):

                unary_union = tessellation.unary_union

                origin.loc[:, constants.ORIGIN] = origin.apply(utils.nearest, geom_union=unary_union,
                                                               tessellation=tessellation, geom1_col='centroid',
                                                               src_column=constants.TILE_ID, axis=1)

                destination.loc[:, constants.DESTINATION] = destination.apply(utils.nearest, geom_union=unary_union,
                                                                              tessellation=tessellation,
                                                                              geom1_col='centroid',
                                                                              src_column=constants.TILE_ID, axis=1)

                df.loc[:, constants.ORIGIN] = origin[constants.TILE_ID]
                df.loc[:, constants.DESTINATION] = destination[constants.TILE_ID]
                df.drop([constants.ORIGIN_LAT, constants.ORIGIN_LNG, constants.DESTINATION_LAT,
                         constants.DESTINATION_LNG])

        # Step 3: call the constructor

        if parameters is None:
            # Init prop dictionary
            parameters = {'from_file': filename}

        return cls(df, origin=origin, destination=destination, flow=flow, dates=dates, timestamp=timestamp,
                   tessellation=tessellation, parameters=parameters)

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
    def _constructor(self):
        return FlowDataFrame

    @property
    def _constructor_sliced(self):
        return FlowSeries

    @property
    def _constructor_expanddim(self):
        return FlowDataFrame

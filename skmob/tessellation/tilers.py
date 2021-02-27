from abc import ABC, abstractmethod
import math
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import cascaded_union
from ..utils import constants, utils
import numpy as np
import h3.api.numpy_int as h3
import warnings


class TessellationTilers:

    def __init__(self):
        self._tilers = {}

    def register_tiler(self, key, tiler):
        self._tilers[key] = tiler

    def create(self, key, **kwargs):
        tiler = self._tilers.get(key)

        if not tiler:
            raise ValueError(key)

        return tiler(**kwargs)

    def get(self, service_id, **kwargs):
        return self.create(service_id, **kwargs)


tiler = TessellationTilers()


class TessellationTiler(ABC):

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def _build(self, **kwargs):
        pass


class VoronoiTessellationTiler(TessellationTiler):

    def __init__(self):

        super().__init__()
        self._instance = None

    def __call__(self, points, crs=constants.DEFAULT_CRS):

        if not self._instance:

            if isinstance(points, gpd.GeoDataFrame):

                if not all(isinstance(x, Point) for x in points.geometry):
                    raise ValueError(
                        "Not valid points object. Accepted type is GeoDataFrame.")

        return self._build(points, crs)

    def _build(self, points, crs=constants.DEFAULT_CRS):

        gdf = gpd.GeoDataFrame(points.copy(), crs=crs)
        gdf.loc[:, constants.TILE_ID] = list(np.arange(0, len(gdf)))

        # Convert TILE_ID to have str type
        gdf[constants.TILE_ID] = gdf[constants.TILE_ID].astype('str')

        return gdf[[constants.TILE_ID, 'geometry']]


# Register the builder
tiler.register_tiler('voronoi', VoronoiTessellationTiler())


class SquaredTessellationTiler(TessellationTiler):

    def __init__(self):

        super().__init__()
        self._instance = None

    def __call__(self, base_shape, meters=50, which_osm_result=-1, crs=constants.DEFAULT_CRS, window_size=None):
        if not self._instance:

            if isinstance(base_shape, str):
                # Try to obtain the base shape from OSM
                base_shapes = utils.bbox_from_name(base_shape, which_osm_result=which_osm_result)
                i = 0
                base_shape = base_shapes.loc[[i]]
                while not (isinstance(base_shape.geometry.iloc[0], Polygon) or
                           isinstance(base_shape.geometry.iloc[0], MultiPolygon)):
                    i += 1
                    base_shape = base_shapes.loc[[i]]

            elif isinstance(base_shape, gpd.GeoDataFrame) or isinstance(base_shape, gpd.GeoSeries):

                if all(isinstance(x, Point) for x in base_shape.geometry):
                    # Build a base shape that contains all the points in the given geodataframe
                    base_shape = utils.bbox_from_points(base_shape)

                elif all(isinstance(x, Polygon) for x in base_shape.geometry) and len(base_shape) >= 1:

                    # Merge all the polygons
                    polygons = base_shape.geometry.values
                    base_shape = gpd.GeoSeries(
                        cascaded_union(polygons), crs=base_shape.crs)

                # elif not all(isinstance(x, Polygon) for x in base_shape.geometry):
                #    raise ValueError("Not valid geometry object. Accepted types are Point and Polygon.")
            else:
                raise ValueError(
                    "Not valid base_shape object. Accepted types are str, GeoDataFrame or GeoSeries.")

        return self._build(base_shape, meters, crs)

    def _build(self, base_shape, meters, crs=constants.DEFAULT_CRS):

        # We work with the universal crs epsg:3857
        tmp_crs = constants.UNIVERSAL_CRS

        area = base_shape.to_crs(tmp_crs)

        # Obtain the boundaries of the geometry
        boundaries = dict({'min_x': area.total_bounds[0],
                           'min_y': area.total_bounds[1],
                           'max_x': area.total_bounds[2],
                           'max_y': area.total_bounds[3]})

        # Find number of square for each side
        x_squares = int(
            math.ceil(math.fabs(boundaries['max_x'] - boundaries['min_x']) / meters))
        y_squares = int(
            math.ceil(math.fabs(boundaries['min_y'] - boundaries['max_y']) / meters))

        # Placeholder for the polygon
        polygons = []

        shape = area.unary_union

        # Iterate on the x
        for i in range(0, x_squares):

            # Increment x
            x1 = boundaries['min_x'] + (meters * i)
            x2 = boundaries['min_x'] + (meters * (i + 1))

            # Iterate on y
            for j in range(0, y_squares):

                # Increment y
                y1 = boundaries['min_y'] + (meters * j)
                y2 = boundaries['min_y'] + (meters * (j + 1))
                polygon_desc = {}

                # Create shape (polygon)
                p = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

                # s = boros_shape.intersection(p)
                s = shape.intersects(p)

                # if(s.area>0):
                if s:
                    # shape.intersection(p) ATTENTION! If you use the intersection than the crawler fails!
                    polygon_desc['geometry'] = p
                    polygons.append(polygon_desc)

        gdf = gpd.GeoDataFrame(polygons, crs=tmp_crs)
        gdf = gdf.reset_index().rename(columns={"index": constants.TILE_ID})

        # Convert TILE_ID to have str type
        gdf[constants.TILE_ID] = gdf[constants.TILE_ID].astype('str')

        return gdf.to_crs(crs)


# Register the builder
tiler.register_tiler('squared', SquaredTessellationTiler())


class H3TessellationTiler(TessellationTiler):

    def __init__(self):

        super().__init__()
        self._instance = None

    def _meters_to_res(self, meters):
        hex_side_len_km = meters / 1000
        array = np.asarray(list(constants.H3_UTILS['avg_hex_edge_len_km'].values()))
        res = (np.abs(array - hex_side_len_km)).argmin()
        return res

    def _get_appropriate_res(self, base_shape, meters):

        # translate meters to h3 resolution
        res = self._meters_to_res(meters)

        # find the minimum resolution level which will cover the base_shape 
        min_res_cover = np.where(
            np.array(list(constants.H3_UTILS['avg_hex_area_km2'].values())) > base_shape.area.values[0])[0][-1]

        # are the hexagons enough to fill the base_shape?
        # if not suggest the largest of the smallest resolutions/meters which fit in base_shape
        if res >= min_res_cover:
            pass
        else:
            warnings.warn(f' The cell side-length you provided is too large to cover the input area.'
                          f' Try something smaller, e.g. :'
                          f' Side-Length {constants.H3_UTILS["avg_hex_edge_len_km"][str(min_res_cover - 1)] / 1000} Km')
            res = min_res_cover - 1
        return res

    def _handle_polyfill(self, base_shape, res):

        def get_hex(x):
            h = h3.polyfill(x.__geo_interface__, res, geo_json_conformant=True)
            if len(h) > 0:
                return h
            else:
                return None

        if base_shape.type[0] == "MultiPolygon":
            tmp_hexs = base_shape.explode().apply(lambda x: get_hex(x))
            hexs = list(set(np.concatenate(tmp_hexs[tmp_hexs.notna()].to_list())))
        else:
            hexs = h3.polyfill(
                base_shape.geometry.__geo_interface__['features'][0]['geometry'], res, geo_json_conformant=True)

        return hexs

    def __call__(self, base_shape, meters=50, which_osm_result=-1, crs=constants.DEFAULT_CRS, window_size=None):
        if not self._instance:

            if isinstance(base_shape, str):
                # Try to obtain the base shape from OSM
                base_shapes = utils.bbox_from_name(base_shape, which_osm_result=which_osm_result)
                i = 0
                base_shape = base_shapes.loc[[i]]
                while not (isinstance(base_shape.geometry.iloc[0], Polygon) or
                           isinstance(base_shape.geometry.iloc[0], MultiPolygon)):
                    i += 1
                    base_shape = base_shapes.loc[[i]]

            elif isinstance(base_shape, gpd.GeoDataFrame) or isinstance(base_shape, gpd.GeoSeries):

                if all(isinstance(x, Point) for x in base_shape.geometry):
                    # Build a base shape that contains all the points in the
                    # given geodataframe
                    base_shape = utils.bbox_from_points(base_shape)
            else:
                raise ValueError(
                    "Not valid base_shape object. Accepted types are str, GeoDataFrame or GeoSeries.")

        # Merge all the polygons
        polygons = base_shape.geometry.values
        base_shape = gpd.GeoSeries(
            cascaded_union(polygons), crs=base_shape.crs)

        return self._build(base_shape, meters, crs)

    def _build(self, base_shape, meters, crs=constants.DEFAULT_CRS):

        #  translate input meters to appropriate h3 resolution 
        res = self._get_appropriate_res(base_shape, meters)

        # cover the base_shape with h3 hexagonal polygons
        hexs = self._handle_polyfill(base_shape, res)

        # get actual geoms out of H3 hexagons
        # from https://geographicdata.science/book/data/h3_grid/build_sd_h3_grid.html
        polygonise = lambda hex_id: Polygon(
            h3.h3_to_geo_boundary(
                hex_id, geo_json=True)
        )
        # prepare a geodf with all the H3 geoms
        all_polys = gpd.GeoDataFrame(
            {'geometry': list(map(polygonise, hexs)),
             'H3_INDEX': hexs},
            crs=constants.DEFAULT_CRS
        )

        # add TileID
        all_polys['TILE_ID'] = all_polys.index
        # Convert TILE_ID to have str type
        all_polys['TILE_ID'] = all_polys['TILE_ID'].astype('str')

        return all_polys


# Register the builder
tiler.register_tiler('h3_tessellation', H3TessellationTiler())

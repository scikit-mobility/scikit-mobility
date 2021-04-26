from abc import ABC, abstractmethod
import math
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import cascaded_union
from skmob.utils import constants, utils
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

    def __call__(self, base_shape, meters=50, which_osm_result=-1, crs=constants.DEFAULT_CRS, window_size=None):
        base_shape = self._create_geometry_if_not_exists(base_shape, which_osm_result)

        base_shape = self._merge_all_polygons_in(base_shape)

        return self._build(base_shape, meters, crs)

    def _create_geometry_if_not_exists(self, base_shape, which_osm_result):
        if not self._instance:

            if isinstance(base_shape, str):
                base_shape = self._str_to_geometry(base_shape, which_osm_result)

            elif ((isinstance(base_shape, gpd.GeoDataFrame) or isinstance(base_shape, gpd.GeoSeries)) and
                    all(isinstance(x, Point) for x in base_shape.geometry)):
                base_shape = utils.bbox_from_points(base_shape)

            else:
                raise ValueError(
                    "Not valid base_shape object. Accepted types are str, GeoDataFrame or GeoSeries.")
        return base_shape

    def _str_to_geometry(self, base_shape, which_osm_result):
        base_shapes = utils.bbox_from_name(base_shape, which_osm_result=which_osm_result)
        polygon_shape = self._find_first_polygon_in(base_shapes)
        return polygon_shape

    def _find_first_polygon_in(self, base_shapes):
        return_shape = base_shapes.iloc[[0]]

        for i, current_shape in enumerate(base_shapes["geometry"].values):
            if self._isinstance_poly_multipoly(current_shape):
                return_shape = base_shapes.iloc[[i]]
                break
        else:
            print(1)

        return return_shape



    def _isinstance_poly_multipoly(self, shape):
        return True if (
                isinstance(shape, Polygon) or isinstance(shape, MultiPolygon)
        ) else False


    def _merge_all_polygons_in(self, base_shape):
        polygons = base_shape.geometry.values
        base_shape = gpd.GeoSeries(cascaded_union(polygons), crs=base_shape.crs)
        return base_shape

    def _build(self, base_shape, meters, crs=constants.DEFAULT_CRS):

        res = self._meters_to_h3_resolution(base_shape, meters)

        # H3 requires epsg=4326
        base_shape = base_shape.to_crs(constants.DEFAULT_CRS)

        # cover the base_shape with h3 hexagonal polygons
        hexs = self._handle_polyfill(base_shape, res)


        # from https://geographicdata.science/book/data/h3_grid/build_sd_h3_grid.html
        # prepare a geodf with the H3 geoms from H3 id
        all_polys = gpd.GeoDataFrame(
            {'geometry': [self._get_h3_geom_from(hex_id) for hex_id in hexs],
             'H3_INDEX': hexs},
            crs=constants.DEFAULT_CRS
        )

        # add TileID
        all_polys[constants.TILE_ID] = all_polys.index
        # Convert TILE_ID to have str type
        all_polys[constants.TILE_ID] = all_polys[constants.TILE_ID].astype('str')

        return all_polys

    def _meters_to_res(self, meters):
        hex_side_len_km = meters / 1000
        array = np.asarray(list(constants.H3_UTILS['avg_hex_edge_len_km'].values()))
        res = (np.abs(array - hex_side_len_km)).argmin()
        return res

    def _meters_to_h3_resolution(self, base_shape, meters):

        base_shape_proj = base_shape.to_crs(constants.UNIVERSAL_CRS)

        res = self._meters_to_res(meters)

        min_res_cover = self._find_min_resolution(base_shape_proj)

        # are the hexagons enough to fill the base_shape?
        # if not suggest the largest of the smallest resolutions/meters which fit in base_shape
        if res <= min_res_cover:
            warnings.warn(f' The cell side-length you provided is too large to cover the input area.'
                          f' Try something smaller, e.g. :'
                          f' Side-Length {constants.H3_UTILS["avg_hex_edge_len_km"][str(min_res_cover - 1)] / 1000} Km')
            res = min_res_cover - 1
        return res

    def _find_min_resolution(self, base_shape):
        min_res_cover = np.where(
            np.array(list(constants.H3_UTILS['avg_hex_area_km2'].values())) > (base_shape.area.values[0] * 1000000)
        )[0][-1]
        return min_res_cover

    def _handle_polyfill(self, base_shape, res):
        if base_shape.type[0] == "MultiPolygon":
            tmp_hexs = base_shape.explode().apply(lambda x: self._get_hex(x, res))
            hexs = list(set(np.concatenate(tmp_hexs[tmp_hexs.notna()].to_list())))
        else:
            hexs = h3.polyfill(
                base_shape.geometry.__geo_interface__['features'][0]['geometry'], res, geo_json_conformant=True)
        return hexs

    def _get_hex(self, x, res):
        h = h3.polyfill(x.__geo_interface__, res, geo_json_conformant=True)
        if h:
            return h

    def _get_h3_geom_from(self, hex_id):
        return Polygon(
            h3.h3_to_geo_boundary(
                hex_id), geo_json=True)



# Register the builder
tiler.register_tiler('h3_tessellation', H3TessellationTiler())

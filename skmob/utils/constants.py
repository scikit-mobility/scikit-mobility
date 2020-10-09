"""
Useful constants
"""

UNIVERSAL_CRS = {"init": "epsg:3857"}
DEFAULT_CRS = {"init": "epsg:4326"}

UID = 'uid'
TID = 'tid'
LATITUDE = 'lat'
LONGITUDE = 'lng'
ORIGIN = 'origin'
DESTINATION = 'destination'
ORIGIN_LAT = 'origin_lat'
ORIGIN_LNG = 'origin_lng'
DESTINATION_LAT = 'destination_lat'
DESTINATION_LNG = 'destination_lng'
DATETIME = 'datetime'
FLOW = 'flow'
TILE_ID = 'tile_ID'
CLUSTER = 'cluster'
LEAVING_DATETIME = 'leaving_datetime'
FREQUENCY = "freq"
PROBABILITY = "prob"
TOTAL_FREQ = "T_freq"
COUNT = "count"
TEMP = "tmp"
PROPORTION = "prop"
TOT_OUTFLOW = "tot_outflow"
PRECISION_LEVELS = ["Year", "Month", "Day", "Hour", "Minute", "Second", "year", "month", "day", "hour", "minute",
                    "second"]

PRIVACY_RISK = "risk"
INSTANCE = "instance"
INSTANCE_ELEMENT = "instance_elem"
REIDENTIFICATION_PROBABILITY = "reid_prob"
RELEVANCE = 'relevance'

# PROPERTIES KEY
CRS = 'crs'
FILTERING_PARAMS = 'filter'
COMPRESSION_PARAMS = 'compress'
CLUSTERING_PARAMS = 'cluster'
DETECTION_PARAMS = 'detect'

# Data files
GEOLIFE_SAMPLE = 'https://raw.githubusercontent.com/scikit-mobility/scikit-mobility/master/examples/geolife_sample.txt.gz'
NY_COUNTIES_2011 = 'https://raw.githubusercontent.com/scikit-mobility/scikit-mobility/master/examples/NY_counties_2011.geojson'
NY_FLOWS_2011 = 'https://raw.githubusercontent.com/scikit-mobility/scikit-mobility/master/examples/NY_commuting_flows_2011.csv'
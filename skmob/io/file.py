import pandas as pd
import fiona
import json
import os
import fnmatch
from ..core.trajectorydataframe import TrajDataFrame
from ..core.flowdataframe import FlowDataFrame
from ..preprocessing import filtering, compression
from tqdm import tqdm

def write(skmob_df, file):
    """
    Write a TrajDataFrame to a json file.

    :param skmob_df: TrajDataFrame
        object that will be saved.

    :param file: str
        path and name of the `json` output file.

    :return: None

    """
    datetime_columns = [c for c in skmob_df.columns
                        if pd.core.dtypes.common.is_datetime64_any_dtype(skmob_df[c].dtype)]

    with open(file, 'w') as f:
        json.dump([str(type(skmob_df)),
                   skmob_df.to_json(orient='split', date_unit='s'),
                   skmob_df.parameters,
                   datetime_columns], f)


def read(file):
    """
    Read a TrajDataFrame from a json file.

    :param file: str
        path and name of the `json` file to read.

    :return:
        object loaded from file.

    """
    with open(file, 'r') as f:
        df_type, json_df, parameters, datetime_columns = json.load(f)

    if 'TrajDataFrame' in df_type:
        tdf = TrajDataFrame(pd.read_json(json_df, orient='split', date_unit='s',
                                         convert_dates=datetime_columns), parameters=parameters)
        return tdf
    elif 'FlowDataFrame' in df_type:
        return None
    else:
        print('DataFrame type not recognised.')
        return None


def load_gpx(file, user_id=None):
    """
    :param str file: str
        path to the gpx file

    :param user_id: str or int
        name or ID of the user

    :return: a TrajDataFrame containing the trajectory
    :rtype: TrajDataFrame

    """
    track = [[p['properties']['time']] + list(p['geometry']['coordinates'])
             for p in fiona.open(file, layer='track_points')]

    tdf = TrajDataFrame(track, datetime=0, longitude=1, latitude=2)

    if user_id is not None:
        tdf['uid'] = [user_id for _ in range(len(tdf))]

    return tdf.sort_by_uid_and_datetime()


def load_geolife_trajectories(path_to_geolife_data_dir, user_ids=[],
                              filter_kwargs={'max_speed_kmh': 400},
                              compress_kwargs={'spatial_radius_km': 0.2}):
    """
    Load the Geolife trajectories in a TrajDataFrame

    :param path_to_geolife_data_dir: str
        local path of the directory 'Geolife Trajectories 1.3/'

    :param user_ids: list
        list of user IDs to load. If empty all users are loaded.

    :param filter_kwargs: dict
        arguments of `preprocessing.filtering.filter()`. If empty, data is not filtered.

    :param compress_kwargs: dict
        arguments of `preprocessing.compression.compress()`. If empty, data is not compressed.

    :return: TrajDataFrame
        a TrajDataFrame containing all trajectories
    :rtype: TrajDataFrame

    """
    tdf = TrajDataFrame(pd.DataFrame())

    path = path_to_geolife_data_dir + 'Data/'
    if len(user_ids) == 0:
        user_ids = os.listdir(path)

    for uid in tqdm(user_ids):
        try:
            all_files = fnmatch.filter(os.listdir(path + '%s/Trajectory/' % uid), "*.plt")

        except NotADirectoryError:
            continue

        dfg = (pd.read_csv(path + '%s/Trajectory/' % uid + f,
                           skiprows=6, header=None, usecols=[0, 1, 5, 6]) for f in all_files)
        df = pd.concat(dfg, ignore_index=True)
        df['datetime'] = df[5] + ' ' + df[6]
        df.drop(columns=[5, 6], inplace=True)
        df['uid'] = [str(uid) for _ in range(len(df))]

        tdf0 = TrajDataFrame(df, latitude=0, longitude=1)

        if len(filter_kwargs) > 0:
            tdf0 = filtering.filter(tdf0, **filter_kwargs)

        if len(compress_kwargs) > 0:
            tdf0 = compression.compress(tdf0, **compress_kwargs)

        tdf = tdf.append(tdf0)
        tdf.parameters = tdf0.parameters

    return tdf


def load_google_timeline(file, user_id=None, min_accuracy_meters=None):
    """
    Load a Google Timeline trajectory in a TrajDataFrame.

    :param file: path to the file "Location History.json" inside the Google Takeout archive
    :type file: str

    :param user_id: ``str`` or ``int``
        name or ID of the user

    :param float min_accuracy_meters:
        remove points with "accuracy" value higher than ``min_accuracy_meters`` meters

    :return: TrajDataFrame
    :rtype: TrajDataFrame

    """
    # Read file
    with open(file, 'r') as f:
        goog = json.load(f)
    df = pd.DataFrame.from_dict(goog['locations'])

    # filter out inaccurate points
    if min_accuracy_meters is not None:
        df = df[df['accuracy'] < min_accuracy_meters]

    if user_id is not None:
        df['uid'] = [user_id for _ in range(len(df))]
    df['latitudeE7'] = df['latitudeE7'] / 1e7
    df['longitudeE7'] = df['longitudeE7'] / 1e7
    df['timestampMs'] = df['timestampMs'].astype(float) / 1e3

    tdf = TrajDataFrame(df, latitude='latitudeE7', longitude='longitudeE7',
                        datetime='timestampMs', timestamp=True)

    return tdf.sort_by_uid_and_datetime()

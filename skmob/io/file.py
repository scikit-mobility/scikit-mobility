import pandas as pd
import json
import os
import fnmatch
from ..core.trajectorydataframe import TrajDataFrame
from ..core.flowdataframe import FlowDataFrame
from ..preprocessing import filtering, compression


def write(skmob_df, file):
    datetime_columns = [c for c in skmob_df.columns
                        if pd.core.dtypes.common.is_datetime64_any_dtype(skmob_df[c].dtype)]

    with open(file, 'w') as f:
        json.dump([str(type(skmob_df)),
                   skmob_df.to_json(orient='split', date_unit='s'),
                   skmob_df.parameters,
                   datetime_columns], f)


def read(file):
    with open(file, 'r') as f:
        df_type, json_df, parameters, datetime_columns = json.load(f)

    if 'TrajDataFrame' in df_type:
        tdf = TrajDataFrame(pd.read_json(json_df, orient='split', date_unit='s', convert_dates=datetime_columns),
                            parameters=parameters)
        return tdf
    elif 'FlowDataFrame' in df_type:
        return None
    else:
        print('DataFrame type not recognised.')
        return None


def load_geolife_trajectories(path_to_geolife_data_dir, user_ids=[],
                              filter_kwargs={'max_speed_kmh': 400},
                              compress_kwargs={'spatial_radius_km': 0.2}):
    """
    Load the Geolife trajectories in a skmob.TrajDataFrame

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

    """
    tdf = pd.DataFrame()

    path = path_to_geolife_data_dir + 'Data/'
    if len(user_ids) == 0:
        user_ids = os.listdir(path)

    for uid in user_ids:

        try:
            all_files = fnmatch.filter(os.listdir(path + '%s/Trajectory/' % uid), "*.plt")

        except NotADirectoryError:
            continue

        dfg = (pd.read_csv(path + '%s/Trajectory/' % uid + f,
                           skiprows=6, header=None, usecols=[0, 1, 5, 6]) for f in all_files)
        df = pd.concat(dfg, ignore_index=True)
        df['datetime'] = df[5] + ' ' + df[6]
        df.drop(columns=[5, 6], inplace=True)
        df['uid'] = [str(uid) for i in range(len(df))]

        tdf0 = TrajDataFrame(df, latitude=0, longitude=1)

        if len(filter_kwargs) > 0:
            tdf0 = filtering.filter(tdf0, **filter_kwargs)

        if len(compress_kwargs) > 0:
            tdf0 = compression.compress(tdf0, **compress_kwargs)

        tdf = tdf.append(tdf0)

    return tdf

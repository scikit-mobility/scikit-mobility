import pandas as pd
import json
from ..core.trajectorydataframe import TrajDataFrame
from ..core.flowdataframe import FlowDataFrame


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

import csv
import numpy as np
import pandas as pd
import ipdb
import math
from datetime import datetime

DST_data = pd.read_csv(
            "2016_2023_SDT_median.csv",
            delimiter=",",
            parse_dates=[0],
            infer_datetime_format=True,
            na_values="0",
            header=None,
        )

Bz_data = pd.read_csv(
            "2016-2023_Bz.csv",
            delimiter=",",
            parse_dates=[0],
            infer_datetime_format=True,
            na_values="0",
            header=None,
        )
time_index = 1
time_index_org = time_index
Bz_column = ["Bz"]
for time in DST_data[0][1:]:
    input_datetime = datetime.strptime(time, "%Y/%m/%d %H:%M")
    time = input_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # ipdb.set_trace()
    while(time != Bz_data[0][time_index]):
        time_index += 1
        if(time_index >= 52451):
            break
    if(time_index < 52451):
        Bz_column.append(Bz_data[1][time_index])
        time_index_org = time_index
    else:
        Bz_column.append('')
        time_index = time_index_org
# ipdb.set_trace()
DST_data = DST_data.assign(Bz = Bz_column)
DST_data.to_csv('2016-2020_DSTBz_median.csv', index=False)
    
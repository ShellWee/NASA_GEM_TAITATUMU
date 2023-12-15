import csv
import numpy as np
import pandas as pd
import ipdb
import math

data = pd.read_csv(
            "2016_2023_SDT_new.csv",
            delimiter=",",
            parse_dates=[0],
            infer_datetime_format=True,
            na_values="0",
            header=None,
        )
ipdb.set_trace()
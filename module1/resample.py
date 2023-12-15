import csv
import numpy as np
import pandas as pd
import ipdb
import math

data = pd.read_csv('2016_2023median_no_clean.csv')
print(data)
ipdb.set_trace()
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
hourly_resampled_data = data.resample('H').median()
print(hourly_resampled_data)

hourly_resampled_data.to_csv('dsc_fc_summed_spectra_2016-2020_resampled.csv', index=True)
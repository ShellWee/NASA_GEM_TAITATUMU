import csv
import numpy as np
import pandas as pd
import ipdb

lines = None
with open('rtsw_plot_data_1998-01-01T00_00_00.txt', 'r') as txtfile:
    lines = txtfile.readlines()
    
with open('rtsw_plot_data_1998-01-01T00_00_00.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    for line in lines:
        fields = line.strip().split('    ')
        writer.writerow(fields)
# Numerical Operations
import random as rnd
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
# For Progress Bar
from tqdm import tqdm
import matplotlib.pyplot as plt

# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 42,      # Seed number, you can pick your lucky number. :)        
    'batch_size': 64,           
    'input_size': 4, 
    'sequence_length': 24,
    'hidden_size': 512,
    'num_layers': 2
}

class SolarWind_Kp_TestDataset(Dataset):
    def __init__(self,file_path,normalized_data_file_path,n_past,approach):
        self.file = pd.read_csv(file_path)
        self.normalized_data = pd.read_csv(normalized_data_file_path)
        self.file = self.file[['proton_speed','proton_density','proton_temperature','K_p']]
        self.file = self.fill_nan(self.file.values)
        max_val = self.normalized_data['max']
        min_val = self.normalized_data['min']
        self.approach = approach
        if self.approach == 'normalize':
            for i in tqdm(range(len(self.file)), desc='Processing'):
                for features in range(self.file.shape[1]):
                    self.file[i,features] = (self.file[i,features] - min_val[features]) / (max_val[features] - min_val[features])
            print(f'Done normalizing data!!!')  
        self.n_past = n_past
        self.X , self.Y = self.createXY(self.file,self.n_past)
        self.approach = approach
    def createXY(self,dataset,n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)):
                dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
                dataY.append(dataset[i,-1])
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        dataY = dataY.reshape(-1, 1)
        return torch.FloatTensor(dataX),torch.FloatTensor(dataY)
    def fill_nan(self,dataset):
        filled_dataset = np.zeros(dataset.shape)
        for i in range(dataset.shape[1]):
            for j in range(dataset.shape[0]):
                if np.isnan(dataset[j,i]):
                    filled_dataset[j,i] = np.nanmedian(dataset[:,i])
                else:
                    filled_dataset[j,i] = dataset[j,i]
        return filled_dataset   
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    def __len__(self):
        return len(self.X)
    
test_file_name = './Data/Predicted_SDT (7).csv'
normalized_data_file_path = './Data/normalized_min_max_data.csv'
preprocess_approach = 'normalize'
test_dataset = SolarWind_Kp_TestDataset(test_file_name,normalized_data_file_path,config['sequence_length'],preprocess_approach)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)  

class RNN_base(nn.Module):
    def __init__(self,seq_length, input_size, hidden_size, num_layers, dropout=0.5):
        super(RNN_base, self).__init__()
        self.input_size = input_size 
        self.seq_length = seq_length 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout , batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 1)
        
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, 1)
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, 1)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # (batch_size , seq_length , hidden_size * 2)
        out, _ = self.lstm(x)
        # (batch_size , hidden_size)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
def denormalize(signal):
        min_max = pd.read_csv('./Data/normalized_min_max_data.csv').values
        n = signal.shape[0]
        output_signal = np.zeros_like(signal, dtype=float)
        for sample_num in range(n):
            output_signal[sample_num] = signal[sample_num] * (min_max[3][0] - min_max[3][1]) + min_max[3][1]
        return output_signal

model_best =  RNN_base( seq_length = config['sequence_length'], input_size = config['input_size'], hidden_size = config['hidden_size'], num_layers = config['num_layers']).to(device)
model_best.load_state_dict(torch.load('./models/model_best.ckpt', map_location=device))
model_best.eval()
GT_ = []
prediction = []
with torch.no_grad():
    for input,GT in test_loader: 
        input = input.to(device)
        GT_  += GT.unsqueeze(0)
        prediction  += model_best(input).unsqueeze(0)
GT_ = torch.concat(GT_).cpu()
GT_ = GT_.reshape(-1,1)
prediction = torch.concat(prediction).cpu()
if preprocess_approach == "normalize":
    GT_ = denormalize(GT_)
    prediction = denormalize(prediction)

print(f"MSE of biLSTM: {mean_squared_error(GT_, prediction):.3E}")
rmse_biLSTM = np.sqrt(mean_squared_error(GT_, prediction))
print(f"RMSE of biLSTM: {rmse_biLSTM:.3E}")
nrmse_biLSTM= rmse_biLSTM / (GT_.max() - GT_.min())
print(f"NRMSE of biLSTM: {nrmse_biLSTM*100:.2f} %")
print(f'R2 score: {r2_score(GT_, prediction)}')

# For first sample in the test set
fig, axes = plt.subplots(1, 1, figsize=(6,7), sharex=True)
sample_num = 100
length = -1
start_idx = 0

line1 = axes.plot(range(len(GT_[5000:10000])), GT_[5000:10000], color="tab:blue", label="GT Kp")
line2 = axes.plot(range(len(prediction[5000:10000])), prediction[5000:10000], color="tab:red", linestyle='dashed', label="Predicted Kp")

fig.legend(handles =[line1[0], line2[0]], loc ='lower center', ncol=4)
fig.suptitle('Visualization of predicted Kp', fontsize=14, y=0.93)
fig.supylabel(r'$Kp$', x=0.02)
plt.savefig(f"./prediction_new2.png", bbox_inches='tight', dpi=500)

def scale_array_to_range(array, new_min=0, new_max=9):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = (array - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return scaled_array

scaled_array = scale_array_to_range(prediction, new_min=0, new_max=9)
prediction = scale_array_to_range(prediction, new_min=0, new_max=9)
prediction = prediction.tolist()
prediction = [item for sublist in prediction for item in sublist]
GT_ = scale_array_to_range(GT_, new_min=0, new_max=9)
GT_ = GT_.tolist()
GT_ = [item for sublist in GT_ for item in sublist]

def hours_to_days_hours(hours):
    days = hours // 24
    remaining_hours = hours % 24
    return days, remaining_hours

def detect_geomagnetic_storm(scaled_array):
    storm_detected = False
    storm_start = None
    storm_duration = 0

    for i, value in enumerate(scaled_array):
        if value[0] > 5:
            if not storm_detected:
                storm_detected = True
                storm_start = i
            storm_duration += 1
        else:
            if storm_detected:
                storm_detected = False
                detected_day, detected_remaining_hours = hours_to_days_hours(storm_start+1)
                storm_day, storm_remaining_hours = hours_to_days_hours(storm_duration)
                print(f"Geomagnetic storm detected after {detected_day:5d} days and {detected_remaining_hours+1:3d} hours and will continue {storm_day:3d} days {storm_remaining_hours:3d} hours\n")
                storm_duration = 0

detect_geomagnetic_storm(scaled_array)


df_test = pd.read_csv(test_file_name)
timestamps = df_test['Time'].values.tolist()

from datetime import datetime, timedelta

utc_times = [datetime.strptime(ts, "%Y/%m/%d %H:%M") for ts in timestamps][24:]
plt.figure(figsize=(10, 4))
plt.plot(utc_times, GT_, label="Real Data",alpha = 0.7)
plt.plot(utc_times, prediction, label="Predicted Data",alpha = 0.5)
plt.axhline(y=5, color='black', linestyle='--',alpha = 0.9)
# fontsize =12
plt.text(utc_times[0], 5.5, 'Kp = 5', fontsize = 15,alpha = 0.9)
plt.xlabel('Time')
plt.ylabel('Planetary K-index (Kp)')
plt.legend()
plt.title("Geomagnetic Storm Prediction")
plt.grid(True)
plt.show()
plt.savefig(f"./occurence.png", bbox_inches='tight', dpi=500)
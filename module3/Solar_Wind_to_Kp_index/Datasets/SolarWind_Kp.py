import os
import h5py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class SolarWind_Kp_Dataset(Dataset):
    def __init__(self,file_path,n_past,approach):
        self.file = pd.read_csv(file_path)
        self.file = self.file[['proton_speed','proton_density','proton_temperature','K_p']]
        self.file = self.fill_nan(self.file.values)
        self.approach = approach
        if self.approach == 'normalize':
            scaler = MinMaxScaler(feature_range=(0,1))
            self.file = scaler.fit_transform(self.file)
            self.max_val = scaler.data_max_
            self.min_val = scaler.data_min_
        self.n_past = n_past
        self.X , self.Y = self.createXY(self.file,self.n_past)
        self.approach = approach
    def fill_nan(self,dataset):
        filled_dataset = np.zeros(dataset.shape)
        for i in range(dataset.shape[1]):
            for j in range(dataset.shape[0]):
                if np.isnan(dataset[j,i]):
                    filled_dataset[j,i] = np.nanmedian(dataset[:,i])
                else:
                    filled_dataset[j,i] = dataset[j,i]
        return filled_dataset   

    def createXY(self,dataset,n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataset)-24):
                dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
                dataY.append(dataset[i:i+24,-1])
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        dataY = dataY.reshape(-1, 24)                
        return torch.FloatTensor(dataX),torch.FloatTensor(dataY)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    def __len__(self):
        return len(self.X)

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
        for i in range(n_past, len(dataset)-24):
                dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
                dataY.append(dataset[i:i+24,-1])
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        dataY = dataY.reshape(-1, 24)
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

if __name__ == "__main__":
    dataset = SolarWind_Kp_Dataset(file_path = './Data/train_std_noise_full',n_past = 24, approach="normalize")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
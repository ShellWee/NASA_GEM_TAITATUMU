# Numerical Operations
import random as rnd
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import datetime

# For Progress Bar
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Pytorch
import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
import wandb


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 42,      # Seed number, you can pick your lucky number. :)
    'valid_ratio': 0.1,   # validation_size = train_size * valid_ratio
    'n_epochs': 100,     # Number of epochs.            
    'batch_size': 32, 
    'learning_rate': 1e-2,              
    'early_stop': 50,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt',  # Your model will be saved here.
    # Hyper-parameters
    'n_past' : 24,
    'input_size': 4, 
    'sequence_length': 24,
    'hidden_size': 512,
    'num_layers': 2
}

descr = f"Train with {config['sequence_length']} hours before the pred; hidden_size = {config['sequence_length']}; Adam; lr= {config['learning_rate']}; batch_size= {config['batch_size']}; {config['n_epochs']} epochs"
run_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} {descr}'
run = wandb.init(
project = "NASA", # Set the project where this run will be logged
name = run_name,
config  = {
    "model_name" : 'biLSTM',
    "lr"         : config['learning_rate'],
    "batch_size" : config['batch_size'],
    "max_epoch" : config['n_epochs'],
    "n_past": config['n_past'],
    "hidden_size": config['hidden_size'],
    "valid_ratio": config['valid_ratio'],
    "Optimizer": 'Adam'
    })

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

same_seed(config['seed'])

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
        for i in range(n_past, len(dataset)):
                dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
                dataY.append(dataset[i,-1])
        dataX = np.array(dataX)
        dataY = np.array(dataY)
        dataY = dataY.reshape(-1, 1)
        return torch.FloatTensor(dataX),torch.FloatTensor(dataY)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    def __len__(self):
        return len(self.X)

def create_dataloader(source,n_past,preprocess_approach):
    num_workers = min(os.cpu_count(), 4)
    source = f'./Data/{source}_SolarWind_Kp_.csv'
    train_valid_dataset = SolarWind_Kp_Dataset(source,n_past,preprocess_approach)

    cols = ['max','min']
    normalized_data = np.vstack((train_valid_dataset.max_val,train_valid_dataset.min_val))
    df_normalized_data = pd.DataFrame(normalized_data.transpose(), columns=cols)
    df_normalized_data.to_csv('./Data/normalized_min_max_data.csv', index=False)

    valid_size = int(config['valid_ratio'] * len(train_valid_dataset))
    train_size = len(train_valid_dataset) - valid_size
    train_dataset, valid_dataset = random_split(train_valid_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(config['seed']))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    return train_loader, valid_loader

train_dataloader, valid_dataloader = create_dataloader('train', config['n_past'] , 'normalize')


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
    

def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean') 
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) 
    # Create the StepLR scheduler
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    train_loss = []
    valid_loss = []
    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []
        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            # (batch_size, seq_length, input_size)
            pred = model(x).view(-1, 1)   
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        train_loss.append(mean_train_loss)
        wandb.log({"Train loss": mean_train_loss})
        # Update the learning rate at the end of each epoch
        # scheduler.step()
        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            # (batch_size,seq_length,input_size)
            with torch.no_grad():
                pred = model(x).view(-1, 1)
                loss = criterion(pred, y)
            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        valid_loss.append(mean_valid_loss)
        wandb.log({"Valid loss": mean_valid_loss})
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.8f}, Valid loss: {mean_valid_loss:.8f}')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.8f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
    # A figure to compare train loss and valid loss over epochs.
    # With title and axis labels.
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train/Valid Loss')
    plt.legend()
    plt.savefig('./loss.png')
        
model =  RNN_base( seq_length = config['sequence_length'], input_size = config['input_size'], hidden_size = config['hidden_size'], num_layers = config['num_layers']).to(device)

trainer(train_dataloader, valid_dataloader, model, config, device)
wandb.finish()

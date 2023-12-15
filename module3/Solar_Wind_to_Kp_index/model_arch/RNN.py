import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import numpy as np
import pandas as pd

import wandb
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt


# Fully connected neural network with one hidden layer
# https://github.com/patrickloeber/pytorch-examples/blob/master/rnn-lstm-gru/main.py
# https://zhuanlan.zhihu.com/p/41261640
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

class RNN(LightningModule):
    # ,seq_length,lr,hidden_size
    def __init__(self,seq_length,lr,hidden_size):
        super().__init__()
        self.save_hyperparameters()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.lr = lr
        self.example_input_array = torch.Tensor(16, self.seq_length, 4)
        self.RNN_base = RNN_base(self.seq_length,4,hidden_size,2)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, X):
        X = self.RNN_base(X)
        return X

    def training_step(self, batch, batch_idx):
        X, Y = batch
        print(Y.shape)
        pred = self.RNN_base(X)
        pred = pred.reshape(-1,1)
        criterion = nn.MSELoss(reduction='mean') 
        mse_loss  = criterion(pred, Y)

        
        self.log("Train_loss", mse_loss)
        self.training_step_outputs.append({"loss": mse_loss})
        # self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": mse_loss}
    
    def validation_step(self, batch, batch_idx):
        X, Y = batch
        pred = self.RNN_base(X) 
        pred = pred.reshape(-1,1)
        criterion = nn.MSELoss(reduction='mean') 
        mse_loss  = criterion(pred, Y)
        
        self.log("val_loss", mse_loss)
        self.validation_step_outputs.append({"loss": mse_loss})
        # self.logger.experiment.add_scalar(f'Learning rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)
        return {"loss": mse_loss}  

    def on_train_epoch_end(self):
        loss = []
        for step_result in self.training_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())

        loss = np.concatenate([loss], axis=0)
        wandb.log({"Train loss epoch": loss.mean()})
        # self.logger.experiment.add_scalar(f'Train/Loss', loss.mean(), self.current_epoch)
    
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        loss = []

        for step_result in self.validation_step_outputs:
            loss.append(step_result["loss"].cpu().detach().numpy())
        loss = np.concatenate([loss], axis=0)
        # self.logger.experiment.add_scalar(f'Valid/Loss', loss.mean(), self.current_epoch)
        wandb.log({"Valid loss epoch": loss.mean()})
        self.validation_step_outputs = []

    def configure_optimizers(self):
        # opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9) 
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [opt],[]
    

    # define optimizer and learning rate scheduler for pytorch lightning module 
    # learning rate decays in a proportion of 0.1 if the validation loss does not improve for 40 epochs

    # def configure_optimizers(self):
    #     opt = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     # opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9) 
    #     return {
    #     "optimizer": opt,
    #     "lr_scheduler": {
    #         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=40, verbose=True),
    #         "monitor": "val_loss",
    #         "frequency": 1
    #     },
    # }

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(metrics=metric)

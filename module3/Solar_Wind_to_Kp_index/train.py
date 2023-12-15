import os
import yaml
import argparse
from argparse import Namespace
import numpy as np 
import pandas as pd
import datetime

from importlib import import_module
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from torch.utils.data import Dataset, DataLoader, random_split
from Datasets.SolarWind_Kp import SolarWind_Kp_Dataset

import wandb
# ! wandb login
# wandb.login()

config = {
    'seed': 42,      
    'valid_ratio': 0.1,      
    'batch_size': 32
}

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

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


def main(args):
    max_epochs = args.max_epoch
    n_past = 1
    hidden_size = 512
    lr         = 1e-2
    
    model = import_module(f'model_arch.{args.arch}').__dict__[args.trainer](n_past,lr,hidden_size)
    print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_dataloader, valid_dataloader = create_dataloader(args.source, n_past , args.preprocess_approach)
    
    #TensorBoard
    save_dir = f"./Logs/RNN/{args.source}"
    name = f"{args.description}/"
    descr = f"Train with {n_past} hours before the pred; hidden_size = {hidden_size}; Adam; lr= {lr}; batch_size= {config['batch_size']}; {args.preprocess_approach}; {args.max_epoch} epochs"
    run_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} {descr}'
    run = wandb.init(
    project = "NASA", # Set the project where this run will be logged
    name = run_name,
    config  = {
        "model_name" : args.arch,
        "lr"         : lr,
        "batch_size" : config['batch_size'],
        "preprocess approach" : args.preprocess_approach,
        "max_epoch" : args.max_epoch,
        "n_past": n_past,
        "hidden_size": hidden_size,
        "valid_ratio": config['valid_ratio'],
        "Optimizer": 'Adam'
        })
    logger = TensorBoardLogger(
        save_dir = save_dir, 
        version = args.version,
        name = name,
        default_hp_metric = True)
    # 1. Save top-1 val loss models
    checkpoint_best_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss", 
        mode="min",
        filename="{epoch:05d}-{val_loss:.8f}"
    )
    # 3. Log learning rate
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    with ClearCache():
        # training
        gpu = "gpu" if args.gpu else "cpu"
        
        trainer = Trainer(accelerator = gpu, # "gpu". "cpu", "tpu", "ipu", "auto"
                            devices = args.device,
                            logger = logger, 
                            log_every_n_steps = 1,
                            max_epochs = max_epochs + 1,
                            profiler = "simple", 
                            num_sanity_val_steps = 30,
                            callbacks = [checkpoint_best_callback,lr_monitor]
                        )
        # checkpoint = torch.load('./Logs/biLSTM/train/LAST/version_34/checkpoints/epoch=00198-val_loss=211.30934143.ckpt')
        # model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=valid_dataloader,
                    )
    wandb.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Residual prediction')
    # python3 train.py --arch RNN --trainer RNN --source train --preprocess_approach normalize --max_epoch 5 --description LAST
    parser.add_argument('--gpu', type=bool, default=True, help = 'Whether use GPU training')
    parser.add_argument('--device', type=int, default=1,  help = 'GPU id (If use the GPU)')
    parser.add_argument('--max_epoch', type=int, default=100, help = 'Maximun epochs')
    parser.add_argument('--arch', type=str,  default="RNN", help = 'The file where trainer located')
    parser.add_argument('--trainer', type=str,  default="RNN", help = 'The trainer we used')
    parser.add_argument('--source', type=str,  default="train", help="Train")
    parser.add_argument('--preprocess_approach', type=str,  default="normalize", help="normalize/scale")
    parser.add_argument('--description', type=str,  default="None")
    parser.add_argument('--version', type=int, help = 'version')

    args = parser.parse_args()
    main(args)
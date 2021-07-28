import os
import numpy as np
from os import listdir
import argparse
import logging
import os
import random
import numpy as np
from networks.unet_model import UNet
from networks.AttUnet_model import Att_UNet
from networks.trainer import Trainer
from torchsummary import summary
from dataloader.data_loader import DataLoader
from torch.utils import data
import torch
from networks.loss import Loss_L1, Loss_MSE, Loss_DVH
import torch.nn as nn
from torch import optim


parser = argparse.ArgumentParser()
parser.add_argument('--training_data_path', type=str,
                    default='../Data/train-pats/', help='training folder')
parser.add_argument('--validation_data_path', type=str,
                    default='../Data/validation-pats/', help='validation data folder')
parser.add_argument('--model_save_path', type=str,
                    default='./trained_models/', help='model save path')

parser.add_argument('--criterion', type=nn.Module,
                    default=Loss_DVH(), help='model save path')

parser.add_argument('--epochs', type=int,
                    default=200, help='the epochs for traning')
parser.add_argument('--lr_scheduler', type=str,
                    default=None, help='learning rate schedular')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size for training and validation')

args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(args.training_data_path):
        raise SystemExit('dicom folder not exist')

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    training_dataset = DataLoader(data_folder = args.training_data_path)
    validation_dataset = DataLoader(data_folder = args.validation_data_path)
    dataloader_training = data.DataLoader(dataset=training_dataset,
                                          batch_size=args.batch_size, 
                                          shuffle=True)

    dataloader_validation = data.DataLoader(dataset=validation_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        torch.device('cpu')

    model = Att_UNet(n_channels=11, n_classes=1).to(device)
    #criterion = torch.nn.MSELoss()
    criterion = args.criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    print(args.model_save_path)
    

    trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  model_save_path = args.model_save_path,
                  lr_scheduler=None,
                  epochs=args.epochs,
                  epoch=0,
                  notebook=True)
                  
    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    
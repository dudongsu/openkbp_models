import os
import numpy as np
from os import listdir
import argparse
import logging
import os
import random
import numpy as np
from networks.DenseUnet import DenseUnet
from networks.unet import Unet
from networks.AttUnet_model import Att_UNet
from networks.AttUnet_dropout import AttUnet_dropout
from networks.cascaded_unet import Cascade_Unet
from networks.unet_dcnn import UNet_dcnn
from networks.trainer import Trainer
from torchsummary import summary
from dataloader.data_loader import DataLoader
from torch.utils import data
import torch
from networks.loss import Loss_L1, Loss_MSE, Loss_weightedMSE, Loss_cas
import torch.nn as nn
from torch import optim
from dataloader.general_functions import get_paths, load_file
from random import sample
from numpy import savetxt
from networks.weight_init import init_weights
from torchsummary import summary


parser = argparse.ArgumentParser()
parser.add_argument('--training_data_path', type=str,
                    default='./Data/train-pats/', help='training folder')
parser.add_argument('--validation_data_path', type=str,
                    default='./Data/validation-pats/', help='validation data folder')
parser.add_argument('--model_save_path', type=str,
                    default='./trained_models/', help='model save path')

parser.add_argument('--criterion', type=nn.Module,
                    default=Loss_L1(), help='model save path')

parser.add_argument('--epochs', type=int,
                    default=200, help='the epochs for traning')
parser.add_argument('--lr_scheduler', type=str,
                    default=None, help='learning rate schedular')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size for training and validation')

parser.add_argument('--random_seed', type=int,
                    default=None, help='a random seed for train validation splitting')

parser.add_argument('--training_validation_path', type=str,
                    default='../Data/train_validation/', help='training validation folder')

parser.add_argument('--transform', type=str,
                    default= None, help='data augmentation prob')

args = parser.parse_args()




if __name__ == "__main__":

    if not os.path.exists(args.training_data_path):
        raise SystemExit('dicom folder not exist')

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    random.seed(args.random_seed)

    picked_train = sample(range(240), 200)
    picked_validation = list(set(range(240))-set(picked_train))

    if args.random_seed!=None:
        training_dataset = DataLoader(data_folder = args.training_validation_path,  mode_name='train', picked = picked_train)
        validation_dataset = DataLoader(data_folder = args.training_validation_path,  mode_name='val', picked=picked_validation)
    else:
        training_dataset = DataLoader(data_folder = args.training_data_path,  mode_name='train')
        validation_dataset = DataLoader(data_folder = args.validation_data_path,  mode_name='val')
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
    torch.cuda.empty_cache()
    # Unet
  #  model = Unet(n_channels=11, n_classes=1).to(device)
   # init_weights(model,init_type='kaiming')
    # Attention Unet model
 #   model = Att_UNet(n_channels=11, n_classes=1).to(device)
    # Attention Unet with dropout 
 #   model = AttUnet_dropout(n_channels=11, n_classes=1).to(device)
       
    ## unet_dcnn
  
  #  model = UNet_dcnn().to(device)
    # Cascade model
    model = Cascade_Unet(in_ch=11, out_ch=1,
                         list_ch_A=[-1, 16, 32, 64, 128, 256],
                         list_ch_B=[-1, 32, 64, 128, 256, 512]).to(device)
    
    #criterion = torch.nn.MSELoss()
    summary(model, (11, 128, 128,128))
    criterion = args.criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    print(args.model_save_path)
    

    trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  model_save_path = args.model_save_path,
                  lr_scheduler=args.lr_scheduler,
                  epochs=args.epochs,
                  epoch=0,
                  notebook=True)
                  
    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    savetxt(args.model_save_path+'train_losses.csv', training_losses, delimiter=',')
    savetxt(args.model_save_path+'validation_losses.csv', validation_losses, delimiter=',')

'''

def multiple_train(ensemble = 5, epochs = 5):

    
    training_validation_path = './Data/train_validation_1/'
    for i in range(ensemble):
        random.seed(i)
        n_total = 8
        picked_train = sample(range(n_total), 4)
        picked_validation = list(set(range(n_total))-set(picked_train))
        training_dataset = DataLoader(data_folder = training_validation_path, picked = picked_train)
        validation_dataset = DataLoader(data_folder = training_validation_path, picked=picked_validation)
        dataloader_training = data.DataLoader(dataset=training_dataset,
                                          batch_size=2, 
                                          shuffle=True)
        dataloader_validation = data.DataLoader(dataset=validation_dataset,
                                            batch_size=2,
                                            shuffle=True)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            torch.device('cpu')
    
        # Attention Unet model
        model = Att_UNet(n_channels=11, n_classes=1).to(device)
        criterion = Loss_L1()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
        model_save_path = './trained_models'+str(i)+'/'
        lr_scheduler = 'MultiStepLR'
        trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  model_save_path = model_save_path,
                  lr_scheduler= lr_scheduler,
                  epochs=epochs,
                  epoch=0,
                  notebook=True)
                  
        training_losses, validation_losses, lr_rates = trainer.run_trainer()
        savetxt(model_save_path+'train_losses.csv', training_losses, delimiter=',')
        savetxt(model_save_path+'validation_losses.csv', validation_losses, delimiter=',')


'''




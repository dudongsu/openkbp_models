import os
import sys
import argparse
from tqdm import tqdm
from networks.cascaded_unet import Cascade_Unet
from networks.unet_dcnn import UNet_dcnn
from networks.AttUnet_model import Att_UNet
from networks.AttUnet_dropout import AttUnet_dropout
from networks.DenseUnet import DenseUnet
from networks.unet import Unet
from dataloader.data_loader import DataLoader
from torch.utils import data
from networks.cascaded_unet import Cascade_Unet
import numpy as np
import torch
from pathlib import Path
from torch import optim
import pandas as pd
from dataloader.general_functions import sparse_vector_function
import os
from numpy import savetxt
import time



class TrainerLog:
    def __init__(self):
        self.iter = -1
        self.epoch = -1

        # Moving average loss, loss is the smaller the better
        self.moving_train_loss = None
        # Average train loss of a epoch
        self.average_train_loss = 99999999.
        self.best_average_train_loss = 99999999.
        # Evaluation index is the higher the better
        self.average_val_index = -99999999.
        self.best_average_val_index = -99999999.

        # Record changes in training loss
        self.list_average_train_loss_associate_iter = []
        # Record changes in validation evaluation index
        self.list_average_val_index_associate_iter = []
        # Record changes in learning rate
        self.list_lr_associate_iter = []

        # Save status of the trainer, eg. best_train_loss, latest, best_val_evaluation_index
        self.save_status = []


class Inference_ens:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 test_DataLoader: torch.utils.data.Dataset = None,
                 model_weight_paths: list=None,
                 notebook: bool = False
                 ):

        self.model = model
        self.lr_scheduler = None
        self.test_DataLoader = test_DataLoader
        self.device = device 
        self.notebook = notebook
        self.model_weight_paths = model_weight_paths
        

    def run_test(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        batch_iter = tqdm(enumerate(self.test_DataLoader), 'Test', total=len(self.test_DataLoader),leave=False)

        for i  in tqdm(range(len(self.test_DataLoader))):
            folder = self.test_DataLoader.file_paths_list[i]
            (x, y, possible_mask) =  self.test_DataLoader.__getitem__(i)
            x = np.expand_dims(x, axis = 0)
            x = torch.from_numpy(x).type(torch.float32)
            input, target, possible_mask = x.to(self.device), y.to(self.device), possible_mask.to(self.device)
            dose_pred = None
            flag = 0
            for model_weight_path in self.model_weight_paths:
                model_weights = torch.load(model_weight_path)
                self.model.load_state_dict(model_weights)
                with torch.no_grad():
                    out = self.model(input)
                out = out.cpu().numpy()
                out[out<0] = 0
                if flag == 0:
                    dose_pred = np.zeros(out.shape)
                    flag = 1
                
                dose_pred = out+dose_pred
            dose_pred = dose_pred/len(self.model_weight_paths)
            dose_pred = np.squeeze(dose_pred)
            possible_mask = np.squeeze(possible_mask.cpu().numpy())
            print('possible mask dimention', possible_mask.shape)
            dose_pred = np.multiply(dose_pred, possible_mask)
            dose_pred = np.transpose(dose_pred, axes=[1, 2, 0])
            dose_pred = np.squeeze(dose_pred)*70.0
            dose_to_save = sparse_vector_function(dose_pred)
            print(dose_to_save)
            dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
                                   columns=['data'])
            dose_df.to_csv('{}/{}.csv'.format(folder, 'dose_pred'))


# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str,
                    default='./Data/test-pats/', help='test data path folder')


args = parser.parse_args()

if __name__ == "__main__":
    print(args.test_data_path)
    if not os.path.exists(args.test_data_path):
        raise SystemExit('test folder not exist')
    

    test_dataset = DataLoader(data_folder = args.test_data_path, mode_name ='val')

    model = Att_UNet(n_channels=11, n_classes=1).to(device)
    weight_1 = './trained_models/AttUnet_MAE.pt'
    weight_2 = './trained_models/AttUnet_MAE_1.pt'
    weight_3 = './trained_models/AttUnet_MAE_2.pt'
    
    tester = Inference_ens(model=model,
                  device=device,
                  test_DataLoader=test_dataset,
                  model_weight_paths = [weight_1, weight_2, weight_3]
                  )

    print('\n\n# Start ensamble inference !')
    tester.run_test()
    
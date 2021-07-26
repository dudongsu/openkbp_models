import numpy as np
import torch
from pathlib import Path
from torch import optim

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


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 model_save_path: str=None,
                 lr_scheduler: str=None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False
                 ):

        self.log = TrainerLog()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = None
        self.set_lr_scheduler(lr_scheduler)
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.model_save_path = model_save_path

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.min_valid_loss = float('inf')

    def set_lr_scheduler(self, lr_scheduler_type):
        
        if lr_scheduler_type == 'stepLR':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

        elif lr_scheduler_type == 'MultiStepLR':
            self.lr_scheduler_type = 'MultiStepLR'
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                               milestones=[30,80], 
                                                               gamma=0.1,
                                                               last_epoch=-1
                                                               )
        elif lr_scheduler_type == 'cosine':
            self.lr_scheduler_type = 'cosine'
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.setting.optimizer,
                                                                     T_max=200,
                                                                     eta_min=0,
                                                                     last_epoch=-1
                                                                     )
        elif lr_scheduler_type == 'ReduceLROnPlateau':
            self.lr_scheduler_type = 'ReduceLROnPlateau'
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                     mode='min',
                                                                     factor=0.1,
                                                                     patience=10,
                                                                     verbose=True,
                                                                     threshold=1e-4,
                                                                     threshold_mode='rel',
                                                                     cooldown=0,
                                                                     min_lr=0,
                                                                     eps=1e-08)
        elif lr_scheduler_type == 'OneCycleLR':
            self.lr_scheduler_type = 'OneCycleLR'
            self.lr_scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                              max_lr=0.001,
                                                              factor=0.1,
                                                              epochs=self.epochs,
                                                              steps_per_epoch=len(self.training_DataLoader)
                                                              )
    
    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter
            print('Epoch-{0} lr: {1}'.format(self.epoch, self.optimizer.param_groups[0]['lr']))
            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None and self.lr_scheduler != 'OneCycleLR':
                if self.validation_DataLoader is not None and self.lr_scheduler == 'ReduceLROnPlateau':
                    self.lr_scheduler.step(self.validation_loss[-1])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.step()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y, possible_mask) in batch_iter:
            print('batch_iter ', i)
            input, target, possible_mask = x.to(self.device), y.to(self.device), possible_mask.to(self.device)   # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, target, possible_mask, input)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            if self.lr_scheduler is not None and self.lr_scheduler == 'OneCycleLR':
                self.lr_scheduler.step()
            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y, possible_mask) in batch_iter:
            input, target, possible_mask = x.to(self.device), y.to(self.device), possible_mask.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target, possible_mask,input)
                loss_value = loss.item()
                valid_losses.append(loss_value)
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
        print(f'Validation Loss: {np.mean(valid_losses)}')
        self.log.list_average_val_index_associate_iter.append(np.mean(valid_losses))
        if self.min_valid_loss > np.mean(valid_losses):
            print(f'Validation Loss Decreased({self.min_valid_loss:.6f}--->{np.mean(valid_losses):.6f}) \t Saving The Model')
            self.min_valid_loss = np.mean(valid_losses)
            # Saving State Dict
            model_name =  'best_Unet_model.pt'
            torch.save(self.model.state_dict(), self.model_save_path+model_name)
        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()

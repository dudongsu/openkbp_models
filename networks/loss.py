import torch.nn as nn
import numpy as np
import torch


class Loss_L1(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, out, target, possible_mask, CT_structure_mask):

        out = out[possible_mask > 0]
        gt_dose = target[possible_mask > 0]

        L1_loss = self.L1_loss_func(out, gt_dose)
        return L1_loss


class Loss_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSELoss_fun = nn.MSELoss(reduction='mean')

    def forward(self, out, target, possible_mask, CT_structure_mask):

        out = out[possible_mask > 0]
        gt_dose = target[possible_mask > 0]

        MSE_loss = self.MSELoss_fun(out, gt_dose)
        return MSE_loss

class Loss_weightedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSELoss_fun = nn.MSELoss(reduction='mean')

    def forward(self, out, target, possible_mask, CT_structure_mask):
        # structure mask
        #'Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid', 'Esophagus', 'Larynx', 'Mandible', 'PTV56', 'PTV63','PTV70'
        structure_mask = CT_structure_mask[:, 1:CT_structure_mask.shape[1],:,:,:]
        out_total = out[possible_mask > 0]
        gt_dose_total = target[possible_mask > 0]
        MSE_loss = self.MSELoss_fun(out_total, gt_dose_total)
        weight = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1]
        for i in range(structure_mask.shape[0]):
            str_mask = structure_mask[:,i:i+1,:,:,:]
            out_str = out[str_mask > 0]
            gt_dose_str = target[str_mask > 0] 
            MSE_str = self.MSELoss_fun(out_str, gt_dose_str)
            MSE_loss = MSE_loss +  MSE_str * (weight[i]**2)
        return MSE_loss


class Loss_cas(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, out, target, possible_mask, CT_structure_mask):
        pred_A = out[0]
        pred_B = out[1]
        gt_dose = target
        possible_dose_mask = possible_mask

        pred_A = pred_A[possible_dose_mask > 0]
        pred_B = pred_B[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        L1_loss = 0.5 * self.L1_loss_func(pred_A, gt_dose) + self.L1_loss_func(pred_B, gt_dose)
        return L1_loss

class Loss_dosescore(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='sum')

    def forward(self, out, target, possible_mask):
        pred_A = out
        gt_dose = target
        L1_loss = self.L1_loss_func(pred_A, gt_dose) / torch.sum(torch.flatten(possible_mask))
        return L1_loss



def dose_score(reference_dose, new_dose, possible_mask):
    dose_score_vec = np.sum(np.abs(reference_dose.flatten() - new_dose.flatten())) / np.sum(possible_mask.flatten())
    return dose_score_vec

import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, out, target, possible_mask):

        out = out[possible_mask > 0]
        gt_dose = target[possible_mask > 0]

        L1_loss = self.L1_loss_func(out, gt_dose)
        return L1_loss
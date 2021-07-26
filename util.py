import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random

HN_structures = ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
                'Esophagus', 'Larynx', 'Mandible', 'PTV56', 'PTV63', 'PTV70']

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1]*imageA.shape[2]*imageA.shape[3])
    return err

def dice_coeff(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = y_true.flatten()
    flat_y_pred = y_pred.flatten()
    return (2. * np.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (np.sum(flat_y_true) + np.sum(flat_y_pred) + smoothing_factor)


class IndexTracker(object):
    def __init__(self, ax, X,fig,bmin,bmax):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.fig = fig
        self.slices, row, cols = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind,:,:],cmap='jet',vmin=bmin, vmax=bmax)
        fig.colorbar(self.im, ax=self.ax )
      #  self.im.colorbar()
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind,:, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def get_outline(data):
    if(data.ndim!=4):
        print('need to be a 4 dimension bool matrix!')
        return
    contour = np.zeros(data.shape)
    for i in range(0,data.shape[3]):
        for row in range(data.shape[1]):
            for col in range(data.shape[2]):
                if(data[0][row][col][i]==True and (col==0 or data[0][row][col-1][i]==False)):
                    contour[0][row][col][i] = i+1
                if(data[0][row][col][i]==True and (col==data.shape[2]-1 or data[0][row][col+1][i]==False)):
                    contour[0][row][col][i] = i+1
    
    for i in range(0,data.shape[3]):
        for col in range(data.shape[2]):
            for row in range(data.shape[1]):
                if(data[0][row][col][i]==True and (row==0 or data[0][row-1][col][i]==False)):
                    contour[0][row][col][i] = i+1
                if(data[0][row][col][i]==True and (col==data.shape[1]-1 or data[0][row+1][col][i]==False)):
                    contour[0][row][col][i] = i+1
    return contour
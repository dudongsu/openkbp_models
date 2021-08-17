from itertools import product as it_product

import numpy as np
import pandas as pd
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from util import IndexTracker


class EvaluateDose:
    """Evaluate a full dose distribution against the reference dose on the OpenKBP competition metrics"""

    def __init__(self, data_loader, dose_loader=None):
        """
        Prepare the class for evaluating dose distributions
        :param data_loader: a data loader object that loads data from the reference dataset
        :param dose_loader: a data loader object that loads a dose tensor from any dataset (e.g., predictions)
        """
        # Initialize objects
        self.data_loader = data_loader  # Loads data related to ground truth patient information
        self.dose_loader = dose_loader  # Loads the data for a benchmark dose

        # Initialize objects for later
        self.patient_list = None
        self.roi_mask = None
        self.new_dose = None
        self.reference_dose = None
        self.voxel_size = None
        self.possible_dose_mask = None

        # Set metrics to be evaluated
        self.oar_eval_metrics = ['D_0.1_cc', 'mean']
        self.tar_eval_metrics = ['D_99', 'D_95', 'D_1']

        # Name metrics for data frame
        oar_metrics = list(it_product(self.oar_eval_metrics, self.data_loader.rois['oars']))
        target_metrics = list(it_product(self.tar_eval_metrics, self.data_loader.rois['targets']))

        # Make data frame to store dose metrics and the difference data frame
        self.metric_difference_df = pd.DataFrame(index=self.data_loader.patient_id_list,
                                                 columns=[*oar_metrics, *target_metrics])
        self.reference_dose_metric_df = self.metric_difference_df.copy()
        self.new_dose_metric_df = self.metric_difference_df.copy()

    def make_metrics(self):
        """Calculate a table of
        :return: the DVH score and dose score for the "new_dose" relative to the "reference_dose"
        """
      #  num_batches = self.data_loader.number_of_batches()
        num_batches = len(self.data_loader)
        dose_score_vec = np.zeros(num_batches)
        # Only make calculations if data_loader is not empty
        if not self.data_loader.file_paths_list:
            print('No patient information was given to calculate metrics')
        else:
            # Change batch size to 1
            self.data_loader.batch_size = 1  # Loads data related to ground truth patient information
            if self.dose_loader is not None:
                self.dose_loader.batch_size = 1  # Loads data related to ground truth patient information

            for idx in tqdm.tqdm(range(num_batches)):
                # Get roi masks for patient
                self.get_constant_patient_features(idx)
                # Get dose tensors for reference dose and evaluate criteria
                reference_dose = self.get_patient_dose_tensor(self.data_loader, idx)
                if reference_dose is not None:
                    self.reference_dose_metric_df = self.calculate_metrics(self.reference_dose_metric_df, reference_dose)
                # If a dose loader was provided, calculate the score
                if self.dose_loader is not None:
                    new_dose = self.get_patient_dose_tensor(self.dose_loader, idx, dose = 'dose_pred')
                    # Make metric data frames
                    self.new_dose_metric_df = self.calculate_metrics(self.new_dose_metric_df, new_dose)
                    # Evaluate mean absolute error of 3D dose
                    dose_score_vec[idx] = np.sum(np.abs(reference_dose - new_dose)) / np.sum(self.possible_dose_mask)
                    # Save metrics at the patient level (this is a template for how DVH stream participants could save
                    # their files
                    # self.dose_metric_df.loc[self.patient_list[0]].to_csv('{}.csv'.format(self.patient_list[0]))

            if self.dose_loader is not None:
                dvh_score = np.nanmean(np.abs(self.reference_dose_metric_df - self.new_dose_metric_df).values)
                dvh_score_std = np.nanstd(np.abs(self.reference_dose_metric_df - self.new_dose_metric_df).values)
                dose_score = dose_score_vec.mean()
                dose_score_std = dose_score_vec.std()
                return dvh_score, dose_score, dvh_score_std, dose_score_std
            else:
                print('No new dose provided. Metrics were only calculated for the provided dose.')


    def get_patient_dose_tensor(self, data_loader, idx= None, dose = 'dose'):
        """Retrieves a flattened dose tensor from the input data_loader.
        :param data_loader: a data loader that load a dose distribution
        :return: a flattened dose tensor
        """
        # Load the dose for the request patient
        dose_batch = data_loader.get_batch(idx)
        dose_key = [key for key in dose_batch.keys() if dose in key.lower()][0]  # The name of the dose
        dose_tensor = dose_batch[dose_key][0]  # Dose tensor
        return dose_tensor.flatten()

    def get_constant_patient_features(self, idx):
        """Gets the roi tensor
        :param idx: the index for the batch to be loaded
        """
        # Load the batch of roi mask
        rois_batch = self.data_loader.get_batch(idx)
        self.roi_mask = rois_batch['structure_masks'][0].astype(bool)
        # Save the patient list to keep track of the patient id
        self.patient_list = rois_batch['patient_list']
        # Get voxel size
        self.voxel_size = np.prod(rois_batch['voxel_dimensions'])
        # Get the possible dose mask
        self.possible_dose_mask = rois_batch['possible_dose_mask']


    def calculate_metrics(self, metric_df, dose):
        """
        Calculate the competition metrics
        :param metric_df: A DataFrame with columns indexed by the metric name and the structure name
        :param dose: the dose to be evaluated
        :return: the same metric_df that is input, but now with the metrics for the provided dose
        """
        # Prepare to iterate through all rois
        roi_exists = self.roi_mask.max(axis=(0, 1, 2))
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100/self.voxel_size))  #
        for roi_idx, roi in enumerate(self.data_loader.full_roi_list):
            if roi_exists[roi_idx]:
                roi_mask = self.roi_mask[:, :, :, roi_idx].flatten()
                roi_dose = dose[roi_mask]
                roi_size = len(roi_dose)
                if roi in self.data_loader.rois['oars']:
                    if 'D_0.1_cc' in self.oar_eval_metrics:
                        # Find the fractional volume in 0.1cc to evaluate percentile
                        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc/roi_size * 100
                        metric_eval = np.percentile(roi_dose, fractional_volume_to_evaluate)
                        metric_df.at[self.patient_list[0], ('D_0.1_cc', roi)] = metric_eval
                    if 'mean' in self.oar_eval_metrics:
                        metric_eval = roi_dose.mean()
                        metric_df.at[self.patient_list[0], ('mean', roi)] = metric_eval
                elif roi in self.data_loader.rois['targets']:
                    if 'D_99' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 1)
                        metric_df.at[self.patient_list[0], ('D_99', roi)] = metric_eval
                    if 'D_95' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 5)
                        metric_df.at[self.patient_list[0], ('D_95', roi)] = metric_eval
                    if 'D_1' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 99)
                        metric_df.at[self.patient_list[0], ('D_1', roi)] = metric_eval

        return metric_df

    def make_metrics_ROI(self):
        num_batches = len(self.data_loader)
        dose_score_vec = {}
        dose_score_vec['PTV70'] =[]
        dose_score_vec['PTV63'] =[]
        dose_score_vec['PTV56'] =[]
        dose_score_vec['Brainstem'] = []
        dose_score_vec['SpinalCord'] = []
        dose_score_vec['RightParotid'] = []
        dose_score_vec['LeftParotid'] = []
        dose_score_vec['Mandible'] = []
        dose_score_vec['Larynx'] = []


        if not self.data_loader.file_paths_list:
            print('No patient information was given to calculate metrics')
        else:
            # Change batch size to 1
            self.data_loader.batch_size = 1  # Loads data related to ground truth patient information
            if self.dose_loader is not None:
                self.dose_loader.batch_size = 1  # Loads data related to ground truth patient information

            for idx in tqdm.tqdm(range(num_batches)):
                 # Get roi masks for patient
                self.get_constant_patient_features(idx)
                # Get dose tensors for reference dose and evaluate criteria
                reference_dose = self.get_patient_dose_tensor(self.data_loader, idx)
                if reference_dose is not None:
                    self.reference_dose_metric_df = self.calculate_metrics(self.reference_dose_metric_df, reference_dose)
                # If a dose loader was provided, calculate the score
                if self.dose_loader is not None:
                    new_dose = self.get_patient_dose_tensor(self.dose_loader, idx, dose = 'dose_pred')
                    # Make metric data frames
                    for roi_idx, roi_ in enumerate(self.data_loader.full_roi_list):
                        for organ in dose_score_vec.keys():
                            if roi_ == organ:
                                roi_mask = self.roi_mask[:, :, :, roi_idx].flatten()
                                roi_dose_reference = reference_dose[roi_mask]
                                roi_dose_new = new_dose[roi_mask]
                                if np.sum(roi_mask)>0:
                                    dose_score_vec[organ].append(np.sum(np.abs(roi_dose_reference - roi_dose_new)) / np.sum(roi_mask))
            return dose_score_vec
        
        

    def plot_DVH(self, idx):
        """
        plot DVH
        """
        # Prepare to iterate through all rois

        self.get_constant_patient_features(idx)

        reference_dose = self.get_patient_dose_tensor(self.data_loader, idx)
        if self.dose_loader is not None:
            new_dose = self.get_patient_dose_tensor(self.dose_loader, idx, dose = 'dose_pred')

        # create DVH curve for reference dose
        roi_exists = self.roi_mask.max(axis=(0, 1, 2))
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100/self.voxel_size))
        DVH_bin = 5000
        DVH_inv = 90.0*1.0/DVH_bin
        dose_bin = np.zeros(DVH_bin)
        dose_bin_plot = np.arange(0,DVH_bin)*DVH_inv
        dose_bin = np.arange(-1,DVH_bin)*DVH_inv
        DVH_all_ref = defaultdict()
        DVH_all_pred = defaultdict()

        for roi_idx, roi in enumerate(self.data_loader.full_roi_list):
            print('roi name ', roi)
            if roi_exists[roi_idx]:
                roi_mask = self.roi_mask[:, :, :, roi_idx].flatten()
                roi_dose_ref = reference_dose[roi_mask]
                roi_dose_pred = new_dose[roi_mask]
                max_dose_ref = np.max(roi_dose_ref)
                max_dose_pred = np.max(roi_dose_pred)
                roi_size = len(roi_dose_ref)
                print('roi size', roi_size)
                DVH = np.zeros(DVH_bin)
                DVH_diff_ref, bin_edges = np.histogram(roi_dose_ref,dose_bin)
                DVH = np.cumsum(DVH_diff_ref)
                DVH = 1 - DVH/DVH.max()
                DVH_all_ref[roi] = DVH

                DVH = np.zeros(DVH_bin)
                DVH_diff_ref, bin_edges = np.histogram(roi_dose_pred,dose_bin)
                DVH = np.cumsum(DVH_diff_ref)
                DVH = 1 - DVH/DVH.max()
                DVH_all_pred[roi] = DVH

                

     #   self.calculate_metrics(self.new_dose_metric_df, new_dose)
        fig = plt.figure(dpi=1200)
        roi_legend = []
        for roi in DVH_all_ref.keys():
            r = random.uniform(0, 1); g = random.uniform(0, 1); b = random.uniform(0, 1)
            line, = plt.plot(dose_bin_plot,DVH_all_ref[roi]*100, color = (r,g,b), linewidth=2, label=roi)
            plt.plot(dose_bin_plot,DVH_all_pred[roi]*100, color = (r,g,b), linewidth=2, linestyle='dashed', label=roi)
            roi_legend.append(line)
        
        plt.ylabel('volume %')
        plt.legend(handles=roi_legend,bbox_to_anchor=(1.1, 1.05),prop={'size': 6}) 
        plt.show()
        plt.savefig('DVH.png', dpi=300)
        print('dose shape', reference_dose.shape)



    def plot_dose(self, idx, slice):
        """
        plot dose and dose different
        """
        # Prepare to iterate through all rois

        self.get_constant_patient_features(idx)

        dose_batch = self.data_loader.get_batch(idx)
        ct = dose_batch['ct'].squeeze()
        dose_key = [key for key in dose_batch.keys() if 'dose' in key.lower()][0]  # The name of the dose
        reference_dose = dose_batch[dose_key][0].squeeze()  # Dose tensor
        if self.dose_loader is not None:
            dose_batch = self.dose_loader.get_batch(idx)
            dose_key = [key for key in dose_batch.keys() if 'dose_pred' in key.lower()][0]  # The name of the dose
            new_dose = dose_batch[dose_key][0].squeeze()  # Dose tensor
        print('dose shape is ', reference_dose.shape)
        # create DVH curve for reference dose
        reference_dose = np.transpose(reference_dose, axes=[2, 0, 1])
        new_dose = np.transpose(new_dose, axes=[2, 0, 1])
        ct = np.transpose(ct, axes=[2, 0, 1])
        dose_diff = new_dose - reference_dose
    
        
        max_img = np.max(ct.flatten())
        max_dose = np.max(reference_dose.flatten())
        max_dose_diff = np.max(abs(dose_diff.flatten()))
        
        fig = plt.figure(figsize=(10, 12), dpi=300)
        ax1 = fig.add_subplot(311)
        im1 = ax1.imshow(reference_dose[slice,:,:].squeeze(), interpolation='None')
        fig.colorbar(im1, orientation='vertical')
        ax2 = fig.add_subplot(312)
        im2 = ax2.imshow(new_dose[slice,:,:].squeeze(), vmin=0, vmax=max_dose)
        fig.colorbar(im2, orientation='vertical')
        ax3 = fig.add_subplot(313)
        im3 = ax3.imshow(dose_diff[slice,:,:].squeeze(), vmin=-max_dose_diff, vmax=max_dose_diff)
        fig.colorbar(im3, orientation='vertical')

        plt.show()
    
   

##### end of Evaluation class



def plot_loss(train_loss_file='./trained_models/train_losses.csv', 
              validation_loss_file = './trained_models/validation_losses.csv'):
    train_losses = np.genfromtxt(train_loss_file,delimiter=',')
    validation_losses = np.genfromtxt(validation_loss_file,delimiter=',')
    fig = plt.figure(figsize=(10, 12), dpi=100)
    plt.plot(range(len(train_losses)),train_losses, 'r', linewidth=2, label='train')
    plt.plot(range(len(validation_losses)),validation_losses, 'r', linewidth=2, linestyle='dashed', label='val')
    plt.ylabel('l1 loss')
  #  plt.legend(handles=roi_legend,bbox_to_anchor=(1.1, 1.05),prop={'size': 6}) 
    plt.show
    
        

def plot_loss_compare(train_loss_1='./trained_models/train_losses.csv', 
              validation_loss_1 = './trained_models/validation_losses.csv', 
              train_loss_2 = './trained_models/train_losses_AttUnet.csv',
              validation_loss_2 = './trained_models/validation_losses_AttUnet.csv'):
    train_losses_1 = np.genfromtxt(train_loss_1,delimiter=',')
    validation_losses_1 = np.genfromtxt(validation_loss_1,delimiter=',')
    train_losses_2 = np.genfromtxt(train_loss_2,delimiter=',')
    validation_losses_2 = np.genfromtxt(validation_loss_2,delimiter=',')
    epochs = 21
    fig = plt.figure(figsize=(10, 12), dpi=100)
    x = range(0, epochs)
    plt.plot(x,train_losses_1[0:epochs], 'r', linewidth=2, label='train_Unet')
    plt.plot(x,validation_losses_1[0:epochs], 'r', linewidth=2, linestyle='dashed', label='val Unet')
    plt.plot(x,train_losses_2[0:epochs], 'b', linewidth=2, label='train AttUnet')
    plt.plot(x,validation_losses_2[0:epochs], 'b', linewidth=2, linestyle='dashed', label='val AttUnet')
    plt.ylabel('loss')
    plt.legend()
  #  plt.legend(handles=roi_legend,bbox_to_anchor=(1.1, 1.05),prop={'size': 6}) 
    plt.show()


    




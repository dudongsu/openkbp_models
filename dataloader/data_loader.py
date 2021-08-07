import numpy as np

from .general_functions import get_paths, load_file
import torch.utils.data as data
import torch
import os
from augmentation.augmentation import *

class DataLoader(data.Dataset):

    def __init__(self, data_folder, patient_shape=(128, 128, 128), transform = None, mode_name='training_mode',picked = None):
        """Initialize the DataLoader class, which loads the data for OpenKBP
        :param file_paths_list: list of the directories or single files where data for each patient is stored
        :param batch_size: the number of data points to lead in a single batch
        :param patient_shape: the shape of the patient data
        :param shuffle: whether or not order should be randomized
        """
        # Set file_loader specific attributes
        self.rois = dict(oars=['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
                               'Esophagus', 'Larynx', 'Mandible'], targets=['PTV56', 'PTV63', 'PTV70'])

        self.patient_shape = patient_shape  # Shape of the patient
        self.mode_name = mode_name
        self.data_folder = data_folder
        print(self.data_folder)
        self.file_paths_list = get_paths(self.data_folder, ext='')
        if picked!=None:
            self.file_paths_list = np.array(self.file_paths_list)[picked]
        self.indices = np.arange(len(self.file_paths_list))  # Indices of file paths
        self.full_roi_list = sum(map(list, self.rois.values()), [])  # make a list of all rois
        self.num_rois = len(self.full_roi_list)
        self.transform = transform
        self.mode_name = mode_name
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.batch_size = 1
        self.patient_id_list = ['pt_{}'.format(k.split('/pt_')[1].split('/')[0].split('.csv')[0]) for k in
                                self.file_paths_list]  # the list of patient ids with information in this data loader
        self.required_files = {'dose': (self.patient_shape + (1,)),  # The shape of dose tensor
                               'ct': (self.patient_shape + (1,)),  # The shape of ct tensor
                               'structure_masks': (self.patient_shape + (self.num_rois,)),
                               # The shape of the structure mask tensor
                               'possible_dose_mask': (self.patient_shape + (1,)),
                               # Mask of where dose can be deposited
                               'voxel_dimensions': (3,)
                               # Physical dimensions (in mm) of voxels
                               }
        if self.mode_name=='evaluate':
            self.required_files['dose_pred'] = (self.patient_shape + (1,))


    
    def __len__(self):
        return len(self.patient_id_list)
    
    def __getitem__(self, index: int):
        # select the sample
        dict_data = self.get_batch(index)
        structure = dict_data['structure_masks'].squeeze()
        dose = dict_data['dose'].squeeze()
        ct = dict_data['ct'].squeeze()
        possible_mask = dict_data['possible_dose_mask'].squeeze()

        structure = np.array(structure, dtype='float32')
        dose = np.array(dose).astype('float32')/70.0
        ct = np.array(ct).astype('float32')
        np.clip(ct, 0, 3000)
        ct = ct/3000.0
        possible_mask = np.array(possible_mask).astype('int')

        structure = np.transpose(structure, axes=[2, 0, 1, 3])
        dose = np.transpose(dose, axes=[2, 0, 1])
        ct = np.transpose(ct, axes=[2, 0, 1])
        possible_mask = np.transpose(possible_mask, axes=[2, 0, 1])

        ct = np.expand_dims(ct, axis = 3)
        dose = np.expand_dims(dose, axis = 3)
        possible_mask = np.expand_dims(possible_mask, axis = 3)
        structure_masks = structure>0
        
        batch_X = np.array(np.concatenate((ct, structure_masks),axis=3))
        batch_Y = np.array(dose)

        x = batch_X 
        y = batch_Y 
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        possible_mask = torch.from_numpy(possible_mask).type(torch.int32)
        x = x.permute(3, 0, 1, 2)
        y = y.permute(3, 0, 1, 2)
        possible_mask = possible_mask.permute(3, 0, 1, 2)

        if self.mode_name == 'evaluate':
            predict_dose = dict_data['dose_pred'].squeeze()
            predict_dose = np.array(predict_dose).astype('float32')
            predict_dose = np.transpose(predict_dose, axes=[2, 0, 1])
            predict_dose = np.array(np.expand_dims(predict_dose, axis = 3))
            predict_dose = torch.from_numpy(predict_dose).type(torch.int32)
            predict_dose = predict_dose.permute(3, 0, 1, 2)
            return x, y, possible_mask, predict_dose

        return x, y, possible_mask


    def get_batch(self, index=None):
        """Loads one batch of data
        :param index: the index of the batch to be loaded
        :param patient_list: the list of patients for which to load data for
        :return: a dictionary with the loaded data
        """

        # Make a list of files to be loaded
       # file_paths_to_load = [self.file_paths_list[k] for k in indices]
        # Make a list of files to be loaded
        file_paths_to_load = [self.file_paths_list[index]]
        # Load the requested files as a tensors
        loaded_data = self.load_data(file_paths_to_load)
        return loaded_data


    def load_data(self, file_paths_to_load):
        """Generates data containing batch_size samples X : (n_samples, *dim, n_channels)
        :param file_paths_to_load: the paths of the files to be loaded
        :return: a dictionary of all the loaded files
        """

        # Initialize dictionary for loaded data and lists to track patient path and ids
        tf_data = {}.fromkeys(self.required_files)
        patient_list = []
        patient_path_list = []

        # Loop through each key in tf data to initialize the tensor with zeros
        for key in tf_data:
            # Make dictionary with appropriate data sizes for bath learning
            tf_data[key] = np.zeros((self.batch_size, *self.required_files[key]))

        # Generate data
        for i, pat_path in enumerate(file_paths_to_load):
            # Get patient ID and location of processed data to load
            patient_path_list.append(pat_path)
            pat_id = pat_path.split('/')[-1].split('.')[0]
            patient_list.append(pat_id)
            # Make a dictionary of all the tensors
            loaded_data_dict = self.load_and_shape_data(pat_path)
            # Iterate through the dictionary add the loaded data to the "batch channel"
            for key in tf_data:
                tf_data[key][i,] = loaded_data_dict[key]

        # Add two keys to the tf_data dictionary to track patient information
        tf_data['patient_list'] = patient_list
        tf_data['patient_path_list'] = patient_path_list

        return tf_data

    def load_and_shape_data(self, path_to_load):
        """ Reshapes data that is stored as vectors into matrices
        :param path_to_load: the path of the data that needs to be loaded. If the path is a directory, all data in the
         directory will be loaded. If path is a file then only that file will be loaded.
        :return: Loaded data with the appropriate shape
        """

        # Initialize the dictionary for the loaded files
        loaded_file = {}
        if '.csv' in path_to_load:
            loaded_file[self.mode_name] = load_file(path_to_load)
        else:
            files_to_load = get_paths(path_to_load, ext='')
            # Load files and get names without file extension or directory
            for f in files_to_load:
                f_name = f.split('/')[-1].split('.')[0]
                if f_name in self.required_files or f_name in self.full_roi_list:
                    loaded_file[f_name] = load_file(f)

        # Initialize matrices for features
        shaped_data = {}.fromkeys(self.required_files)
        for key in shaped_data:
            shaped_data[key] = np.zeros(self.required_files[key])

        # Populate matrices that were no initialized as []
        for key in shaped_data:
            if key == 'structure_masks':
                # Convert dictionary of masks into a tensor (necessary for tensorflow)
                for roi_idx, roi in enumerate(self.full_roi_list):
                    if roi in loaded_file.keys():
                        np.put(shaped_data[key], self.num_rois * loaded_file[roi] + roi_idx, int(1))
            elif key == 'possible_dose_mask':
                np.put(shaped_data[key], loaded_file[key], int(1))
            elif key == 'voxel_dimensions':
                shaped_data[key] = loaded_file[key]
            else:  # Files with shape
                np.put(shaped_data[key], loaded_file[key]['indices'], loaded_file[key]['data'])
            

        return shaped_data


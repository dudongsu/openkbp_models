import os
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict 
import math

import shutil
from data_loader import DataLoader
from dose_evaluation_class import EvaluateDose
from general_functions import get_paths, make_directory_and_return_path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import skimage.transform



def get_plans_kbp(matrix_size = (64, 64, 64), dataset_save_path = './preprocessing/open-kbp_data/npy_data_128', save_npy = True, batch_size = 1):
    """ Get the dataset from the open-kdp data folder of csv files, for each patient, we will 
        get the plan class, and resample and re-cut the plan, then save the plan to a folder
        matrix_size --> z, x, y 
        save the data to npy, [1, z_dim, x_dim, z_dim, n_feature+1], 1 means the label (dose) is concated to the end of the matrix
    """
    primary_directory = os.getcwd()

    main_data_dir = os.getcwd()
    training_data_dir = '{}/train-pats'.format(main_data_dir)
    validation_data_dir = '{}/validation-pats'.format(main_data_dir)
    testing_data_dir = '{}/test-pats'.format(main_data_dir)

    train_plan_paths = get_paths(training_data_dir, ext='')
    validation_plan_paths = get_paths(validation_data_dir, ext='')
    test_plan_paths = get_paths(testing_data_dir, ext='')

    training_save_path = '{}/npy_data_128/train'.format(main_data_dir)
    validation_save_path = '{}/npy_data_128/validation'.format(main_data_dir)
    test_save_path = '{}/npy_data_128/test'.format(main_data_dir)

    if training_save_path[-1]!='/': training_save_path = training_save_path+'/'
    if validation_save_path[-1]!='/': validation_save_path = validation_save_path+'/'
    if test_save_path[-1]!='/': test_save_path = test_save_path+'/'

    if not os.path.exists(training_save_path):
        os.makedirs(training_save_path)
    if not os.path.exists(validation_save_path):
        os.makedirs(validation_save_path)
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)


    ## get training npy data into the training folder
    train_validation_test = zip([train_plan_paths, validation_plan_paths, test_plan_paths], 
                                [training_save_path, validation_save_path, test_save_path])

    for plan_paths, save_path in train_validation_test:
        data_loader = DataLoader(plan_paths, batch_size=1)
        data_loader.patient_id_list
        pt_list = data_loader.patient_id_list

        for i, pt in enumerate(pt_list):
            test_data = data_loader.get_batch(index = i)
            structure = test_data['structure_masks'].squeeze()
            dose = test_data['dose'].squeeze()
            ct = test_data['ct'].squeeze()
            # scale the matrix to [64,64,64]
        #    structure = skimage.transform.resize(image=structure, output_shape=[64,64,64,10])
        #    dose = skimage.transform.resize(image=dose, output_shape=[64,64,64])
        #    ct = skimage.transform.resize(image=ct, output_shape=[64,64,64])


            structure = np.array(structure, dtype='float32')
            dose = np.array(dose).astype('float32')
            ct = np.array(ct).astype('float32')
            structure = np.transpose(structure, axes=[2, 0, 1, 3])
            dose = np.transpose(dose, axes=[2, 0, 1])
            ct = np.transpose(ct, axes=[2, 0, 1])

            dose = np.expand_dims(dose, axis = 3)
            structure_masks = structure>=0.5
            batch_npy = np.concatenate((structure_masks,dose),axis=3)
            batch_npy = np.expand_dims(batch_npy, axis = 0)
            batch_npy = np.array(batch_npy)
            np.save(save_path +'/batch_{0}_64.npy'.format(pt), batch_npy)


if __name__ == '__main__':
    get_plans_kbp()
    
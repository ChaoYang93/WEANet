#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import sktime
from sktime.utils.data_io import load_from_tsfile_to_dataframe

import pandas as pd
import numpy as np

from sktime.utils.data_processing import (
    from_3d_numpy_to_nested,
    from_multi_index_to_3d_numpy,
    from_nested_to_3d_numpy,
    is_nested_dataframe,
)


# In[ ]:


# fixed length dataset 
folder_list = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
    "UWaveGestureLibrary"
]


# In[ ]:


def mkdir(path):
    folder = os.path.exists(path)
    
    if not folder:
        os.makedirs(path)


# In[ ]:


def deal_file(folder_name):
    path = os.getcwd()
    data_path = os.path.join(path, "data")
    
    file_name_train = folder_name + '/' + folder_name + '_TRAIN.ts'
    file_name_test = folder_name + '/' + folder_name + '_TEST.ts'
    
    train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(data_path, file_name_train))
    test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(data_path, file_name_test))
    
    train_x_3d = from_nested_to_3d_numpy(train_x)
    test_x_3d = from_nested_to_3d_numpy(test_x)
    
    new_folder_name = "npydata" + "/" + folder_name
    mkdir(new_folder_name)
    
    train_x_file_name = new_folder_name + "/" + folder_name + "_train_x.npy"
    train_y_file_name = new_folder_name + "/" + folder_name + "_train_y.npy"
    test_x_file_name = new_folder_name + "/" + folder_name + "_test_x.npy"
    test_y_file_name = new_folder_name + "/" + folder_name + "_test_y.npy"
    
    np.save(train_x_file_name, train_x_3d)
    np.save(train_y_file_name, train_y)
    np.save(test_x_file_name, test_x_3d)
    np.save(test_y_file_name, test_y)

    print(folder_name + " save .npy done")


# In[ ]:


for i in folder_list:
    deal_file(i)


# In[ ]:


# different length dataset
folder_list2 = [
    "CharacterTrajectories",
    #"InsectWingbeat",
    "JapaneseVowels",
    "SpokenArabicDigits"
]


# In[ ]:


def pad_nested(ndf): #input:nested dataframe
    ndf_padded = pd.DataFrame(index=ndf.index, columns=ndf.columns)
    len_t = ndf.iloc[0,0].shape[0] #find the max length of series
    for i in range(0,ndf.shape[0]):
        for j in range(0,ndf.shape[1]):
            if len_t < ndf.iloc[i,j].shape[0]:
                len_t = ndf.iloc[i,j].shape[0]

    for i in range(0,ndf.shape[0]):
        for j in range(0,ndf.shape[1]):
            ndf_padded.iloc[i,j] = ndf.iloc[i,j].reindex(range(0,len_t)).fillna(0)
    
    return(ndf_padded)


# In[ ]:


def deal_file2(folder_name):
    path = os.getcwd()
    data_path = os.path.join(path, "data")
    
    file_name_train = folder_name + '/' + folder_name + '_TRAIN.ts'
    file_name_test = folder_name + '/' + folder_name + '_TEST.ts'
    
    train_x, train_y = load_from_tsfile_to_dataframe(os.path.join(data_path, file_name_train))
    test_x, test_y = load_from_tsfile_to_dataframe(os.path.join(data_path, file_name_test))
    
    train_x_3d = from_nested_to_3d_numpy(pad_nested(train_x)) #padding
    test_x_3d = from_nested_to_3d_numpy(pad_nested(test_x)) #padding
    
    new_folder_name = "npydata" + "/" + folder_name
    mkdir(new_folder_name)
    
    train_x_file_name = new_folder_name + "/" + folder_name + "_train_x.npy"
    train_y_file_name = new_folder_name + "/" + folder_name + "_train_y.npy"
    test_x_file_name = new_folder_name + "/" + folder_name + "_test_x.npy"
    test_y_file_name = new_folder_name + "/" + folder_name + "_test_y.npy"
    
    np.save(train_x_file_name, train_x_3d)
    np.save(train_y_file_name, train_y)
    np.save(test_x_file_name, test_x_3d)
    np.save(test_y_file_name, test_y)

    print(folder_name + " save .npy done")


# In[ ]:


for i in folder_list2:
    deal_file2(i)


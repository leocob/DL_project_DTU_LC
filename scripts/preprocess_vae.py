#!/usr/bin/env pythonf
"""preprocess_vae.py: script that contains the preprocessing methods prior to loading into pytorch """

import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def metadata_mapping(X, metadata_path):
    """Reads metadata and list of encodings and outputs list of targets
    Parameters
    ----------
    encodings_file : path to encoding file
    metadata_path   : path to metadata file

    Returns
    -------
    targets         : list of targets linked to X
    """    
    # I read the csv file containing the list of patients and their annotated ancestries
    metadata = pd.read_csv(metadata_path, sep="\t", names=["index", "index2", "encoding", "encoding2", "time_num", "time_str", "Country", "SNP_cov", "SNP_cov2", "SNP_cov3", "Annotated_ancestry"], header=None)

    # # Replace nans by "Unknown"
    metadata = metadata.replace(np.nan, "Unknown")

    # I create a new column
    metadata = metadata.assign(patient_string = "_patient.pt")

    # I concatenate the two columns
    metadata["patient_index"] = metadata["index"].astype(str) + metadata["patient_string"]

    # I reorder the dataframe
    metadata = metadata[["index", "index2", "patient_index", "Annotated_ancestry", "Country", "encoding", "encoding2", "time_num", "time_str", "SNP_cov", "SNP_cov2","SNP_cov3", "patient_string"]]

    # I create a dictionary that maps the patient number and the annotated ancestry
    mapping_dictionary = metadata.set_index("patient_index").to_dict()["Annotated_ancestry"]

    # I create the targets
    targets = [mapping_dictionary[x] for x in X]

    return targets

def one_hot_encoding(list_to_encode):
    """Hot encodes a list of strings
    Parameters
    ----------
    list_to_encode : list containing strings

    Returns
    -------
    one_hot_encoded : one-hot encoded numpy array
    """
    # I convert the list into a numpy array of shape (6155,)
    array = np.array(list_to_encode)

    # I reshape the array to a shape (6155,1)
    array = array.reshape(len(array),1)

    # I initialize the OneHotEncoder function from scikit-learn
    scikit_onehotencoder = OneHotEncoder(sparse=False)

    # I one-hot encode the array, the new shape is (6155, 13)
    encoded_array = scikit_onehotencoder.fit_transform(array)

    return encoded_array


def split_train_test(X, targets, prop):
    """Splits X and targets np.arrays into train and test according to prop
    Parameters
    ----------
    X : vector of inputs
    targets : vector of targets
    prop : floating point corresponding to the training test partition
    
    Returns
    -------
    X_train : X * prop
    X_test X * (1-prop)
    targets_train : targets * prop
    targets_test : targets * (1-prop)

    X_train = X[:int(len(X)*prop)]
    X_test = X[int(len(X)*prop+1):]
    
    y_train = targets[:int(len(X)*prop)]
    y_test = targets[int(len(X)*prop+1):]
    """
    # Implement your code
    X_train, X_test, y_train, y_test = train_test_split(X, targets, train_size= prop, test_size=1-prop, random_state=42, stratify=None)
    return X_train, X_test, y_train, y_test


def impute_data(tensor, frequency_df="no_frequency", batch_size=20, impute_all=False, categorical=False):
    """Replaces tensor values with nas with zeroes
    Parameters
    ----------
    X : tensor 
    
    variants : path to list of variants matching the tensor 
    
    Returns
    -------
    tensor with replaced values
    """
    index_nan = (tensor!=tensor).nonzero()

    if categorical is True:
        tensor = tensor.cpu()
        shape = tensor.shape

        # Get index X == nan and frequencies for those
        tensor =  np.array(tensor)
        index_nan = np.where(np.isnan(tensor))
        tensor[index_nan] = np.zeros_like(tensor[index_nan])
        tensor = torch.FloatTensor(tensor)

        assert len((tensor!=tensor).nonzero()) == 0
        return tensor
            

def get_enc_dict(original_targets, targets):
    """Takes a list of encoded targets and original strings and makes a dictionary of the encoding
    Parameters
    ----------
    original_targets : str targets
    targets : encoded targets
    
    Returns
    -------
    dictionary with encoding mapping
    """
    targets = tuple(map(tuple, targets))

    # Initialize df
    df = pd.DataFrame({"targets":list(targets), "original_targets":original_targets})
    df = df.drop_duplicates(subset=["original_targets"])

    # Create dict 
    dict_encoding = defaultdict()
    dict_encoding = pd.Series(df.original_targets.values, index=df.targets.values).to_dict()

    return dict_encoding


def loss_ignore_nans(loss, x):
    """Takes loss values and multiplies losss by zero for values corresponding to a np.nan"""
    # Implement your code

    # print("loss.shape_unflattened", loss.shape)
    # print("x.shape", x.shape)

    #Flatten both tensors
    shape = loss.shape
    loss = loss.flatten()
    x = x.flatten()

    # print("loss.shape_flattened", loss.shape)
    # print("x.shape", x.shape)

    # print("loss[0:10]", loss[0:10])
    # print("x[0:10]", x[0:10])

    # Get indices of missing values
    if torch.cuda.is_available():
        idx_nan = (x.cpu()!=x.cpu()).nonzero().cuda()
    else :

        # creates an index in which it's TRUE in the positions of x that are NOT nonzero
        idx_nan = torch.isnan(x)
        # idx_nan = x!=x.nonzero()


    # Multiply the loss of those indices by 0

    # Create a tensor of ones with the shape of loss.shape
    ignore_tensor = torch.ones(loss.shape)
    # Put 0.0 where the index
    ignore_tensor[idx_nan] = 0.0
    if torch.cuda.is_available():
        ignore_tensor = ignore_tensor.cuda()
    loss = loss.mul(ignore_tensor)
    loss = torch.reshape(loss, shape)

    return loss





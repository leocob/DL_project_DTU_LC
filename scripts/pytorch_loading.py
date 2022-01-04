#!/usr/bin/env python
"""pytorch_loading.py: script that contains the pytorch tensors loading class """
__author__      = "Felix Pacheco"


import torch
from torch.utils.data import DataLoader
from preprocess_vae import one_hot_encoding
import numpy as np

class SNPLoading(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path ,data_files, targets):
        '''Initialization of the class
        Parameters
        ----------
        data_path  : path to the data files
        data_files : list of files to open
        targets    : targets already hot encoded
        '''
        self.targets = targets
        self.data_files = data_files
        self.data_path = data_path

  def __len__(self):
        'returns the total number of samples'
        return len(self.targets)

  def __getitem__(self, idx):
        'Return one sample of data with its label'
        # Load data and get label
        path = str(self.data_path)+str(self.data_files[idx])
        
        X = torch.load(path)
        #X = one_hot_encoding(np.array(torch.load(path).cpu()))
        #X = torch.FloatTensor(X)
        y = torch.FloatTensor(self.targets[idx]).view(-1)
        return X, y

# class SNPLoading(torch.utils.data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, data_path ,data_files, targets):
#         '''Initialization of the class
#         Parameters
#         ----------
#         data_path  : path to the data files
#         data_files : list of files to open
#         targets    : targets already hot encoded
#         '''
#         self.targets = targets
#         self.data_files = data_files
#         self.data_path = data_path

#   def __len__(self):
#         'returns the total number of samples'
#         return len(self.targets)


#   def __getitem__(self, idx):
#         'Return one sample of data with its label'
#         # Load data and get label
#         path = str(self.data_path)+str(self.data_files[idx])
        
#         X = torch.load(path)
#         y = self.targets[idx]
#         return X, y




if __name__ == "__main__":
    pass
# When script is imported
else:
    print("--> SNPLoading class imported succesfully")
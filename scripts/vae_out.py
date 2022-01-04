#!/usr/bin/env python
"""vae_out.py: Save / plot data from VAE model """
import pandas as pd
from preprocess_vae import impute_data
import torch


def de_encoding(enc_targets, dict_encoding):
    """Uses a dictionary to de-encode the target values using dict_encoding
    Parameters
    ----------
    enc_targets     : np.array of encoded target values
    dict_encoding   : dictionary containing encoding pairing
    Returns
    -------
    df              : list of non-encoded target values
    """
    # If there is only one target we save all the values as one string
    targets = []

    if len(enc_targets.shape) == 1:
        targets = [next(iter(dict_encoding.items()))[1]] * len(enc_targets)
        return targets

    # If there is more than one target then we de_code the targets
    enc_targets = tuple(map(tuple,enc_targets))
    for i in enc_targets:
        target = dict_encoding[i]
        targets.append(target)
    return targets


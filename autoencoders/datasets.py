#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:16:29 2020

@author: francesco
"""

import numpy as np
from autoencoders.utils import get_project_root


def load_cbf(verbose=True):
    folder = get_project_root() / "datasets" / "cached" / "cbf"
    X_train = np.load(folder / "X_train.npy")
    X_test = np.load(folder / "X_test.npy")
    y_train = np.load(folder / "y_train.npy")
    y_test = np.load(folder / "y_test.npy")

    if verbose:
        print("\nSHAPES:")
        print("TRAINING SET: ", X_train.shape)
        print("TEST SET: ", X_test.shape)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cbf()

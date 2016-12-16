# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix


def save_csr(out_file, array):
    np.savez(out_file, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_csr(in_file):
    loader = np.load(in_file)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

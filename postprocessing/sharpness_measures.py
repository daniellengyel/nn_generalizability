import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision

from sklearn.manifold import TSNE

from utils import *
from training import *
from data_getters import get_data

import yaml, os, sys, re

from data_getters import *

from pyhessian import hessian

import torch
from hessian_eigenthings import compute_hessian_eigenthings

import pickle


# TODO fix this to be clear that we are taking traces for all datapoints.
def get_traces(net, data, device=None):
    criterion = torch.nn.CrossEntropyLoss()
    traces = []
    if device is not None:
        is_gpu = True
    else:
        is_gpu = False
    for i in range(len(data[0])):
        inputs, labels = data[0][i], data[1][i]
        inputs, labels = inputs.view(1, *inputs.shape), labels.view(1, *labels.shape)

        if device is not None:
            inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                torch.cuda.LongTensor)

        traces.append(np.mean(hessian(net, criterion, data=(inputs, labels), cuda=is_gpu).trace()))
    return traces


# get eigenvalues of specific model folder.
def get_models_eig(models, train_loader, test_loader, loss, num_eigenthings=5, full_dataset=True, device=None, only_vals=True):
    eig_dict = {}
    # get eigenvals
    for k, m in models.items():
        print(k)
        if device is not  None:
            m = m.to(device)
            is_gpu = True
        else:
            is_gpu = False

        eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
                                                           loss, num_eigenthings, use_gpu=is_gpu,
                                                           full_dataset=full_dataset, mode="lanczos",
                                                           max_steps=100, tol=1e-2)
        try:
            #     eigenvals, eigenvecs = compute_hessian_eigenthings(m, train_loader,
            #                                                        loss, num_eigenthings, use_gpu=use_gpu, full_dataset=full_dataset , mode="lanczos",
            #                                                        max_steps=50)
            if only_vals:
                eig_dict[k] = eigenvals
            else:
                eig_dict[k] = (eigenvals, eigenvecs)
        except:
            print("Error for net {}.".format(k))

    return eig_dict



def get_models_trace(models, data_loader, criterion, full_dataset=False, verbose=False, device=None):
    trace_dict = {}

    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(data_loader):
        hessian_dataloader.append((inputs, labels))
        if not full_dataset:
            break

    # get trace
    for k, m in models.items():
        if verbose:
            print(k)
        a = time.time()
        ts = []

        if device is not  None:
            m = m.to(device)
            is_gpu = True
        else:
            is_gpu = False

        if full_dataset:
            trace = hessian(m, criterion, dataloader=hessian_dataloader, cuda=is_gpu).trace()
        else:
            trace = hessian(m, criterion, data=hessian_dataloader[0], cuda=is_gpu).trace()

        trace_dict[k] = trace

    return trace_dict

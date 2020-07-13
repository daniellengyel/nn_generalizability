import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision

from sklearn.manifold import TSNE

from .utils import *

import yaml, os, sys, re

import torch
from hessian_eigenthings import compute_hessian_eigenthings

import pickle


def get_models_loss_acc(models, train_loader, test_loader, device=None):
    loss_dict = {}
    acc_dict = {}

    for k, m in models.items():
        if device is not None:
            m = m.to(device)
        loss_dict[k] = (get_net_loss(m, train_loader, device=device), get_net_loss(m, test_loader, device=device))
        acc_dict[k] = (get_net_accuracy(m, train_loader, device=device), get_net_accuracy(m, test_loader, device=device))
    return loss_dict, acc_dict



def get_models_grad(models, data_loader, criterion, full_dataset=True):
    grad_dict = {}

    # get trace
    for k, m in models.items():
        weight_sum = 0
        for i, (inputs, labels) in enumerate(data_loader):
            print(k)
            # Compute gradients for input.
            inputs.requires_grad = True

            m.zero_grad()

            outputs = m(inputs)
            loss = criterion(outputs.float(), labels)
            loss.backward(retain_graph=True)

            param_grads = get_grad_params_vec(m)
            weight_sum += torch.norm(param_grads)

            if full_dataset:
                break

        grad_dict[k] = weight_sum / float(len(data_loader))

    return grad_dict


def get_models_tsne(models):
    models_vecs = np.array(
        [get_params_vec(m).detach().numpy() for k, m in sorted(models.items(), key=lambda item: int(item[0]))])
    X_embedded = TSNE(n_components=2).fit_transform(models_vecs)
    return X_embedded


def get_models_final_distances(beginning_models, final_models):
    dist_arr = []
    for i in range(len(beginning_models)):
        b_vec = get_params_vec(beginning_models[str(i)])
        f_vec = get_params_vec(final_models[str(i)])
        dist_arr.append(float(torch.norm(b_vec - f_vec)))

    return dist_arr

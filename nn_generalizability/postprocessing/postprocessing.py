import numpy as np
import pandas as pd

from .utils import *
from ..utils import *
from ..save_load import *

from .sharpness_measures import *
from .model_related import * 

import yaml, os, sys, re

import torch

import pickle

# TODO: Consider moving the runs/models getters to a util which does saving/loading exp related stuff.

# +++ process experiment results +++
def tb_to_dict(path_to_events_file, names):
    tb_dict = {}  # step, breakdown by / and _

    for e in summary_iterator(path_to_events_file):
        for v in e.summary.value:
            t_split = re.split('/+|_+', v.tag)
            if t_split[0] in names:
                tmp_dict = tb_dict
                t_split = [e.step] + t_split
                for i in range(len(t_split) - 1):
                    s = t_split[i]
                    if s not in tmp_dict:
                        tmp_dict[s] = {}
                        tmp_dict = tmp_dict[s]
                    else:
                        tmp_dict = tmp_dict[s]
                tmp_dict[t_split[-1]] = v.simple_value
    return tb_dict


# iterate through runs
def get_runs(experiment_folder, names):
    run_dir = {}
    for root, dirs, files in os.walk("{}/runs".format(experiment_folder), topdown=False):
        if len(files) != 2:
            continue
        run_file_name = files[0] if ("tfevents" in files[0]) else files[1]
        curr_dir = os.path.basename(root)
        print(root)
        try:
            run_dir[curr_dir] = tb_to_dict(os.path.join(root, run_file_name), names)
            cache_data(experiment_folder, "runs", run_dir)
        except:
            print("Error for this run.")

    return run_dir


def get_configs(experiment_folder):
    config_dir = {}
    for root, dirs, files in os.walk("{}/runs".format(experiment_folder), topdown=False):
        if len(files) != 2:
            continue
        curr_dir = os.path.basename(root)
        with open(os.path.join(root, "config.yml"), "rb") as f:
            config = yaml.load(f)
        config_dir[curr_dir] = config
        config_dir[curr_dir]["net_params"] = tuple(config_dir[curr_dir]["net_params"])
        if ("softmax_adaptive" in config_dir[curr_dir]) and (
        isinstance(config_dir[curr_dir]["softmax_adaptive"], list)):
            config_dir[curr_dir]["softmax_adaptive"] = tuple(config_dir[curr_dir]["softmax_adaptive"])

    return pd.DataFrame(config_dir).T


def get_exp_final_distances(experiment_folder, device=None):
    # init
    dist_dict = {}

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):

        beginning_models_dict = get_models(curr_path, 0, device)
        final_models_dict = get_models(curr_path, -1, device)
        dist_dict[exp_name] = get_models_final_distances(beginning_models_dict, final_models_dict)

        # cache data
        cache_data(experiment_folder, "dist", dist_dict)

    return dist_dict


def get_exp_tsne(experiment_folder, step):
    # init
    tsne_dict = {}

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        models_dict = get_models(curr_path, step)
        tsne_dict[exp_name] = get_models_tsne(models_dict)

        # cache data
        cache_data(experiment_folder, "tsne", tsne_dict)

    return tsne_dict

def get_exp_grad(experiment_folder, step, use_gpu=False, FCN=False):
    # init
    grad_dict = {}
    criterion = torch.nn.CrossEntropyLoss()

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, FCN)
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        models_dict = get_models(curr_path, step)
        grad_dict[exp_name] = _get_grad(models_dict, train_loader, criterion, full_dataset=False)

        # cache data
        cache_data(experiment_folder, "grad", grad_dict)

    return grad_dict


def get_exp_loss_acc(experiment_folder, step, FCN=False, device=None):
    print("Get loss acc")
    # init
    loss_dict = {}
    acc_dict = {}

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, FCN)
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        models_dict = get_models(curr_path, step)
        loss_dict[exp_name], acc_dict[exp_name] = get_models_loss_acc(models_dict, train_loader, test_loader,
                                                                      device=device)
        # cache data
        cache_data(experiment_folder, "loss", loss_dict)
        cache_data(experiment_folder, "acc", acc_dict)

    return loss_dict, acc_dict


# get eigenvalues of specific model folder.
def get_exp_eig(experiment_folder, step, num_eigenthings=5, FCN=False, device=None, only_vals=True):
    # init
    eigenvalue_dict = {}
    loss = torch.nn.CrossEntropyLoss()

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, FCN)
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):
        models_dict = get_models(curr_path, step)
        eigenvalue_dict[exp_name] = get_models_eig(models_dict, train_loader, test_loader, loss, num_eigenthings,
                                                   full_dataset=True, device=device, only_vals=only_vals)

        # cache data
        cache_data(experiment_folder, "eig", eigenvalue_dict)

    return eigenvalue_dict


def get_exp_trace(experiment_folder, step, seed=0, use_gpu=False, FCN=False, device=None):
    # init
    trace_dict = {}
    meta_dict = {"seed": seed}
    set_seed(seed)
    criterion = torch.nn.CrossEntropyLoss()

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, FCN)
    train_loader = DataLoader(train_data, batch_size=5000, shuffle=True)  # fix the batch size
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # iterate through models
    for exp_name, curr_path in exp_models_path_generator(experiment_folder):

        models_dict = get_models(curr_path, step)
        trace_dict[exp_name] = get_models_trace(models_dict, train_loader, criterion, full_dataset=False, verbose=True,
                                                device=device)

        # cache data
        cache_data(experiment_folder, "trace", trace_dict)

    return trace_dict


def get_stuff(experiment_folder):
    stuff = {}

    stuff_to_try = ["tsne", "runs", "trace", "acc", "dist", "loss", "grad", "eig"]

    for singular_stuff in stuff_to_try:
        print("Getting {}.".format(singular_stuff))
        try:
            with open(os.path.join(experiment_folder, "postprocessing/{}/data.pkl".format(singular_stuff)), "rb") as f:
                stuff[singular_stuff] = pickle.load(f)
        except:
            print("Error: {} could not be found".format(singular_stuff))

    stuff["configs"] = get_configs(experiment_folder)

    return stuff


def main(experiment_name):
    # # # save analysis processsing

    root_folder = os.environ["PATH_TO_GEN_FOLDER"]
    data_name = "CIFAR10"
    exp = "Jul13_16-26-18_Daniels-MacBook-Pro-4.local"
    experiment_folder = os.path.join(root_folder, "experiments", data_name, exp)

    # init torch
    is_gpu = False
    if is_gpu:
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None
        # device = torch.device("cpu")

    # get_runs(experiment_folder, ["Loss", "Kish", "Potential", "Accuracy", "WeightVarTrace", "Norm",
    #                          "Trace"])  # TODO does not find acc and var

    #
    # get_exp_final_distances(experiment_folder, device=device)

    # get_exp_eig(experiment_folder, -1, num_eigenthings=5, FCN=True, device=device)
    get_exp_trace(experiment_folder, -1, False, FCN=True, device=device)

    get_exp_loss_acc(experiment_folder, -1, FCN=True, device=device)

    # get_grad(experiment_folder, -1, False, FCN=True)

    # get_dirichlet_energy(experiment_folder, -1, num_steps=20, step_size=0.001, var_noise=0.5, alpha=1, seed=1, FCN=True)
    # get_exp_tsne(experiment_folder, -1)


if __name__ == "__main__":
    main("")
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Postprocess experiment.')
    # parser.add_argument('exp_name', metavar='exp_name', type=str,
    #                     help='name of experiment')
    #
    # args = parser.parse_args()
    #
    # print(args)
    #
    # experiment_name = args.exp_name

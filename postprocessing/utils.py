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

def get_postprocessing_data(experiment_folder, vectorized=True):
    data_type = experiment_folder.split("/")[-2]
    if data_type == "MNIST":
        return get_data("MNIST", vectorized, reduce_train_per=0.1)
    if data_type == "FashionMNIST":
        return get_data("FashionMNIST", vectorized)
    if data_type == "CIFAR10":
        return get_data("CIFAR10", vectorized, reduce_train_per=0.1)
    elif (data_type == "gaussian") or (data_type == "mis_gauss"):
        with open(os.path.join(experiment_folder, "data.pkl"), "rb") as f:
            data = pickle.load(f)
        return data
    else:
        raise NotImplementedError("{} data type is not implemented.".format(data_type))

def exp_models_path_generator(experiment_folder):
    for curr_dir in os.listdir("{}/models".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/models".format(experiment_folder), curr_dir)
        yield root

def exp_resampling_path_generator(experiment_folder):
    for curr_dir in os.listdir(os.path.join(experiment_folder, "resampling")):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join(experiment_folder, "resampling" , curr_dir)
        yield root


def different_cols(df):
    a = df.to_numpy()  # df.values (pandas<0.24)
    return (a[0] != a[1:]).any(0)


def get_hp(cfs):
    filter_cols = different_cols(cfs)
    hp_names = cfs.columns[filter_cols]
    hp_dict = {hp: cfs[hp].unique() for hp in hp_names}
    return hp_dict


def cache_data(experiment_folder, name, data, meta_dict=None):
    cache_folder = os.path.join(experiment_folder, "postprocessing", "name")
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    with open(os.path.join(cache_folder, "data.pkl"), "wb") as f:
        pickle.dump(data, f)

    if meta_dict is not None:
        with open(os.path.join(cache_folder, "meta.yml"), "wb") as f:
            yaml.dump(meta_dict, f)


def get_config_to_id_map(configs):
    map_dict = {}

    for net_id in configs:
        conf = configs[net_id]
        tmp_dict = map_dict
        for k, v in conf.items():
            if isinstance(v, list):
                v = tuple(v)

            if k not in tmp_dict:
                tmp_dict[k] = {}
            if v not in tmp_dict[k]:
                tmp_dict[k][v] = {}
            prev_dict = tmp_dict
            tmp_dict = tmp_dict[k][v]
        prev_dict[k][v] = net_id
    return map_dict


def get_ids_matching_config(config_to_id_map, config):
    if not isinstance(config_to_id_map, dict):
        return [config_to_id_map]
    p = list(config_to_id_map.keys())[0]

    ids = []
    for c in config_to_id_map[p]:
        if isinstance(config[p], list):
            config_compare = tuple(config[p])
        else:
            config_compare = config[p]
        if (config_compare is None) or (config_compare == c):
            ids += get_ids_matching_config(config_to_id_map[p][c], config)
    return ids


def get_all_model_steps(models_dir):
    step_dict = {}
    for root, dirs, files in os.walk(models_dir):
        for step_dir in dirs:
            name_split_underscore = step_dir.split("_")
            if len(name_split_underscore) == 1:
                continue
            step_dict[int(name_split_underscore[1])] = step_dir
    return step_dict


def get_models(model_folder_path, step, device=None):
    if step == -1:
        largest_step = -float("inf")
        for root, dirs, files in os.walk(model_folder_path):
            for sample_step_dir in dirs:
                name_split_underscore = sample_step_dir.split("_")
                if len(name_split_underscore) == 1:
                    continue
                largest_step = max(int(name_split_underscore[-1]), largest_step)

        step = largest_step

    model_path = os.path.join(model_folder_path, "step_{}".format(step))

    nets_dict = {}
    for root, dirs, files in os.walk(model_path):
        for net_file_name in files:
            net_idx = net_file_name.split("_")[1].split(".")[0]
            with open(os.path.join(root, net_file_name), "rb") as f:
                if device is None:
                    net = torch.load(f, map_location=torch.device('cpu'))
                else:
                    net = torch.load(f, map_location=device)
            nets_dict[net_idx] = net

    return nets_dict


def get_all_models(experiment_folder, step):
    models_dict = {}
    # iterate through models
    for curr_path in exp_models_path_generator(experiment_folder):
        try:
            models_dict[curr_dir] = get_models(curr_path, step)
        except:
            continue
    return models_dict

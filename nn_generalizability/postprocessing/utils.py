import numpy as np
import pandas as pd
import torch

from ..utils import *

import yaml, os, sys, re

from ..data_getters import *

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


def different_cols(df):
    a = df.to_numpy()  # df.values (pandas<0.24)
    return (a[0] != a[1:]).any(0)


def get_hp(cfs):
    filter_cols = different_cols(cfs)
    hp_names = cfs.columns[filter_cols]
    hp_dict = {hp: cfs[hp].unique() for hp in hp_names}
    return hp_dict


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



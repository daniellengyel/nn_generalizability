import numpy as np
import pandas as pd
import torch

from .utils import *
from .training_utils import get_nets

import yaml, os, sys, re, copy
from .nets.Nets import *

from .data_getters import *

import pickle

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

    models_dict = {}
    for root, dirs, files in os.walk(model_path):
        for model_file_name in files:
            model_idx = model_file_name.split("_")[1].split(".")[0]
            model = load_model(os.path.join(root, model_file_name), device)
            models_dict[model_idx] = model

    return models_dict


def get_all_models(experiment_folder, step):
    models_dict = {}
    # iterate through models
    for curr_path in exp_models_path_generator(experiment_folder):
        try:
            models_dict[curr_dir] = get_models(curr_path, step)
        except:
            continue
    return models_dict

def cache_data(experiment_folder, name, data, meta_dict=None):
    cache_folder = os.path.join(experiment_folder, "postprocessing", "name")
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    with open(os.path.join(cache_folder, "data.pkl"), "wb") as f:
        pickle.dump(data, f)

    if meta_dict is not None:
        with open(os.path.join(cache_folder, "meta.yml"), "wb") as f:
            yaml.dump(meta_dict, f)

def exp_models_path_generator(experiment_folder):
    for curr_dir in os.listdir("{}/models".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/models".format(experiment_folder), curr_dir)
        yield curr_dir, root

def exp_resampling_path_generator(experiment_folder):
    for curr_dir in os.listdir(os.path.join(experiment_folder, "resampling")):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join(experiment_folder, "resampling" , curr_dir)
        yield curr_dir, root

def save_sampling_idx(sampled_idx, experiment_root, curr_exp_name, step):
    resampling_idx_path = os.path.join(experiment_root, "resampling", curr_exp_name)
    if not os.path.exists(resampling_idx_path):
        os.makedirs(resampling_idx_path)

    with open(os.path.join(resampling_idx_path, "step_{}.pkl".format(step)), "wb") as f:
          pickle.dump(sampled_idx, f)

def save_models(models, model_name, model_params, experiment_root, curr_exp_name, step):
    models_path = os.path.join(experiment_root, "models", curr_exp_name, "step_{}".format(step))
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    for idx_model in range(len(models)):
         torch.save({'model_name': model_name,
                     'model_params': model_params,
                     'model_state_dict': models[idx_model].state_dict()}
                    , os.path.join(models_path, "model_{}.pt".format(idx_model)))

def load_model(PATH, device=None):
    meta_data = torch.load(PATH)
    model = get_nets(meta_data["model_name"], meta_data["model_params"], num_nets=1, device=device)[0]
    model.load_state_dict(meta_data['model_state_dict'])
    return model
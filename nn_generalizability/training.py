import numpy as np
import matplotlib.pyplot as plt
import copy, yaml, pickle
import torch
import torchvision
import torch.optim as optim

from .pyhessian import hessian

from torch.utils.tensorboard import SummaryWriter

from .utils import *
from .training_utils import *
from .GA import *
from .nets.Nets import SimpleNet, LeNet
from .save_load import *

from torch.utils.data import DataLoader

import os, time


def train(config, folder_path, train_data, test_data):
    # init torch
    # TODO should also be a method. Probably in utils
    if config["device"] == "gpu":
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda:0")
    else:
        device = None
        # device = torch.device("cpu")

    set_seed(config["seed"])

    # get dataloaders
    train_loader = DataLoader(train_data, batch_size=config["batch_train_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config["batch_test_size"], shuffle=True)

    # Init neural nets and weights
    nets = get_nets(config["net_name"], config["net_params"], config["num_nets"], device=device)

    num_nets = config["num_nets"]
    nets_weights = np.zeros(num_nets)

    #  Define a Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    get_opt_func = get_optimizers(config)
    optimizers = get_opt_func(nets)

    # Set algorithm params
    weight_type = config["weight_type"]
    beta = config["softmax_beta"]
    softmax_adaptive = config["softmax_adaptive"]
    sampling_wait = config["sampling_wait"]
    should_resample_checker = get_resampling_checker(config["sampling_tau"], config["sampling_wait"], config["sampling_stop"], config["ess_threshold"], beta)

    # should all be seperate methods. This actually probably in some training_utils.
    stopping_criterion = get_stopping_criterion(config["num_steps"], config["mean_loss_threshold"])


    # init saving
    file_stamp = str(time.time())  # get_file_stamp()
    writer = SummaryWriter("{}/runs/{}".format(folder_path, file_stamp))
    with open("{}/runs/{}/{}".format(folder_path, file_stamp, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    save_models(nets, config["net_name"], config["net_params"], folder_path, file_stamp, step=0)
    sampling_idx = np.array(list(range(0, num_nets)))
    save_sampling_idx(sampling_idx, folder_path, file_stamp, step=0)

    # init number of steps
    curr_step = 1
    mean_loss = float("inf")

    # train
    while (not stopping_criterion(mean_loss, curr_step)):

        # get train loaders for each net
        net_data_loaders = [iter(train_loader) for _ in range(num_nets)]

        is_training_curr = True
        while is_training_curr and (not stopping_criterion(mean_loss, curr_step)):
            if (curr_step % 100) == 1:
                print("Step: {}".format(curr_step))
                print("Mean Loss: {}".format(mean_loss))

            # do update step for each net
            if config["hard_train_eps"] is None:
                nets, nets_weights, steps_taken, mean_loss_after_step = training_step(nets, nets_weights, optimizers,
                                                                                 net_data_loaders, criterion,
                                                                                 weight_type, var_noise=config["var_noise"], curr_step=curr_step,
                                                                                 writer=writer, device=device)
            else:
                nets, nets_weights, steps_taken, mean_loss_after_step = _hard_training_step(nets, nets_weights, optimizers,
                                                                                 net_data_loaders, criterion,
                                                                                 weight_type, curr_step=curr_step,
                                                                                 writer=writer, device=device, hard_train_eps=config["hard_train_eps"])
            if steps_taken == 0:
                break
            else:
                mean_loss = mean_loss_after_step

            # # Get variation of network weights
            covs = get_params_cov(nets)
            writer.add_scalar('WeightVarTrace/', torch.norm(covs), curr_step)

            # # Check resample. TODO move this in the should resample or something.
            # if sampling_wait == curr_step:
            #     nets_weights = np.zeros(num_nets)

            # if should_resample_checker(nets_weights, curr_step):
            #     nets, nets_weights, optimizers, is_training_curr = do_resampling_TMP(nets, optimizers, get_opt_func, beta, nets_weights, softmax_adaptive, curr_step, folder_path, file_stamp, mean_loss)

            # update curr_step
            curr_step += steps_taken

        # # get test error
        # for idx_net in range(num_nets):
        #     accuracy = get_net_accuracy(nets[idx_net], test_loader, device=device)
        #     writer.add_scalar('Accuracy/net_{}'.format(idx_net), accuracy, curr_step)

    # save final nets
    save_models(nets, config["net_name"], config["net_params"], folder_path, file_stamp, step=curr_step)

    sampled_idx = np.array(list(range(0, num_nets)))
    save_sampling_idx(sampling_idx, folder_path, file_stamp, step=curr_step)

    return nets

def do_resampling_TMP(nets, optimizers, get_opt_func, beta, nets_weights, softmax_adaptive, curr_step, folder_path, file_stamp, mean_loss):

    # resample particles
    if softmax_adaptive is not None:
        offset, strength = softmax_adaptive
        strength *= np.mean(mean_loss) - offset  # TODO change so we can iterate over methods
        sampling_idx = sample_index_softmax(nets_weights, nets, beta=strength)
        sample_nets(nets, copy.deepcopy(sampling_idx)) # nets = [copy.deepcopy(nets[i]) for i in sampled_idx]
        optimizers = get_opt_func(nets, optimizers)

    elif beta != 0:
        sampling_idx = sample_index_softmax(nets_weights, nets, beta=beta)
        sample_nets(nets, copy.deepcopy(sampling_idx)) # nets = [copy.deepcopy(nets[i]) for i in sampled_idx]
        optimizers = get_opt_func(nets, optimizers)

    else:
        sampling_idx = list(range(len(nets)))

    nets_weights = np.zeros(len(nets))

    # save the resample indecies
    save_sampling_idx(sampling_idx, folder_path, file_stamp, step=curr_step)

    is_training_curr = False
    return nets, nets_weights, optimizers, is_training_curr


def training_step(nets, nets_weights, net_optimizers, net_data_loaders, criterion, weight_type, var_noise=None,
                   curr_step=0, writer=None, device=None):
    """Does update step on all networks and computes the weights.
    If wanting to do a random walk, set learning rate of net_optimizer to zero and set var_noise to noise level."""
    taking_step = True
    steps_taken = 0

    mean_loss = 0

    for idx_net in range(len(nets)):

        # get net and optimizer
        net = nets[idx_net]
        optimizer = net_optimizers[idx_net]

        # get the inputs; data is a list of [inputs, labels]
        try:
            data = next(net_data_loaders[idx_net])
        except:
            taking_step = False
            break
        inputs, labels = data

        if device is not None:
            inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                torch.cuda.LongTensor)

        # Compute gradients for input.
        inputs.requires_grad = True

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        if var_noise is not None:
            net = add_noise(net, var_noise, device)

        # update weights
        if weight_type == "input_output_forbenius":
            # zero the parameter
            optimizer.zero_grad()

            # get input gradients
            output_forb = torch.norm(outputs)
            output_forb.backward()
            input_grads = inputs.grad

            curr_weight = weight_function_input_jacobian(input_grads)
            nets_weights[idx_net] += curr_weight
        elif weight_type == "loss_gradient_weights":

            param_grads = get_grad_params_vec(net)
            curr_weight = torch.norm(param_grads)
            nets_weights[idx_net] += curr_weight
        elif weight_type is None:
            pass
        else:
            raise NotImplementedError()

        # store metrics for each net
        # TODO cahnge names to Gradient/train
        if writer is not None:
            writer.add_scalar('Loss/train/net_{}'.format(idx_net), loss, curr_step)
            writer.add_scalar('Gradient/train/net_{}'.format(idx_net), curr_weight, curr_step)
            writer.add_scalar('Norm/net_{}'.format(idx_net), torch.norm(get_params_vec(net)), curr_step)
            # if (curr_step % 50) == 0:
            #     # a = time.time()
            #     is_gpu = device is not None
            #     trace = np.mean(hessian(net, criterion, data=(inputs, labels), cuda=is_gpu).trace())
            #     writer.add_scalar('Trace/net_{}'.format(idx_net), trace, curr_step)
            #     # print("Getting trace took {}".format(time.time() - a))

        mean_loss += float(loss)

    assert taking_step or (idx_net == 0)

    return nets, nets_weights, 1*taking_step, mean_loss / len(nets)


def _hard_training_step(nets, nets_weights, net_optimizers, net_data_loaders, criterion, weight_type, var_noise=None,
                   curr_step=0, writer=None, device=None, hard_train_eps=None):
    """Does update step on all networks and computes the weights.
    If wanting to do a random walk, set learning rate of net_optimizer to zero and set var_noise to noise level."""
    taking_step = True
    steps_taken = 0

    mean_loss = 0
    curr_loss = float("inf")

    assert not (hard_train_eps and (len(nets) > 0))
    continue_training = True

    for idx_net in range(len(nets)):

        # get net and optimizer
        net = nets[idx_net]
        optimizer = net_optimizers[idx_net]

        # get the inputs; data is a list of [inputs, labels]
        try:
            data = next(net_data_loaders[idx_net])
        except:
            taking_step = False
            break
        inputs, labels = data
        if device is not None:
            inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                torch.cuda.LongTensor)

        while continue_training:

            # Compute gradients for input.
            inputs.requires_grad = True

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            if var_noise is not None:
                net = add_noise(net, var_noise, device)

            # update weights
            param_grads = get_grad_params_vec(net)
            curr_weight = torch.norm(param_grads)
            nets_weights[idx_net] += curr_weight


            # store metrics for each net
            if writer is not None:
                writer.add_scalar('Loss/train/net_{}'.format(idx_net), loss, curr_step)
                writer.add_scalar('Potential/curr/net_{}'.format(idx_net), curr_weight, curr_step)
                writer.add_scalar('Norm/net_{}'.format(idx_net), torch.norm(get_params_vec(net)), curr_step)
                if (curr_step % 50) == 0:
                    # a = time.time()
                    is_gpu = device is not None
                    trace = np.mean(hessian(net, criterion, data=(inputs, labels), cuda=is_gpu).trace())
                    writer.add_scalar('Trace/net_{}'.format(idx_net), trace, curr_step)
                    # print("Getting trace took {}".format(time.time() - a))

            mean_loss += float(loss)
            steps_taken += 1
            curr_step += 1
            continue_training = float(loss) > hard_train_eps

    assert taking_step or (idx_net == 0)

    return nets, nets_weights, steps_taken, mean_loss / steps_taken

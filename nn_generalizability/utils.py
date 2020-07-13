import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import datetime
import socket, sys, os, copy

from collections import defaultdict
from tensorflow.python.summary.summary_iterator import summary_iterator

from .nets.Nets import *

import random


def get_file_stamp():
    """Return time and hostname as string for saving files related to the current experiment"""
    host_name = socket.gethostname()
    mydate = datetime.datetime.now()
    return "{}_{}".format(mydate.strftime("%b%d_%H-%M-%S"), host_name)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# +++ algorithm +++

def sample_index_softmax(weights, positions, beta=1):
    probabilities = softmax(weights, beta)
    pos_filter = np.random.choice(list(range(len(positions))), len(positions), p=probabilities)
    return pos_filter


def softmax(weights, beta=-1):
    # # normalize weights:
    #
    # weight_normalization = np.sum(weights)
    # weight_normalization = weight_normalization if weight_normalization != 0 else 1
    # weights /= weight_normalization

    sum_exp_weights = sum([np.exp(beta * w) for w in weights])

    probabilities = np.array([np.exp(beta * w) for w in weights]) / sum_exp_weights
    return probabilities


def weight_function_input_jacobian(grad):
    input_shape = grad.shape  # batch, filters, x_dim, y_dim
    grad = grad.reshape((input_shape[0], np.product(input_shape[1:]))).T

    return np.sum(np.linalg.norm(grad, axis=0))


def kish_effs(weights):
    """Assume weights are just a list of numbers"""
    N = len(weights)
    weights = np.array(weights)
    sum_weights = np.sum(weights)

    norm = weights.dot(weights)
    if norm != 0:
        return 1 / float(N) * sum_weights ** 2 / weights.dot(weights)
    else:
        return 1

def get_params_vec(net):
    param_vec = torch.cat([p.view(-1) for p in net.parameters()])
    return param_vec


# def get_grad_params_vec(net):
#     list_params = list(net.parameters())
#     param_vec = [list_params[i].grad.view(list_params[i].nelement()).detach().numpy() for i in range(len(list_params))]
#     return np.concatenate(param_vec, 0)

def get_grad_params_vec(net):
    param_vec = torch.cat([p.grad.view(-1) for p in net.parameters()])
    return param_vec


def vec_to_net(vec, net):
    new_net = copy.deepcopy(net)

    dict_new_net = dict(new_net.named_parameters())

    start_point = 0
    for name1, param1 in net.named_parameters():
        end_point = start_point + param1.numel()
        dict_new_net[name1].data.copy_(vec[start_point:end_point].reshape(param1.shape))

        start_point = end_point

    return new_net

def torch_cov(m):
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    if x.size(1) > 1:
        C = (1 / (x.size(1) - 1))
    else:
        C = 1
    cov = C * x.mm(x.t())
    return cov


def torch_cov_trace(m):
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    if x.size(1) > 1:
        C = (1 / (x.size(1) - 1))
    else:
        C = 1
    return C * torch.stack([torch.norm(x[:, i]) for i in range(m.size(1))]).sum()


def get_params_cov(nets):
    """The variance of the weights of the neural networks.
    Output is nets.shape[1]xnets.shape[1]."""
    nets_param_vecs = torch.stack([get_params_vec(nets[net_idx]) for net_idx in range(len(nets))])
    cov = torch_cov_trace(nets_param_vecs.T)
    return cov


def get_correlation(X, Y):
    return (X - np.mean(X)).dot(Y - np.mean(Y)) / np.sqrt(np.var(X) * np.var(Y)) * 1 / float(len(Y))




# Viz

def classification_regions_2d(v1, v2, center_image, alpha_min, alpha_max, beta_min, beta_max, N, net):
    """
    Returns the alpha (X) and beta (Y) range used in the basis of v1 and v2. Use meshgrid(X, Y) to get the corresponding
    coordinates for the result. """

    alpha_range = np.linspace(alpha_min, alpha_max, N)
    beta_range = np.linspace(beta_min, beta_max, N)

    results = []

    mesh = np.array(np.meshgrid(alpha_range, beta_range))

    mesh_2d = mesh.reshape(2, N * N)

    max_batch = 250 * 250
    i = 1
    results = np.array([])

    net.eval()

    while N * N > max_batch * (i - 1):
        lin_comb = torch.stack([v1, v2]).T.mm(torch.Tensor(mesh_2d[:, (i - 1) * max_batch:i * max_batch])).T
        lin_comb = lin_comb.reshape(lin_comb.shape[0], 1, center_image.shape[1], center_image.shape[2])
        lin_comb += center_image
        curr_results = net(lin_comb)
        curr_results = torch.argmax(curr_results, 1).detach().numpy()

        results = np.concatenate([results, curr_results])
        i += 1

    mesh = np.array(np.meshgrid(alpha_range, beta_range))
    return alpha_range, beta_range, np.array(results).reshape(mesh.shape[1:]), v1, v2


# from: https://github.com/facebookresearch/jacobian_regularizer/blob/master/jacobian/jacobian.py
def _random_vector(C, B):
    '''
    creates a random vector of dimension C with a norm of C^(1/2)
    (as needed for the projection formula to work)
    '''
    if C == 1:
        return torch.ones(B)
    v = torch.randn(B, C)
    arxilirary_zero = torch.zeros(B, C)
    vnorm = torch.norm(v, 2, 1, True)
    v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
    return v


def get_vec(img):
    return img.reshape(np.product(img.shape[1:]))


def get_orthogonal_basis(img1, img2, img3):
    """img1 is center. img_2 is first orthogonal vector. use graham schmidt to get second vector from img_3."""
    img1_to_img2 = get_vec(img2) - get_vec(img1)
    unit_img1_to_img2 = img1_to_img2 / np.linalg.norm(img1_to_img2)

    img1_to_img3 = get_vec(img3) - get_vec(img1)
    unit_img1_to_img3 = img1_to_img3 / np.linalg.norm(img1_to_img3)

    assert abs(unit_img1_to_img2.dot(unit_img1_to_img3)) != 1

    # get orthogonal vectors which span the above subspace. Use grahamschmidt
    v1 = unit_img1_to_img2
    v2 = unit_img1_to_img3 - v1.dot(unit_img1_to_img3) * v1
    v2 = v2 / np.linalg.norm(v2)

    return v1, v2


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# net predictions
def get_average_output(nets, inp):
    outs = [net(inp).detach().numpy() for net in nets]
    return np.mean(outs, axis=0)


def get_net_accuracy(net, data_loader, full_dataset=False, device=None):
    correct = 0
    total = 0

    for idx, (inputs, labels) in enumerate(data_loader):
        if device is not None:
            inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                torch.cuda.LongTensor)
        else:
            inputs = inputs.float()

        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if not full_dataset:
            break
    return correct / float(total)


def get_net_loss(net, data_loader, full_dataset=False, device=None):
    criterion = torch.nn.CrossEntropyLoss()

    loss_sum = 0
    for idx, (inputs, labels) in enumerate(data_loader):
        if device is not None:
            inputs, labels = inputs.to(device).type(torch.cuda.FloatTensor), labels.to(device).type(
                torch.cuda.LongTensor)
        else:
            inputs = inputs.float()

        outputs = net(inputs)
        loss_sum += float(criterion(outputs, labels))
        if not full_dataset:
            break

    return loss_sum / (idx + 1)



def same_model(model1, model2):
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    params2_dict = dict(params2)
    for name1, param1 in params1:
        if name1 in params2_dict:
            if not torch.equal(params2_dict[name1].data, param1.data):
                return False
        else:
            return False

    return True

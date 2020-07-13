import os, copy
import torch


def _get_sample_idxs(model_folder_path):
    sample_idx_dir = {}
    for root, dirs, files in os.walk(model_folder_path, topdown=False):
        for sample_step_dir in dirs:
            name_split_underscore = sample_step_dir.split("_")
            if len(name_split_underscore) == 1:
                continue
            with open(os.path.join(model_folder_path, sample_step_dir, "sampled_idx.pkl"), "rb") as f:
                sample_idx_dir[name_split_underscore[-1]] = pickle.load(f)
    return sample_idx_dir


def get_sample_idxs(experiment_folder):
    # init
    sampled_idxs_dict = {}

    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        if "DS_Store" in curr_dir:
            continue
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        sampled_idxs_dict[curr_dir] = _get_sample_idxs(root)

    return sampled_idxs_dict


def _get_dirichlet_energy(nets, data, num_steps, step_size, var_noise, alpha=1, seed=1):
    """We use an OU process cetered at net.
    alpha is bias strength in OU."""
    # TODO add with noise that only comes from data_loader directions. i.e. same covariance as gradient RV.
    # TODO watch out with seed. we already are inputing nets and maybe other stuff?
    torch.manual_seed(seed)
    np.random.seed(seed)

    # init weights and save initial position.
    Xs_0 = [list(copy.deepcopy(n).parameters()) for n in nets]
    nets_weights = np.zeros(len(nets))
    criterion = torch.nn.CrossEntropyLoss()

    for i in range(num_steps):
        for idx_net in range(len(nets)):
            # get net and optimizer
            net = nets[idx_net]
            # Do OU step with EM discretization
            with torch.no_grad():
                for layer_idx, ps in enumerate(net.parameters()):
                    ps.data += torch.randn(ps.size()) * np.sqrt(var_noise * step_size)
                    ps.data += step_size * alpha * (Xs_0[idx_net][layer_idx].data - ps.data)
            nets_weights[idx_net] += unbiased_weight_estimate(net, data, criterion, num_samples=3, batch_size=500,
                                                              max_steps=3)  # max_steps and batch_size are from running some analysis. just checking
        print(i)
        # cache data
    return nets_weights


def get_dirichlet_energy(experiment_folder, model_step, num_steps=20, step_size=0.001, var_noise=0.5, alpha=1, seed=1,
                         FCN=False):
    # init
    energy_dict = {}

    # get data
    train_data, test_data = get_postprocessing_data(experiment_folder, FCN)

    # iterate through models
    for curr_dir in os.listdir("{}/resampling".format(experiment_folder)):
        root = os.path.join("{}/resampling".format(experiment_folder), curr_dir)
        print(curr_dir)
        models_dict = get_models(root, model_step)
        nets = [v for k, v in sorted(models_dict.items(), key=lambda item: int(item[0]))]

        energy_dict[curr_dir] = _get_dirichlet_energy(nets, train_data, num_steps, step_size, var_noise, alpha=1,
                                                      seed=1)

        # cache data
        cache_data(experiment_folder, "energy", energy_dict)

    return energy_dict

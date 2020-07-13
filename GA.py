def sample_nets(nets, idxs):
    """Memory efficient method to sample nets given a sample idx array. Mutates nets and idxs.
        Step 1: check if there are leaf nodes i.e. models which are not spawn particles. We can safely assign
                new weights to those. Notice, these particles have now also become spawn particles, so some permutation
                cycles are broken now.
        Step 2: If there is a permutation cycle left, we keep one temporary variable and fix the rest of the permutation
                cycle. This can only happen if none of the particles ever had a leaf node attached to them.
            """
    unique_s = set(idxs)

    # leaf routine
    replace_dict = {}
    saw_leaf = False
    we_done = True
    for p in range(len(idxs)):
        s = idxs[p]
        we_done = we_done and (p == s)
        if p not in unique_s:
            saw_leaf = True
            # mutate net
            nets[p].load_state_dict(nets[s].state_dict())  # nets[p] = nets[s]

            # change idxs to indicate we have updated this net
            idxs[p] = p
            if s not in replace_dict:
                replace_dict[s] = p

    if we_done:
        return

    if saw_leaf:
        for i in range(len(idxs)):
            if i == idxs[i]:
                continue
            if idxs[i] in replace_dict:
                idxs[i] = replace_dict[idxs[i]]
        sample_nets(nets, idxs)
    else:
        # cycle routine
        for i in range(len(idxs)):
            if (i == idxs[i]):
                continue
            else:
                tmp_net = copy.deepcopy(nets[i])  # tmp_net = nets[i]
                curr_p = i
                prev_p = idxs[i]
                while prev_p != i:
                    nets[curr_p].load_state_dict(nets[prev_p].state_dict())  # nets[curr_p] = nets[prev_p]

                    idxs[curr_p] = curr_p
                    curr_p = prev_p
                    prev_p = idxs[curr_p]
                nets[curr_p].load_state_dict(tmp_net.state_dict())  # nets[curr_p] = tmp_net
                idxs[curr_p] = curr_p

# exploring loss landscape
def unbiased_weight_estimate(net, data, criterion, num_samples=3, batch_size=500, max_steps=3):
    weights = []
    optimizer = optim.SGD(net.parameters(), lr=0,
                          momentum=0)
    iter_data = iter(DataLoader(data, batch_size=batch_size, shuffle=True))  # fix the batch size

    should_continue = True
    steps = 0
    while should_continue and (steps < max_steps):
        tmp_w_2 = 0
        curr_grad = None
        for _ in range(num_samples):
            try:
                inputs, labels = next(iter_data)
            except:
                should_continue = False
                break

            optimizer.zero_grad()

            # Compute gradients for input.
            inputs.requires_grad = True

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs.float(), labels)
            loss.backward(retain_graph=True)

            param_grads = get_grad_params_vec(net)
            if curr_grad is None:
                curr_grad = param_grads
            else:
                curr_grad += param_grads
            tmp_w_2 += torch.norm(param_grads) ** 2

        if should_continue:
            weights.append((torch.norm(curr_grad) ** 2 - tmp_w_2) / (num_samples * (num_samples - 1)))
            steps += 1
    return np.mean(weights)

from ..utils import *

import numpy as np

def get_assignments(curr_res_ids):
    assignments = {}
    for i, c in enumerate(curr_res_ids):
        if c not in assignments:
            assignments[c] = []
        assignments[c].append(i)
    return assignments


def find_lineages(res_ids, starting_lineage=None, starting_assignments=None):
    if res_ids is []:
        return None

    lineages_dict = starting_lineage
    prev_assignments = starting_assignments
    for t in range(len(res_ids)):
        curr_assignments = get_assignments(list(res_ids[t]))

        if lineages_dict is None:
            lineages_dict = {c: [c] for c in curr_assignments}
        else:
            new_linages_dict = {}
            for c in curr_assignments:
                for a, p_lin in lineages_dict.items():
                    if c in prev_assignments[a]:
                        new_linages_dict[c] = p_lin + [c]
                        break  # we know we found what we wanted
            lineages_dict = new_linages_dict
        prev_assignments = curr_assignments
    return lineages_dict, curr_assignments


def get_linages_vals(lineages, val_arr):
    arr_lineages = np.array(list(lineages.values())).T
    lin_vals = np.array(take_slice(val_arr, arr_lineages)).T
    return {k: lin_vals[i] for i, k in enumerate(lineages.keys())}






def sampling_plot_arr(values_arr, resampling_arr):
    x_vals = []
    y_vals = []

    for p in range(len(values_arr)):
        last_resampling = 0
        already_added = False

        for t in range(len(values_arr[p])):
            if resampling_arr[t][p] != p:
                if (last_resampling == 0) and (not already_added):
                    x_vals.append(list(range(last_resampling, t + 1)))
                    y_vals.append(values_arr[p][last_resampling:t + 1])
                    already_added = True
                else:
                    starts_at = values_arr[resampling_arr[last_resampling][p]][last_resampling]
                    x_vals.append(list(range(last_resampling, t + 1)))
                    y_vals.append([starts_at] + list(values_arr[p][last_resampling + 1:t + 1]))

                last_resampling = t

        if (last_resampling == 0):
            x_vals.append(list(range(last_resampling, t + 1)))
            y_vals.append(values_arr[p][last_resampling:t + 1])
        else:
            starts_at = values_arr[resampling_arr[last_resampling][p]][last_resampling]
            x_vals.append(list(range(last_resampling, t + 1)))
            y_vals.append([starts_at] + list(values_arr[p][last_resampling + 1:t + 1]))
    return x_vals, y_vals


def find_when_converged(a):
    for t, c in enumerate(a.T):
        if len(set(c)) > 1:
            print(c)
            print(t)
            break


if __name__ == "__main__":
    res_ids = [[0, 1, 2], [1, 1, 2], [0, 1, 2], [2, 1, 0], [1, 0, 1]]
    lineages, assignments = find_lineages(res_ids)

    val_arr = [[100, 200, 300], [1, 2, 3], [6, 2, 74], [7, 3, 67], [12, 4, 2]]

    lineages, assignments = find_lineages(res_ids)

    b = get_linages_vals(lineages, np.array(val_arr))

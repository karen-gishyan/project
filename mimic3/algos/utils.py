import torch
import itertools
import json
from typing import Callable
import matplotlib.pyplot as plt
import os
import numpy as np


class Evaluation:

    @classmethod
    def create_combinations(cls, path="json_files/training_combinations.json"):
        cls.path = path
        parameters = [
            ["Adam", "Adadelta"],
            [0.001, 0.01, 0.1],
            [0.9, 0.99],
            [50, 100],
            [10],
            [0.1, 0.2],
            [False]
        ]
        parameter_names = ["OPTIMIZER", "LEARNING_RATE", "DISCOUNT_FACTOR",
                           "MAX_TIME_STEP", "UPDATE_RATE", "EPSILON_GREEDY_RATE", "TRANSFER"]
        parameter_combinations = list(itertools.product(*parameters))
        param_list = []
        for p in parameter_combinations:
            param_list.append(dict(zip(parameter_names, p)))
        with open(path, 'w') as file:
            json.dump(param_list, file,indent=4)

    @classmethod
    def evaluate_results(cls, training_function: Callable,
                         save_path="json_files/evaluation.json"):
        with open(cls.path) as file:
            param_list = json.load(file)

        evaluation_list = []
        for i, p in enumerate(param_list, 1):
            for t in [1, 2, 3]:
                p.update({"TIME_PERIOD": t})
                rewards, network = training_function(**p)
                p.update({f"REWARD_{t}": max(rewards)})
                dir_path = f"results/{i}"
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

                # saving weights
                torch.save(network, f"{dir_path}/weights_t{t}.pt")
                # saving figure
                plt.clf()
                plt.plot(rewards)
                plt.ylim(-10000, 1000)
                plt.savefig(f"results/{i}/figure_t{t}.png")
            evaluation_list.append(p)
        with open(save_path, 'w') as file:
            json.dump(param_list, file,indent=4)


def compare_results():
    def compare_within_t(file_obj_1,file_obj_2):
        """ Compare results with and without transfer learning for each timestep."""

        res=json.load(file_obj_1)
        max_rewards=np.array([d['max_reward'] for d in res])
        nt_res=json.load(file_obj_2)
        nt_max_rewards=np.array([d['max_reward'] for d in nt_res])
        diff=np.subtract(max_rewards,nt_max_rewards)
        p_diff=diff[diff>0]
        np_diff=diff[diff<0]
        print("Maximum reward with transfer learning: {}".format(max(max_rewards)))
        print("Maximum reward without transfer learning: {}".format(max(nt_max_rewards)))
        print("Average reward with transfer learning: {}".format(np.mean(max_rewards)))
        print("Average reward without transfer learning: {}\n".format(np.mean(nt_max_rewards)))
        print("Number of times transfer learning has improved the results within each time-frame out of total: {}/{}".format(len(p_diff),len(diff)))
        print("Maximum improvement by transfer learning within time-stage: {}".format(max(p_diff)))
        print("Maximum impairment by transfer learning within time-stage: {} \n".format(min(np_diff)))

    def compare_across_t(file_obj_1,file_obj_2):
        """ Compare result across time-frames with transfer learning first, then without."""
        res_t=json.load(file_obj_1)
        res_t_next=json.load(file_obj_2)
        res_t_max_rewards=np.array([d['max_reward'] for d in res_t])
        res_t_next_max_rewards=np.array([d['max_reward'] for d in res_t_next])
        diff=np.subtract(res_t_max_rewards,res_t_next_max_rewards)
        p_diff=diff[diff>0]
        np_diff=diff[diff<0]
        print("Maximum improvement: {}".format(max(p_diff)))
        print("Average improvement: {}".format(np.mean(p_diff)))
        print("Maximum impairment: {} \n".format(min(np_diff)))

    print('Within time-stage.')
    with open("results_t2.json") as file1, open("no_transfer_results_t2.json") as file2:
        print('Stage 2 results with and without transfer.')
        compare_within_t(file1,file2)

    with open("results_t3.json") as file1, open("no_transfer_results_t3.json") as file2:
        print('Stage 3 results with and without transfer.')
        compare_within_t(file1,file2)

    print('Across time-stage.')
    with open("results_t1.json") as file1, open("results_t2.json") as file2:
        print('Stage 1-> 2 results with transfer.')
        compare_across_t(file1,file2)

    with open("no_transfer_results_t1.json") as file1, open("no_transfer_results_t2.json") as file2:
        print('Stage 1-> 2 results without transfer.')
        compare_across_t(file1,file2)

    with open("results_t2.json") as file1, open("results_t3.json") as file2:
        print('Stage 2-> 3 results with transfer.')
        compare_across_t(file1,file2)

    with open("no_transfer_results_t2.json") as file1, open("no_transfer_results_t3.json") as file2:
        print('Stage 2-> 3 results without transfer.')
        compare_across_t(file1,file2)

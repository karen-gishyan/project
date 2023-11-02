import itertools
import json
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from joblib import Parallel, delayed, parallel_backend
from networkx.algorithms import is_strongly_connected, number_strongly_connected_components, is_weakly_connected, \
    number_weakly_connected_components, is_semiconnected, strongly_connected_components, weakly_connected_components
from networkx.algorithms.bipartite import is_bipartite
from networkx.algorithms.cluster import average_clustering
from networkx.algorithms.distance_regular import is_distance_regular, is_strongly_regular
from networkx.algorithms.euler import is_eulerian, is_semieulerian, has_eulerian_path
from networkx.algorithms.isolate import number_of_isolates


class Evaluation:
    nx_functions = [
        is_bipartite,
        is_strongly_connected,
        number_strongly_connected_components,
        is_weakly_connected,
        number_weakly_connected_components,
        is_semiconnected,
        average_clustering,
        is_distance_regular,
        is_strongly_regular,
        is_eulerian,
        is_semieulerian,
        has_eulerian_path,
        number_of_isolates,
    ]

    def __init__(self, MimicEnv, Agent, **kwargs):
        self.Agent = Agent
        self.n_actions_per_state = kwargs.get('n_actions_per_state')
        self.env1 = MimicEnv(
            time_period=1, n_actions_per_state=self.n_actions_per_state)
        self.env2 = MimicEnv(
            time_period=2, n_actions_per_state=self.n_actions_per_state)
        self.env3 = MimicEnv(
            time_period=3, n_actions_per_state=self.n_actions_per_state)
        self.env_list = [self.env1, self.env2, self.env3]

    @classmethod
    def create_combinations(cls, path="json_files/training_combinations.json"):
        cls.combinations_path = path
        parameters = [
            ["torch.optim.Adam", "torch.optim.Adadelta"],
            [0.001, 0.01, 0.1],
            [0.9, 0.99],
            [25, 50],
            [10],
            [0.1, 0.2],
        ]
        parameter_names = ["OPTIMIZER", "LEARNING_RATE", "DISCOUNT_FACTOR",
                           "MAX_TIME_STEP", "UPDATE_RATE", "EPSILON_GREEDY_RATE"]
        parameter_combinations = list(itertools.product(*parameters))
        param_list = []
        for p in parameter_combinations:
            param_list.append(dict(zip(parameter_names, p)))

        with open(path, 'w') as file:
            json.dump(param_list, file, indent=4)

    def connect_graphs(self):
        """Add goal state of i periods graph to i+1.
        """
        for i in range(len(self.env_list)-1):
            goal_state = self.env_list[i].state_space.mdp.goal_state
            n_nodes = self.env_list[i +
                                    1].state_space.mdp.graph.number_of_nodes()
            label = n_nodes+1
            actual_label = f"{goal_state['label']}_t{i+1}"
            self.env_list[i+1].state_space.mdp.graph.add_node(label, label=label, features=goal_state['features'],
                                                              value=goal_state['value'], reward=0, actual_label=actual_label)
            self.env_list[i+1].start_state = self.env_list[i +
                                                           1].state_space.mdp.graph.nodes[label]

        return self

    def create_actions(self):
        self.env1.state_space.create_actions()
        self.env2.state_space.create_actions()
        self.env3.state_space.create_actions()
        return self

    def generate_connected_components(self):
        for env in self.env_list:
            env.sc_components = list(strongly_connected_components(
                env.state_space.mdp.graph))
            env.wc_components = list(weakly_connected_components(
                env.state_space.mdp.graph))
        return self

    def get_graph_statistics(self, path="json_files/graphs_stats.json"):
        graph_stats = defaultdict(dict)
        for i, env in enumerate(self.env_list):
            for function in self.nx_functions:
                graph_stats[f"t_{i}"].update(
                    {function.__name__: function(env.state_space.mdp.graph)})

        with open(path, 'w') as file:
            json.dump(graph_stats, file, indent=4)
        return self

    def train_models(self, save_path="json_files/evaluation.json"):
        with open(self.combinations_path) as file:
            self.param_list = json.load(file)
        with parallel_backend("loky"):
            results = Parallel(n_jobs=5, verbose=10)(delayed(self.subtrain)(i, p)
                                                     for i, p in enumerate(self.param_list, 1))
        with open(save_path, 'w') as file:
            json.dump(results, file, indent=4)

    def subtrain(self, i, p):
        p = defaultdict(lambda: defaultdict(dict), p)
        patient_outcomes = []
        n_patients = self.env1.state_space.mdp.graph.number_of_nodes()
        for patient_id in range(1, n_patients+1):
            policies = []
            outcome_per_t = []
            for t in [1, 2, 3]:
                optimizer = p.get("OPTIMIZER")
                lr = p.get("LEARNING_RATE")
                discount_factor = p.get("DISCOUNT_FACTOR")
                max_time_step = p.get("MAX_TIME_STEP")
                update_rate = p.get("UPDATE_RATE")
                epsilon_greedy = p.get("EPSILON_GREEDY_RATE")
                time_period = t

                env = getattr(self, f'env{t}')
                if t == 1:
                    env.start_state = env.state_space.mdp.graph.nodes[patient_id]

                agent = self.Agent(state_dim=10, env=env,
                                   optimizer=optimizer, lr=lr, discount_factor=discount_factor)
                rewards, network, policy_graph = agent.train(
                    time_period, max_time_step, epsilon_greedy, update_rate)

                p[f"patient_id_{patient_id}"][f"t_{t}"].update(
                    {"MAX_REWARD": max(rewards)})
                p[f"patient_id_{patient_id}"][f"t_{t}"].update(
                    {"SOLUTION": policy_graph.graph['solution']})

                p[f"patient_id_{patient_id}"][f"t_{t}"].update(
                    {"EPISODE_STEPS": policy_graph.graph['episode_steps']})
                p[f"patient_id_{patient_id}"][f"t_{t}"].update(
                    {"EPISODE_TERMINALS": policy_graph.graph['episode_terminals']})
                p[f"patient_id_{patient_id}"][f"t_{t}"].update(
                    {"EPISODE_REWARDS": policy_graph.graph['episode_rewards']})

                start_state_is_part_of_scc = any(
                    env.start_state['label'] in c for c in env.sc_components)
                start_state_is_part_of_wcc = any(
                    env.start_state['label'] in c for c in env.wc_components)
                p[f"patient_id_{patient_id}"][f"t_{t}"].update(
                    {"START_STATE_IS_PART_OF_STRONGLY_CONNECTED": start_state_is_part_of_scc})
                p[f"patient_id_{patient_id}"][f"t_{t}"].update(
                    {"START_STATE_IS_PART_OF_WEAKLY_CONNECTED": start_state_is_part_of_wcc})

                if self.n_actions_per_state:
                    dir_path = f"results_for_{self.n_actions_per_state}_actions/{i}/patient_id{patient_id}"
                else:
                    dir_path = f"results/{i}/patient_id{patient_id}"
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

                policies.append(policy_graph)

                # saving weights
                # joblib.dump(network, f"{dir_path}/weights_t{t}.pt")
                # saving figure
                # plt.clf()
                # plt.plot(rewards)
                # plt.ylim(-5000, 1000)
                # plt.savefig(f"{dir_path}/figure_t{t}.png")
                outcome_per_t.append(policy_graph.graph['solution'])

            final_result = all(outcome_per_t)
            patient_outcomes.append(final_result)
            p[f"patient_id_{patient_id}"].update({
                f"HAS_FINAL_SOLUTION": final_result})

        results = self.evaluate_results(
            patient_outcomes, self.env3.state_space.mdp.model3.output)
        p.update({"RESULTS": results})
        print(f"{i}th combination completed.\n")
        return p

    def train_rl_classification(self, save_path="json_files/classification_results.json"):
        with open(self.combinations_path) as file:
            self.param_list = json.load(file)
        with parallel_backend("loky"):
            results = Parallel(n_jobs=5, verbose=10)(delayed(self.subtrain_rl_classification)(i, p)
                                                     for i, p in enumerate(self.param_list, 1))
        with open(save_path, 'w') as file:
            json.dump(results, file, indent=4)

    def subtrain_rl_classification(self, i, p):
        p = defaultdict(lambda: defaultdict(dict), p)
        patient_outcomes = []
        n_patients = self.env1.state_space.mdp.graph.number_of_nodes()
        for patient_id in range(1, n_patients+1):
            policies = []
            outcome_per_t = []
            class_output = self.env3.state_space.mdp.model3.output[patient_id-1].item(
            )
            for t in [1, 2, 3]:
                optimizer = p.get("OPTIMIZER")
                lr = p.get("LEARNING_RATE")
                discount_factor = p.get("DISCOUNT_FACTOR")
                max_time_step = p.get("MAX_TIME_STEP")
                update_rate = p.get("UPDATE_RATE")
                epsilon_greedy = p.get("EPSILON_GREEDY_RATE")
                time_period = t

                env = getattr(self, f'env{t}')
                if t == 1:
                    env.start_state = env.state_space.mdp.graph.nodes[patient_id]

                agent = self.Agent(state_dim=10, env=env,
                                   optimizer=optimizer, lr=lr, discount_factor=discount_factor)
                rewards, network, policy_graph = agent.train(
                    time_period, max_time_step, epsilon_greedy, update_rate, class_output=class_output)

                p[f"patient_id_{patient_id}"][f"t_{t}"].update(
                    {"SOLUTION": policy_graph.graph['solution']})

                policies.append(policy_graph)
                plt.clf()
                plt.plot(rewards)
                plt.ylim(-5000, 1000)
                outcome_per_t.append(policy_graph.graph['solution'])

            final_result = all(outcome_per_t)
            patient_outcomes.append(final_result)
            p[f"patient_id_{patient_id}"].update({
                f"HAS_FINAL_SOLUTION": final_result})

        results = self.evaluate_results(
            patient_outcomes, self.env3.state_space.mdp.model3.output)
        p.update({"PRED_OUTCOMES": str(patient_outcomes)})
        p.update({"OUTCOMES": str(self.env3.state_space.mdp.model3.output)})
        p.update({"RESULTS": results})

        with open("json_files/classification_results.json", 'w') as file:
            json.dump(p, file, indent=4)
        print(f"{i}th combination completed.\n")
        return p

    @classmethod
    def evaluate_results(cls, target, actual):
        return precision_recall_fscore_support(target, actual[:len(target)], average='binary')

    @classmethod
    def summary_statistics(cls):
        """Summary results from training.
        """
        with open("json_files/evaluation.json") as file:
            evalualtions = json.load(file)

        summary = []
        for i, method in enumerate(evalualtions, 1):
            summary_results = {}
            t1_rewards, t2_rewards, t3_rewards = [], [], []
            t1_solutions, t2_solutions, t3_solutions = [], [], []
            final_solutions = []

            for _, value in method.items():
                if isinstance(value, dict):
                    t1_rewards.append(value['t_1']['MAX_REWARD'])
                    t2_rewards.append(value['t_2']['MAX_REWARD'])
                    t3_rewards.append(value['t_3']['MAX_REWARD'])

                    t1_solutions.append(value['t_1']['SOLUTION'])
                    t2_solutions.append(value['t_2']['SOLUTION'])
                    t3_solutions.append(value['t_3']['SOLUTION'])
                    final_solutions.append(value['HAS_FINAL_SOLUTION'])

            summary_results['id'] = i
            summary_results['t1_average_reward'] = sum(
                t1_rewards)/len(t1_rewards)
            summary_results['t2_average_reward'] = sum(
                t2_rewards)/len(t2_rewards)
            summary_results['t3_average_reward'] = sum(
                t3_rewards)/len(t3_rewards)

            summary_results['t1_number_of_solutions'] = sum(t1_solutions)
            summary_results['t2_number_of_solutions'] = sum(t2_solutions)
            summary_results['t3_number_of_solutions'] = sum(t3_solutions)
            summary_results['number_of_final_solutions'] = sum(final_solutions)
            summary_results['results'] = method['RESULTS']

            summary.append(summary_results)

        with open("json_files/summary.json", 'w') as file:
            json.dump(summary, file, indent=4)

    @classmethod
    def visualize_summary_statistics(cls):
        with open("json_files/summary.json") as file:
            summary = json.load(file)

        vis_summary_dict = defaultdict(list)
        for dict_ in summary:
            vis_summary_dict['t_1'].append(dict_['t1_average_reward'])
            vis_summary_dict['t_2'].append(dict_['t2_average_reward'])
            vis_summary_dict['t_3'].append(dict_['t3_average_reward'])
            vis_summary_dict['reward_sum'].append(dict_['t1_average_reward'] +
                                                  dict_['t2_average_reward'] +
                                                  dict_['t3_average_reward'])
            vis_summary_dict['solutions_count'].append(
                dict_['number_of_final_solutions'])

        # stacked chart for rewards
        x_axis = list(range(1, len(summary)+1))
        width = 0.5
        _, ax = plt.subplots()
        bottom = np.zeros(len(summary))

        for label, average_max_reward in vis_summary_dict.items():
            if label in ['solutions_count', 'reward_sum']:
                continue
            p = ax.bar(x_axis, average_max_reward,
                       width, label=label, bottom=bottom)
            bottom += average_max_reward

        y_offset = 30
        for method, sum_ in enumerate(vis_summary_dict['reward_sum'], 1):
            ax.text(method, sum_ + y_offset, round(sum_),
                    ha='center', rotation='vertical')

        # increase ylim so as labels fit
        ax.set_xticks(np.array(x_axis) + width, x_axis)
        ax.set_xlabel('Methods')
        ax.set_ylabel('Reward')
        ax.set_ylim(top=max(vis_summary_dict['reward_sum'])+200)
        ax.set_title(
            "Total reward per method grouped by timestep and averaged over patients")
        # move legent outside of box
        ax.legend(bbox_to_anchor=(1.1, 1), loc="upper right")
        plt.show()

        # Number of solutions per method
        _, ax = plt.subplots()
        ax.bar(x_axis, vis_summary_dict['solutions_count'])
        for method, count in enumerate(vis_summary_dict['solutions_count'], 1):
            ax.text(method, count + 0.5, round(count), ha='center')

        ax.set_xticks(np.array(x_axis) + width, x_axis)
        ax.set_xlabel('Methods')
        ax.set_ylabel('Count')
        ax.set_ylim(top=max(vis_summary_dict['solutions_count'])+5)
        ax.set_title("Number of solutions per method averaged over patients")
        plt.show()

    @classmethod
    def visualize_summary_statistics_v2(cls):
        with open("json_files/evaluation.json") as file:
            summary = json.load(file)

        steps_terminals_average_correlations = []
        steps_rewards_average_correlations = []
        terminals_rewards_average_correlations = []
        for method in summary:
            patient_keys = [key for key in method if 'patient' in key]
            t1, t2, t3 = [], [], []
            for patient_id in patient_keys:
                patient_data = method[patient_id]
                timestep_keys = [key for key in patient_data if "t" in key]
                for t in timestep_keys:
                    steps = patient_data[t]['EPISODE_STEPS']
                    terminals = patient_data[t]['EPISODE_TERMINALS']
                    rewards = patient_data[t]['EPISODE_REWARDS']
                    steps_terminals = np.corrcoef(
                        steps, terminals).flatten()[1]
                    steps_rewards = np.corrcoef(steps, rewards).flatten()[1]
                    terminals_rewards = np.corrcoef(
                        terminals, rewards).flatten()[1]
                    if np.isnan(steps_terminals):
                        steps_terminals = 0
                    if np.isnan(steps_rewards):
                        steps_rewards = 0
                    if np.isnan(terminals_rewards):
                        terminals_rewards = 0

                    if "1" in t:
                        t1.append(
                            [steps_terminals, steps_rewards, terminals_rewards])
                    elif "2" in t:
                        t2.append(
                            [steps_terminals, steps_rewards, terminals_rewards])
                    else:
                        t3.append(
                            [steps_terminals, steps_rewards, terminals_rewards])

            t1_steps_terminals, t1_steps_rewards, t1_terminals_rewards = np.array(
                t1).mean(axis=0)
            t2_steps_terminals, t2_steps_rewards, t2_terminals_rewards = np.array(
                t2).mean(axis=0)
            t3_steps_terminals, t3_steps_rewards, t3_terminals_rewards = np.array(
                t3).mean(axis=0)
            steps_terminals_average_correlations.append(
                [t1_steps_terminals, t2_steps_terminals, t3_steps_terminals])
            steps_rewards_average_correlations.append(
                [t1_steps_rewards, t2_steps_rewards, t3_steps_rewards])
            terminals_rewards_average_correlations.append(
                [t1_terminals_rewards, t2_terminals_rewards, t3_terminals_rewards])

        _, ax = plt.subplots(layout='constrained')
        width = 0.25  # the width of the bars
        multiplier = 0
        steps_terminals_average_correlations = np.array(
            steps_terminals_average_correlations).reshape((-1, len(summary)))
        x = np.arange(1, steps_terminals_average_correlations.shape[1]+1)
        for i, values in enumerate(steps_terminals_average_correlations, 1):
            offset = width * multiplier
            ax.bar(x + offset, values, width, label=f"t_{i}")
            multiplier += 1

        ax.set_xticks(x + width, x)
        ax.set_xlabel('Methods')
        ax.set_ylabel('Correlation')
        ax.set_title(
            'Average correlation per method for each timestep: number of steps taken and final solutions')
        ax.legend(bbox_to_anchor=(1.08, 1), loc="upper right")
        plt.show()

        _, ax = plt.subplots(layout='constrained')
        width = 0.25  # the width of the bars
        multiplier = 0
        steps_rewards_average_correlations = np.array(
            steps_rewards_average_correlations).reshape((-1, len(summary)))
        x = np.arange(1, steps_rewards_average_correlations.shape[1]+1)
        for i, values in enumerate(steps_rewards_average_correlations, 1):
            offset = width * multiplier
            ax.bar(x + offset, values, width, label=f"t_{i}")
            multiplier += 1

        ax.set_xticks(x + width, x)
        ax.set_xlabel('Methods')
        ax.set_ylabel('Correlation')
        ax.set_title(
            'Average correlation per method for each timestep: number of steps taken and accumulated reward')
        ax.legend(bbox_to_anchor=(1.08, 1), loc="upper right")
        plt.show()

        _, ax = plt.subplots(layout='constrained')
        width = 0.25  # the width of the bars
        multiplier = 0
        terminals_rewards_average_correlations = np.array(
            terminals_rewards_average_correlations).reshape((-1, len(summary)))
        x = np.arange(1, terminals_rewards_average_correlations.shape[1]+1)
        for i, values in enumerate(terminals_rewards_average_correlations):
            offset = width * multiplier
            ax.bar(x + offset, values, width, label=f"t_{i}")
            multiplier += 1

        ax.set_xticks(x + width, x)
        ax.set_xlabel('Methods')
        ax.set_ylabel('Correlation')
        ax.set_title(
            'Average correlation per method for each timestep: final solutions and accumulated reward')
        ax.legend(bbox_to_anchor=(1.08, 1), loc="upper right")
        plt.show()

    @classmethod
    def visualize_summary_statistics_classification(cls):
        with open("json_files/classification_results.json") as file:
            summary = json.load(file)

        x_axis = list(range(1, len(summary)+1))
        y_axis = []
        for result in summary:
            # append f scores
            y_axis.append(result['RESULTS'][2])

        _, ax = plt.subplots()
        ax.bar(x_axis, y_axis)
        for method, f_score in enumerate(y_axis, 1):
            ax.text(method, f_score+0.005,
                    f"{round(f_score,2)}", ha='center')

        ax.set_title("F-score per method")
        ax.set_ylabel('F-score')
        ax.set_xlabel('Methods')
        plt.show()

    @classmethod
    def plot_rewards(cls):
        """
        Plot of a sample patient and a sample timestep.
        """
        with open("json_files/evaluation.json") as file:
            summary = json.load(file)

        _, ax = plt.subplots()
        # first eight demonstrate the highest solutions
        for method in summary[:8]:
            y_axis = np.diff(method["patient_id_1"]["t_1"]["EPISODE_REWARDS"])
            x_axis = range(1, len(y_axis)+1)
            ax.plot(x_axis, y_axis)

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Reward')
        ax.set_title("First difference of rewards per epoch")
        plt.show()

    def __call__(self):
        self.create_combinations()
        self.connect_graphs().create_actions()
        self.get_graph_statistics()
        self.generate_connected_components().train_models()

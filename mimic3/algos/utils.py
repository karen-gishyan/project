import itertools
import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from joblib import Parallel, delayed, parallel_backend

class Evaluation:

    @classmethod
    def create_combinations(cls, path="json_files/training_combinations.json"):
        cls.path = path
        parameters = [
            ["torch.optim.Adam", "torch.optim.Adadelta"],
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
            json.dump(param_list, file, indent=4)

    @classmethod
    def train_models(cls, mdp, Env, Agent,
                     save_path="json_files/evaluation.json"):
        with open(cls.path) as file:
            cls.param_list = json.load(file)
        mdp.make_models()
        #NOTE class attributes assigned did not work in subtrain function
        with parallel_backend("loky"):
            results = Parallel(n_jobs=5,verbose=10)(delayed(cls.subtrain)(i, p,mdp=mdp,save_path=save_path,
                                                                          Env=Env,Agent=Agent)
                                for i, p in enumerate(cls.param_list, 1))
        with open(save_path, 'w') as file:
            json.dump(results, file, indent=4)

    @classmethod
    def subtrain(cls,i, p,**kwargs):
        mdp=kwargs.get('mdp')
        Env=kwargs.get('Env')
        Agent=kwargs.get('Agent')

        p = defaultdict(dict, p)
        patient_outcomes = []
        for patient_id, _ in enumerate(mdp.model1.feature_tensors, 1):
            start_state = patient_id
            policies = []
            outcome_per_t = []
            for t in [1, 2, 3]:
                optimizer = p.get("OPTIMIZER")
                lr = p.get("LEARNING_RATE")
                discount_factor = p.get("DISCOUNT_FACTOR")
                max_time_step = p.get("MAX_TIME_STEP")
                update_rate = p.get("UPDATE_RATE")
                epsilon_greedy = p.get("EPSILON_GREEDY_RATE")
                transfer = p.get("TRANSFER")
                time_period = t
                start_state = start_state

                env = Env(time_period, start_state)
                agent = Agent(state_dim=10, env=env, transfer=transfer, double_optimization=False,
                                  optimizer=optimizer, lr=lr, discount_factor=discount_factor)
                rewards, network, policy_graph, goal_state = agent.train(
                    time_period, max_time_step, epsilon_greedy, update_rate)

                p[f"patient_id_{patient_id}"].update(
                    {f"REWARD_{t}": max(rewards)})
                # p.update({f"REWARD_{t}": max(rewards)})
                dir_path = f"results/{i}/patient_id{patient_id}"
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)
                start_state = goal_state
                policies.append(policy_graph)

                # saving weights
                # joblib.dump(network, f"{dir_path}/weights_t{t}.pt")
                # saving figure
                plt.clf()
                plt.plot(rewards)
                plt.ylim(-10000, 1000)
                plt.savefig(f"{dir_path}/figure_t{t}.png")
                outcome_per_t.append(policy_graph.graph['solution'])

            patient_outcomes.append(all(outcome_per_t))
            print(f"{patient_id}_patient_id completed.")

        results = cls.evaluate_results(patient_outcomes, mdp.model3.output)
        p.update({"RESULTS": results})
        print(f"{i}th completed.\n")
        return p

    @classmethod
    def evaluate_results(cls, target, actual):
        return precision_recall_fscore_support(target, actual[:len(target)], average='binary')

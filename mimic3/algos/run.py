from utils import Evaluation
from dqn import set_seed, MimicEnv, Agent, MimicEnvClassification
from helpers import configure_logger
import os

if __name__ == "__main__":
    logger = configure_logger(default=False, path=os.path.dirname(__file__))
    set_seed()
    # evaluation = Evaluation(MimicEnv, Agent)()
    # Evaluation.summary_statistics()
    Evaluation.visualize_summary_statistics()

    ### classification ###
    # Evaluation.create_combinations()
    # Evaluation(MimicEnvClassification, Agent).connect_graphs().create_actions().train_rl_classification()
    Evaluation.visualize_summary_statistics_classification()
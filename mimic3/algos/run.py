from utils import Evaluation
from dqn import set_seed, MimicEnv, Agent
from helpers import configure_logger
import os

if __name__ == "__main__":
    logger = configure_logger(default=False, path=os.path.dirname(__file__))
    set_seed()
    # evaluation = Evaluation(MimicEnv, Agent,n_actions_per_state=17)()
    # Evaluation.summary_statistics()
    Evaluation.visualize_summary_statistics()
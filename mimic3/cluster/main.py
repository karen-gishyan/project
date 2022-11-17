import os
import yaml
from sklearn.metrics import mean_squared_error, median_absolute_error
from model import DistanceModel

dir_=os.path.dirname(__file__)
os.chdir(dir_)
print(os.getcwd())


def calculate_distances():
    """
    Calculate distances using the metrics for each diagnosis and each timestep.
    """
    with open('../datasets/sqldata/stats.yaml') as stats, open('info.yaml') as info:
        stats=yaml.safe_load(stats)
        info=yaml.safe_load(info)

    diagnoses=stats['diagnosis_for_selection']
    similarity_functions=info['similarity_functions']
    timesteps=info['timesteps']
    for diagnosis in diagnoses:
        for f in similarity_functions:
            for t in timesteps:
                DistanceModel(diagnosis=diagnosis,timestep=t)(similarity_function=eval(f))

calculate_distances()


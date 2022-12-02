import os
import yaml
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.cluster import KMeans
from model import DistanceModel,ClusteringModel, DistributionModel

dir_=os.path.dirname(__file__)
os.chdir(dir_)


def calculate_distances():
    """
    Calculate distances  for each diagnosis and each timestep using the predefined metrics.
    """

    with open('../datasets/sqldata/stats.yaml') as stats, open('info.yaml') as info:
        stats=yaml.safe_load(stats)
        info=yaml.safe_load(info)

    diagnoses=stats['diagnosis_for_selection']
    similarity_functions=info['similarity_functions']
    clustering_functions=info['clustering_functions']
    timesteps=info['timesteps']
    for diagnosis in diagnoses:
        for f in similarity_functions:
            for t in timesteps:
                DistanceModel(diagnosis=diagnosis,timestep=t)(similarity_function=eval(f))

        # for c in clustering_functions:
        #     for t in timesteps:
        #         ClusteringModel(diagnosis=diagnosis,timestep=t)(clustering_function=eval(c))

        for t in timesteps:
            DistributionModel(diagnosis=diagnosis, timestep=t)()


calculate_distances()
import os
import yaml
# metrics, cluster imports are not unused and are used as arguments for DistanceModel, ClusteringModel.
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.cluster import KMeans, MiniBatchKMeans
from model import DistanceModel,ClusteringModel, DistributionModel
from utils import combine_drug_sequences, train_individual

dir_=os.path.dirname(__file__)
os.chdir(dir_)

with open('../datasets/sqldata/stats.yaml') as stats, open('info.yaml') as info:
        stats=yaml.safe_load(stats)
        info=yaml.safe_load(info)

diagnoses=stats['diagnosis_for_selection']
similarity_functions=info['similarity_functions']
clustering_functions=info['clustering_functions']
timesteps=info['timesteps']

def calculate_distances():
    """
    Calculate distances for each diagnosis and each timestep using the predefined metrics.
    """
    for diagnosis in diagnoses:
        for f in similarity_functions:
            for t in timesteps:
                DistanceModel(diagnosis=diagnosis,timestep=t)(similarity_function=eval(f))

        for c in clustering_functions:
            for t in timesteps:
                ClusteringModel(diagnosis=diagnosis,timestep=t)(clustering_function=eval(c))

        for t in timesteps:
            DistributionModel(diagnosis=diagnosis, timestep=t)()

def combine_store_drug_sequences():
    #NOTE functionality is performed for each diagnoses, but training is done only for 1 of them.
    for diagnosis in diagnoses:
        for dir_name in ['distance','cluster','kstest']:
            if dir_name=='distance':
                for f in similarity_functions:
                    combine_drug_sequences(diagnosis,dir_name,f)
            elif dir_name=='cluster':
                for f in clustering_functions:
                    combine_drug_sequences(diagnosis,dir_name,f)
            else:
                combine_drug_sequences(diagnosis,dir_name)


def train():
    #NOTE ['CONGESTIVE HEART FAILURE','SEPSIS','ALTERED MENTAL STATUS' have only 0s in generated output]
    # for 'DIABETIC KETOACIDOSIS', drastic accuracies between folds (maybe due to data), not reported.
    for dir_name in ['distance','cluster','kstest']:
        if dir_name=='distance':
            for f in similarity_functions:
                train_individual('PNEUMONIA',dir_name,f)
        elif dir_name=='cluster':
            for f in clustering_functions:
                train_individual('PNEUMONIA',dir_name,f)
        else:
            train_individual('PNEUMONIA',dir_name)



"""
plots configs in train() may be affected by the configs in calculate_distances(), better to run seperately
"""
# calculate_distances()
# combine_store_drug_sequences()
train()

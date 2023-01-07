import os
import yaml
from model import Graph

dir_=os.path.dirname(__file__)
os.chdir(dir_)

with open('../datasets/sqldata/stats.yaml') as stats:
    stats=yaml.safe_load(stats)

#TODO django management command for running the experiment
diagnoses=stats['diagnosis_for_selection']
for diagnosis in diagnoses:
    Graph(diagnosis=diagnosis)()

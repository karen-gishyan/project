import os
import yaml
from utils import DataConversion

dir_=os.path.dirname(__file__)
os.chdir(dir_)

with open('../datasets/sqldata/stats.yaml') as stats, open('info.yaml') as info:
    stats=yaml.safe_load(stats)
    info=yaml.safe_load(info)

diagnoses=stats['diagnosis_for_selection']
timesteps=info['timesteps']
for diagnosis in diagnoses:
    for t in timesteps:
        DataConversion(diagnosis=diagnosis,timestep=t).average_save_feature_time_series().\
            convert_drugs_dummy_data_format()



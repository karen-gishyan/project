from configure import configure
configure()
import os
import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence

dir_=os.path.dirname(__file__)
dir_=os.path.join(dir_,'datasets')
os.chdir(dir_)


class DataWithStages:

    drugs_df=pd.read_csv('sqldata/drugs.csv')
    features_df=pd.read_csv('sqldata/features.csv')

    def __init__(self,diagnosis_name=None) -> None:
        """
        Patients which have been given between 30 and 50 drugs each,
        have a list of features which have been given measured between 30 and 50 times each,
        and have stayed between 6 and 8 days (each two days is a timestep, the last timestep
        can have more than 2 days).
        """

        drugs_df= self.drugs_df[self.drugs_df.diagnosis==diagnosis_name]
        features_df=self.features_df[self.features_df.diagnosis==diagnosis_name]
        self.diagnosis_name=diagnosis_name

        drugs_df[['admittime','dischtime','startdate','enddate']] =drugs_df[['admittime','dischtime','startdate','enddate']] .\
            apply(lambda x: pd.to_datetime(x).dt.date)
        drugs_df['stay_length']=(drugs_df['dischtime']-drugs_df['admittime']).dt.days
        drugs_df=drugs_df[(drugs_df['stay_length']>=6) & (drugs_df['stay_length']<=8)]

        features_df[['admittime','dischtime','charttime']]=features_df[['admittime','dischtime','charttime']].\
            apply(lambda x: pd.to_datetime(x).dt.date)
        features_df['stay_length']=(features_df['dischtime']-features_df['admittime']).dt.days
        features_df=features_df[(features_df['stay_length']>=6) & (features_df['stay_length']<=8)]

        drugs_df_admissions=list(drugs_df.hadm_id_id.unique())
        features_df_admissions=list(features_df.hadm_id_id.unique())

        self.admissions_intersection=sorted(list(set(drugs_df_admissions) & set(features_df_admissions)))
        self.drugs_df=drugs_df[drugs_df.hadm_id_id.isin(self.admissions_intersection)]
        self.features_df=features_df[features_df.hadm_id_id.isin(self.admissions_intersection)]

    def feature_and_drugs_to_timsetep_tensors(self):
        """
        For each admission, obtain the values for features for each stage.
        Args:
            no_padding (bool, optional): _description_. Defaults to True.
        """
        feature_count_vector=self.features_df['itemid'].value_counts()
        feature_ids=feature_count_vector.index[:10]

        #generating feature data
        t1_feature_data,t2_feature_data,t3_feature_data=[],[],[]
        t1_drug_data,t2_drug_data,t3_drug_data=[],[],[]
        output_labels=[]
        for c, adm_id in enumerate(self.admissions_intersection):
            admission_df=self.features_df[self.features_df['hadm_id_id']==adm_id]
            admittime=admission_df['admittime'].iloc[0]
            dischtime=admission_df['dischtime'].iloc[0]
            # create a date range between admission start and end dates
            date_range=pd.date_range(admittime,dischtime,freq='d')
            date_range_df={i.date():index+1 for index, i in enumerate(date_range)}
            t1_feature_batch,t2_feature_batch,t3_feature_batch=[],[],[]

            for feature_id in feature_ids:
                admission_feature_df=admission_df[admission_df['itemid']==feature_id]
                t1_single_feature,t2_single_feature,t3_single_feature=[],[],[]
                for _, row in admission_feature_df.iterrows():
                    try:
                        if date_range_df[row['charttime']] in [1,2]:

                            t1_single_feature.append(float(row['value']))
                        elif date_range_df[row['charttime']] in [3,4]:
                            t2_single_feature.append(float(row['value']))
                        else:
                            t3_single_feature.append(float(row['value']))
                    except:
                        # db issue, such cases are not ok
                        # feature value simply stays empty
                        print(f'Exception on iteration: {c}')
                        print("Chartevent date not a date from patient's staying interval")
                        continue
                t1_feature_batch.append(torch.Tensor(t1_single_feature))
                t2_feature_batch.append(torch.Tensor(t2_single_feature))
                t3_feature_batch.append(torch.Tensor(t3_single_feature))

            # first pad the features to have the same length
            t1_feature_data.append(pad_sequence((*t1_feature_batch,),padding_value=-1))
            t2_feature_data.append(pad_sequence((*t2_feature_batch,),padding_value=-1))
            t3_feature_data.append(pad_sequence((*t3_feature_batch,),padding_value=-1))
            print(f"Iter {c}completed for features.")

            #generating output and drugs data
            self.drugs_df['drug']=LabelEncoder().fit_transform(self.drugs_df['drug'])
            admission_drugs_df=self.drugs_df[self.drugs_df['hadm_id_id']==adm_id]

            discharge=admission_drugs_df['discharge_location'].iloc[0]

            if discharge=="HOME":
                output_labels.append(1)
            else:
                output_labels.append(0)

            t1_drug, t2_drug, t3_drug=[],[],[]
            for _, row in admission_drugs_df.iterrows():
                drug_startdate=row['startdate']
                drug_enddate=row['enddate']
                try:
                    drug_date_range=list(map(lambda i:i.date(),pd.date_range(drug_startdate,drug_enddate,freq='d')))
                except ValueError:
                    print('Exception on date-range generation.')
                    continue
                for drug_date in drug_date_range:
                    try:
                        if date_range_df[drug_date] in [1,2]:
                            t1_drug.append(row['drug'])
                        elif date_range_df[drug_date] in [3,4]:
                            t2_drug.append(row['drug'])
                        else:
                            t3_drug.append(row['drug'])
                    except:
                        print(f'Exception on iteration: {c}')
                        print("Drug date not a date from patient's staying interval")
                        continue

            t1_drug_data.append(torch.IntTensor(t1_drug))
            t2_drug_data.append(torch.IntTensor(t2_drug))
            t3_drug_data.append(torch.IntTensor(t3_drug))
            print(f"Iter {c}completed for drugs.")

        #list cannot be converted to Tensor if Tensors inside have unequal shapes.
        #second pad the feature batches within each t to have the same length
        t1_feature_data=pad_sequence((*t1_feature_data,),batch_first=True, padding_value=-1)
        t2_feature_data=pad_sequence((*t2_feature_data,),batch_first=True,padding_value=-1)
        t3_feature_data=pad_sequence((*t3_feature_data,),batch_first=True,padding_value=-1)

        t1_drug_data=pad_sequence((*t1_drug_data,),batch_first=True,padding_value=-1)
        t2_drug_data=pad_sequence((*t2_drug_data,),batch_first=True,padding_value=-1)
        t3_drug_data=pad_sequence((*t3_drug_data,),batch_first=True,padding_value=-1)

        general_path=os.path.join(dir_,f"{self.diagnosis_name}")
        t1_path=os.path.join(general_path,"t1")
        t2_path=os.path.join(general_path,"t2")
        t3_path=os.path.join(general_path,"t3")

        if not os.path.isdir(t1_path):os.makedirs(t1_path)
        if not os.path.isdir(t2_path):os.makedirs(t2_path)
        if not os.path.isdir(t3_path):os.makedirs(t3_path)

        #save first timestage data
        torch.save(torch.Tensor(t1_feature_data),os.path.join(t1_path,'features.pt'))
        torch.save(torch.IntTensor(t1_drug_data),os.path.join(t1_path,'drugs.pt'))

        #save second timestage data
        torch.save(torch.Tensor(t2_feature_data),os.path.join(t2_path,'features.pt'))
        torch.save(torch.IntTensor(t2_drug_data),os.path.join(t2_path,'drugs.pt'))

        # save third timestage data
        torch.save(torch.Tensor(t3_feature_data),os.path.join(t3_path,'features.pt'))
        torch.save(torch.IntTensor(t3_drug_data),os.path.join(t3_path,'drugs.pt'))

        # save output
        torch.save(torch.IntTensor(output_labels),os.path.join(general_path,'output.pt'))


def save_data():
    with open('sqldata/stats.yaml','r') as file:
        stats=yaml.safe_load(file)

    diagnoses=stats['diagnosis_for_selection']
    for diagnosis in diagnoses:
        DataWithStages(diagnosis_name=diagnosis).feature_and_drugs_to_timsetep_tensors()

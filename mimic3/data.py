from configure import configure
configure()
import os
import json
import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence

dir_=os.path.dirname(__file__)
dir_=os.path.join(dir_,'datasets')
os.chdir(dir_)


class RNNData():
    """
    Logical issues:
        1. Stage lengths do not match between drugs.
            e.g. first 10 drugs in 2 days, becomes stage1, second 10 drugs in 5 days, becomes stage 2.
        2. Stage lengtsh do not match between drugs and features.
            e.g first 10 feature measurements are taken over 7 days, but as the count is 10,
            is included in the first stage with the drugs given in 2 days.
            Clearly some of the measrumenets are taken after the drugs are given and would need
            to be part of stage two.
    """
    def __init__(self) :
        drugs_df=pd.read_csv('rnn/drugs.csv')
        features_df=pd.read_csv('rnn/features.csv')
        drugs_df_admissions=list(drugs_df.hadm_id_id.unique())
        features_df_admissions=list(features_df.hadm_id_id.unique())

        self.admissions_intersection=sorted(list(set(drugs_df_admissions) & set(features_df_admissions)))
        self.drugs_df=drugs_df[drugs_df.hadm_id_id.isin(self.admissions_intersection)]
        self.features_df=features_df[features_df.hadm_id_id.isin(self.admissions_intersection)]


    def feature_to_tensor(self):
        feature_count_vector=self.features_df['itemid'].value_counts()
        # select ids of 10 most frequently appearing features
        #TODO feature length can be modified from here
        feature_ids=feature_count_vector.index[:10]

        total=[]
        empty_admissions=[]
        for adm_id in self.admissions_intersection:
            adm_feature=[]
            for feature_id in feature_ids:
                values=torch.Tensor(self.features_df[(self.features_df['hadm_id_id']==adm_id) &\
                    (self.features_df['itemid']==feature_id)]['value'].values)
                adm_feature.append(values)
            # if there are values for at least one feature, append to total
            if any([len(i) for i in adm_feature]):
                total.append(pad_sequence(adm_feature,batch_first=True).T)
            else:
                empty_admissions.append(adm_id)
        self.feature_tensors=pad_sequence(total,batch_first=True)
        torch.save(self.feature_tensors,'rnn/tensors/features.pt')
        self.admissions_intersection=list(set(self.admissions_intersection)-set(empty_admissions))

    def drug_to_tensor(self):
        # fit without transforming and store labels
        self.store_encoded_labels(self.drugs_df['drug'],'drug')
        self.store_encoded_labels(self.drugs_df['discharge_location'],'discharge_location')
        # transform actual column
        self.drugs_df['drug']=LabelEncoder().fit_transform(self.drugs_df['drug'])
        self.drugs_df['discharge_location']=LabelEncoder().fit_transform(self.drugs_df['discharge_location'])

        total_drugs,total_labels=[],[]
        for adm_id in self.admissions_intersection:
            drugs=torch.Tensor(self.drugs_df[self.drugs_df['hadm_id_id']==adm_id].drug.values).view(-1,1)
            total_drugs.append(drugs)
            labels=torch.Tensor(self.drugs_df[self.drugs_df['hadm_id_id']==adm_id].discharge_location.values).view(-1,1)
            total_labels.append(labels)
        self.drug_tensors=pad_sequence(total_drugs,batch_first=True)
        self.label_tensors=pad_sequence(total_labels,batch_first=True)
        torch.save(self.drug_tensors,'rnn/tensors/drugs.pt')
        torch.save(self.label_tensors,'rnn/tensors/labels.pt')

    def divide_tensors_to_timesteps(self):
        def concat_and_save(list_of_tensors,features=True):
            for i,t in enumerate(list_of_tensors):
                t=torch.cat((*t,))
                if features:
                    torch.save(t,f'rnn/tensors/features_t{i+1}.pt')
                else:
                    torch.save(t,f'rnn/tensors/drugs_t{i+1}.pt')

        features=torch.load('rnn/tensors/features.pt')
        drugs=torch.load('rnn/tensors/drugs.pt')
        features_t1,features_t2,features_t3=[],[],[]
        drugs_t1,drugs_t2,drugs_t3=[],[],[]
        for batch in features:
            features_t1.append(batch[0:10].view(-1,10,10))
            features_t2.append(batch[10:20].view(-1,10,10))
            features_t3.append(batch[20:30].view(-1,10,10))

        for batch in drugs:
            drugs_t1.append(batch[0:10].view(-1,10,1))
            drugs_t2.append(batch[10:20].view(-1,10,1))
            drugs_t3.append(batch[20:30].view(-1,10,1))

        concat_and_save([features_t1,features_t2,features_t3],features=True)
        concat_and_save([drugs_t1,drugs_t2,drugs_t3],features=False)

    def store_encoded_labels(self,series,col_name):
        encoder=LabelEncoder()
        encoder.fit(series)
        # convert int32 to int to be json serializable
        transform=list(map(lambda i:int(i),encoder.transform(encoder.classes_)))
        drug_mapping=dict(zip(encoder.classes_,transform))
        with open(f"../json/{col_name}_mapping.json",'w') as file:
            json.dump(drug_mapping,file)

    def __call__(self):
        #feature_to_tensor() should be run first
        self.feature_to_tensor()
        self.drug_to_tensor()
        self.divide_tensors_to_timesteps()


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

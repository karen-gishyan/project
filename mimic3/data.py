from configure import configure
configure()
import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence

dir_=os.path.dirname(__file__)
dir_=os.path.join(dir_,'datasets')
os.chdir(dir_)

def process_and_combine_datasets_limited():
    """
    We get each unique admissions last 10 'Drug and 'Heart Rate' feature values ordered by
    the care unit. In other words, what are the patient features based on the fact that they
    have been admitted to a given care unit.
    Returns:
        pf.DataFrame: Combined/concatinated dataframe.
    """

    care_units_with_admissions=pd.read_csv('care_units_with_admisisons_and_drugs.csv',usecols=[i for i in range(7)])
    care_units_with_admissions.rename(columns={'hadm_id_id':'hadm_id'},inplace=True)
    # copied for joining purposes
    care_units_with_admissions_copy=care_units_with_admissions.copy().reset_index()
    admissions_and_patient_features=pd.read_csv('admissions_and_patient_features.csv',usecols=[i for i in range(5)])
    admissions_and_patient_features=admissions_and_patient_features[admissions_and_patient_features.itemid==220045]

    care_units_admissions_col=list(care_units_with_admissions.hadm_id.unique())
    admissions_admissions_col=list(admissions_and_patient_features.hadm_id_id.unique())
    admisisons_intersection=sorted(list(set(care_units_admissions_col) & set(admissions_admissions_col)))

    care_units_with_admissions_copy=care_units_with_admissions_copy.reindex(columns=list(care_units_with_admissions.columns)
                                            +list(admissions_and_patient_features.columns))
    for admission_id in care_units_admissions_col:
        if not admission_id in admisisons_intersection:
            continue
        subset_index=care_units_with_admissions[care_units_with_admissions.hadm_id==admission_id].index
        df_admissions=admissions_and_patient_features[(admissions_and_patient_features.hadm_id_id==admission_id)
                                                      &(admissions_and_patient_features.itemid==220045)]

        df_admissions.index=subset_index
        lower,upper=list(subset_index)[0],list(subset_index)[-1]
        # copy values from the second df to the first df
        # if concat did not append the same columns for each concact operations it
        # would have been a better solution.
        care_units_with_admissions_copy.loc[lower:upper,list(df_admissions.columns)]=df_admissions.values

    care_units_with_admissions_copy.dropna(inplace=True)
    # care_units_with_admissions_copy.to_csv('combined.csv')

    return care_units_with_admissions_copy.dropna(inplace=True)


def construct_pytorch_dataset_limited():
    df=pd.read_csv('combined.csv')
    # remove some columns then select only categorical columns
    df_categorical=df.loc[:,~df.columns.isin(['startdate','enddate','charttime'])].\
        select_dtypes(include=['object'])

    # encode labels into a single column
    for col in df_categorical.columns:
        df_categorical[col]=LabelEncoder().fit_transform(df_categorical[col])
    df_categorical['value']=df['value']
    df_categorical['hadm_id']=df['hadm_id']

    unique_admissions=df_categorical.hadm_id.unique()
    timestep1_df,timestep2_df, timestep3_df=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for adm in unique_admissions:
        iter_df=df_categorical[df_categorical.hadm_id==adm]
        timestep1_df=pd.concat([timestep1_df,iter_df[0:3]])
        timestep2_df=pd.concat([timestep2_df,iter_df[3:6]])
        timestep3_df=pd.concat([timestep3_df,iter_df[6:9]])


    # timestep1_df.to_csv("pytorch_t1_df.csv",index=False)
    # timestep2_df.to_csv("pytorch_t2_df.csv",index=False)
    timestep3_df.to_csv("pytorch_t3_df.csv",index=False)
    print('Done')


class RNNData():
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

import os
import torch
import sys
import yaml
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from helpers import convert_drugs_dummy_data_format
import pandas as pd
from sdv.tabular import TVAE
from sdv.evaluation import evaluate
import warnings

warnings.filterwarnings("ignore")

class DataConversion:
    def __init__(self,diagnosis,timestep):
        self.diagnosis=diagnosis
        self.timestep=timestep
        self.feature_tensors=torch.load(f"../datasets/{diagnosis}/t{timestep}/features.pt")
        self.drug_tensors=torch.load(f"../datasets/{self.diagnosis}/t{self.timestep}/drugs.pt")

        self.path=f"{self.diagnosis}/t{self.timestep}"
        if not os.path.exists(self.path): os.makedirs(self.path)

    def average_save_feature_time_series(self):
        # 0 so as results are not affected as a result of averaging
        self.feature_tensors[self.feature_tensors==-1]=0
        self.feature_tensors=self.feature_tensors.mean(dim=1)
        self.save_path=os.path.join(self.path,"features.pt")
        torch.save(self.feature_tensors,self.save_path)
        return self

    def convert_drugs_dummy_data_format(self):
        self.dummy_format_tensors=convert_drugs_dummy_data_format(self.drug_tensors)
        self.save_path=os.path.join(self.path,"drugs.pt")
        torch.save(self.dummy_format_tensors,self.save_path)
        return self


class Synthetic_Data_With_Stages:
    def __init__(self,diagnosis,timestep,df_type) -> None:
        self.diagnosis=diagnosis
        assert timestep in [1,2,3], f"Unkown timestep {timestep}."
        self.timestep=timestep
        assert df_type in ['features','drugs'], f"Unkown type {df_type}."
        self.df_type=df_type
        self.X=torch.load(f"{self.diagnosis}/t{self.timestep}/{df_type}.pt")
        assert len(self.X.shape)==2, "Data shape is not 2D, use DataConversion class for transforming it into 2D."


    def get_synthetic_data(self):
        if self.timestep in [1,2]:
            self.X=self.X.numpy()
            n_cols=self.X.shape[1]
            df=pd.DataFrame(self.X,columns=[str(i) for i in range(n_cols)])
            try:
                df_synthetic=pd.read_csv(f"{self.diagnosis}/t{self.timestep}/{self.df_type}_synthetic.csv")
            except FileNotFoundError:
                model = TVAE()
                model.fit(df)
                df_synthetic = model.sample(5000)
                df_synthetic.to_csv(f"{self.diagnosis}/t{self.timestep}/{self.df_type}_synthetic.csv",index=False)
        else:
            # in the 3rd timestep, output is a single vector, so we generate data with [drugs,output] and [features,output],
            # concatenated instead of separately as for the previous two stages with [drugs] and [features].
            self.output=torch.load(f"../datasets/{self.diagnosis}/output.pt").reshape(-1,1)
            self.XY=torch.cat((self.X,self.output),dim=1).numpy()
            n_cols=self.XY.shape[1]
            df=pd.DataFrame(self.XY,columns=[str(i) for i in range(n_cols)])
            try:
                df_synthetic=pd.read_csv(f"{self.diagnosis}/t{self.timestep}/{self.df_type}_synthetic.csv")
            except FileNotFoundError:
                model = TVAE()
                model.fit(df)
                df_synthetic = model.sample(5000)
                df_synthetic.to_csv(f"{self.diagnosis}/t{self.timestep}/{self.df_type}_synthetic.csv",index=False)

        # evaluate the fitted dataset similarity
        res=evaluate(df_synthetic,df,aggregate=False)
        print(self.df_type)
        print(f"For {self.diagnosis},t{self.timestep},synthetic model score is:{round(res['raw_score'][0],2)}")

def save_synthetic_data():
    dir_=os.path.dirname(__file__)
    os.chdir(dir_)

    with open('../datasets/sqldata/stats.yaml','r') as file:
        stats=yaml.safe_load(file)

    diagnoses=stats['diagnosis_for_selection']
    for diagnosis in diagnoses:
        for t in [1,2,3]:
            for df_type in ['features','drugs']:
                Synthetic_Data_With_Stages(diagnosis=diagnosis,timestep=t,df_type=df_type).get_synthetic_data()


def reset_weights(m):
  '''
    #NOTE: source: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def balance_datasets(dataset_list,output):
    non_zero_indices=(output.flatten()==1).nonzero().flatten()
    #select 0 elements equal to the number of non_zero_indices
    zero_indices=(output.flatten()==0).nonzero()[:len(non_zero_indices)].flatten()
    for index, dataset in enumerate(dataset_list):
        X_0=torch.index_select(input=dataset.X,dim=0,index=zero_indices)
        X_1=torch.index_select(input=dataset.X,dim=0,index=non_zero_indices)
        X=torch.cat((X_0,X_1))

        Y_O=torch.index_select(input=dataset.Y,dim=0,index=zero_indices)
        Y_1=torch.index_select(input=dataset.Y,dim=0,index=non_zero_indices)
        Y=torch.cat((Y_O,Y_1))

        dataset.X=X
        dataset.Y=Y
        dataset_list[index]=dataset

    return dataset_list



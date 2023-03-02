import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import sigmoid
import pandas as pd



class BaseDataset(Dataset):
    def __init__(self,diagnosis, timestep):
        super().__init__()
        self.diagnosis=diagnosis
        self.timestep=timestep
        self.validate_timestep()

    def validate_timestep(self):
        if self.timestep not in [1,2,3]:
            raise ValueError(f"timestep {self.timestep} not supported.")

    def __getitem__(self, index) :
        return self.X[index].float(),self.Y[index].float()

    def __len__(self):
        return len(self.X)


class FeatureDataset(BaseDataset):
    def __init__(self,diagnosis, timestep):
        super().__init__(diagnosis, timestep)
        self.X=torch.Tensor(pd.read_csv(f"{diagnosis}/t{timestep}/features_synthetic.csv").values)
        if timestep==1:
            self.Y=torch.Tensor(pd.read_csv(f"{diagnosis}/t{timestep+1}/features_synthetic.csv").values)
        elif timestep==2:
            self.Y=torch.Tensor(pd.read_csv(f"{diagnosis}/t{timestep+1}/features_synthetic.csv").iloc[:,:-1].values)
        else:
            self.Y=torch.Tensor(pd.read_csv(f"{diagnosis}/t3/features_synthetic.csv").iloc[:,-1].values)



class DrugDataset(FeatureDataset):
    def __init__(self,diagnosis, timestep):
        self.diagnosis=diagnosis
        self.timestep=timestep
        self.X=torch.Tensor(pd.read_csv(f"{diagnosis}/t{timestep}/drugs_synthetic.csv").values)
        if timestep==1:
            self.Y=torch.Tensor(pd.read_csv(f"{diagnosis}/t{timestep+1}/drugs_synthetic.csv").values)
        elif timestep==2:
            self.Y=torch.Tensor(pd.read_csv(f"{diagnosis}/t{timestep+1}/drugs_synthetic.csv").iloc[:,:-1].values)
        else:
            # should be t3 feautures, it is correct
            self.Y=torch.Tensor(pd.read_csv(f"{diagnosis}/t3/features_synthetic.csv").iloc[:,-1].values)


class Model(nn.Module):
    def __init__(self,input_size,output_size,sigmoid_activation=False):
        super().__init__()
        self.linear=nn.Linear(input_size,output_size)
        self.sigmoid_activation=sigmoid_activation

    def forward(self,x):

        out=self.linear(x)
        return out

class MultiStageModel(nn.Module):
    def __init__(self,feature_model, drug_model,output_model):
        super().__init__()
        self.feature_model=feature_model
        self.drug_model=drug_model
        self.output_model=output_model

    def forward(self,features_t1,drugs_t1):
        #t2 pred
        X=torch.cat((features_t1,drugs_t1),dim=1)
        features_t2_pred=self.feature_model(X)
        drugs_t2_pred=sigmoid(self.drug_model(X))

        #t3 pred
        X=torch.cat((features_t2_pred,drugs_t2_pred),dim=1)
        features_t3_pred=self.feature_model(X)
        drugs_t3_pred=sigmoid(self.drug_model(X))

        #output pred
        X=torch.cat((features_t3_pred,drugs_t3_pred),dim=1)
        output_pred=sigmoid(self.output_model(X))
        #NOTE feature preds are returned but not used
        return features_t2_pred,drugs_t2_pred,features_t3_pred,drugs_t3_pred,output_pred



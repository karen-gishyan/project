import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch import sigmoid
from torch.nn import ReLU



class BaseDataset(Dataset):
    def __init__(self,diagnosis, timestep):
        super().__init__()
        self.diagnosis=diagnosis
        self.timestep=timestep
        self.validate_timestep()

    def validate_timestep(self):
        if self.timestep not in [1,2,3]:
            raise ValueError(f"timestep {self.timestep} not supported.")

    def train_test_split(self):
        train_size = int(0.8 * len(self.X))
        self.train_X, self.test_X = self.X[:train_size],self.X[train_size:]
        self.train_Y,self.test_y=self.Y[:train_size],self.Y[train_size:]

    def __getitem__(self, index) :
        # test will not be iterated.
        return self.train_X[index].float(),self.train_Y[index].float()

    def __len__(self):
        return len(self.train_X)


class FeatureDataset(BaseDataset):
    def __init__(self,diagnosis, timestep):
        super().__init__(diagnosis, timestep)
        self.X=torch.load(f"{diagnosis}/t{timestep}/features.pt")
        if timestep in [1,2]:
            self.Y=torch.load(f"{diagnosis}/t{timestep+1}/features.pt")
        else:
            self.Y=torch.load(f"../datasets/{diagnosis}/output.pt")
        self.train_test_split()


class DrugDataset(FeatureDataset):
    def __init__(self,diagnosis, timestep):
        super().__init__(diagnosis, timestep)
        self.X=torch.load(f"{diagnosis}/t{timestep}/drugs.pt")
        if timestep in [1,2]:
            self.Y=torch.load(f"{diagnosis}/t{timestep+1}/drugs.pt")
        else:
            self.Y=torch.load(f"../datasets/{diagnosis}/output.pt")
        self.train_test_split()

class Model(nn.Module):
    def __init__(self,input_size,output_size,sigmoid_activation=False):
        super().__init__()
        self.linear=nn.Linear(input_size,output_size)
        self.sigmoid_activation=sigmoid_activation

    def forward(self,x):
        if self.sigmoid_activation:
            out=sigmoid(self.linear(x))
        else:
            out=ReLU()(self.linear(x))
        return out

class MultiStageModel(nn.Module):
    def __init__(self,feature_model, drug_model,output_model):
        super().__init__()
        self.feature_model=feature_model
        self.drug_model=drug_model
        self.output_model=output_model

    def forward(self,feature_Xt1,drug_Xt1):
        #t2 pred
        X=torch.cat((feature_Xt1,drug_Xt1),dim=1)
        feature_Xt2=self.feature_model(X)
        drug_Xt2=self.drug_model(X)

        #t3 pred
        X=torch.cat((feature_Xt2,drug_Xt2),dim=1)
        feature_Xt3=self.feature_model(X)
        drug_Xt3=self.drug_model(X)

        #output pred
        X=torch.cat((feature_Xt3,drug_Xt3),dim=1)
        out=self.output_model(X)

        return feature_Xt2,drug_Xt2,feature_Xt3,drug_Xt3,out


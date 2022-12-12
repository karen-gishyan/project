# Do drug conversion dummy format, to see if drug exists or not.
import os
import sys
import torch
from torch.utils.data import Dataset, random_split
import torch.nn as nn
from torch import sigmoid

path=os.path.dirname(os.path.dirname(__file__))
sys.path.append(path)
from helpers import convert_drugs_dummy_data_format

class DataSet(Dataset):
    def __init__(self,diagnosis,dir_name,method=None) -> None:
        super().__init__()
        self.diagnosis=diagnosis
        if method:
            self.drug_path=f"{diagnosis}/{dir_name}/{method}/combined_drugs.pt"
            self.output_path=f"{diagnosis}/{dir_name}/{method}/test_output_expanded.pt"
        else:
            self.drug_path=f"{diagnosis}/{dir_name}/combined_drugs.pt"
            self.output_path=f"{diagnosis}/{dir_name}/test_output_expanded.pt"

        drug_tensor=torch.load(self.drug_path)
        output_tensor=torch.load(self.output_path).view(-1,1)

        self.drug_path=os.path.join(os.path.split(self.drug_path)[0],"combined_drugs_dummy.pt")
        if not os.path.exists(self.drug_path):
            drug_tensor=convert_drugs_dummy_data_format(drug_tensor)
            torch.save(drug_tensor,self.drug_path)
        else:
            drug_tensor=torch.load(self.drug_path)

        train_size=int(len(drug_tensor)*0.7)
        test_size=len(drug_tensor)-train_size
        # we could use train, test directly, but we rather deal with tensors than Subdatasets.
        train,test=random_split(drug_tensor,[train_size,test_size])
        self.train_X=drug_tensor[train.indices]
        self.train_y=output_tensor[train.indices]
        self.test_X=drug_tensor[test.indices]
        self.test_y=output_tensor[test.indices]

    def __getitem__(self, index):
        return self.train_X[index].float(),self.train_y[index].float()

    def __len__(self):
        return len(self.train_X)


class EvaluationModel(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear=nn.Linear(input_size,output_size)

    def forward(self,x):
         out=sigmoid(self.linear(x))
         return out

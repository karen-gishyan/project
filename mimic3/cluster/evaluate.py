# Do drug conversion dummy format, to see if drug exists or not.
import os
import sys
import torch
from torch.utils.data import Dataset
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
        self.output_tensor=torch.load(self.output_path).view(-1,1)

        self.drug_path=os.path.join(os.path.split(self.drug_path)[0],"combined_drugs_dummy.pt")
        #NOTE in case output, data generation logic changes, removing reading from existing
        # and generate new dummy data.
        if not os.path.exists(self.drug_path):
            self.drug_tensor=convert_drugs_dummy_data_format(drug_tensor)
            torch.save(drug_tensor,self.drug_path)
        else:
            self.drug_tensor=torch.load(self.drug_path)

    def __getitem__(self, index):
        return self.drug_tensor[index].float(),self.output_tensor[index].float()

    def __len__(self):
        return len(self.drug_tensor)

from torch.nn import LogSoftmax
class EvaluationModel(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear=nn.Linear(input_size,output_size)

    def forward(self,x):
         out=sigmoid(self.linear(x))
         return out

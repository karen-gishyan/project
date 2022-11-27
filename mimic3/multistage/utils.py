import os
import torch
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from helpers import convert_drugs_dummy_data_format

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



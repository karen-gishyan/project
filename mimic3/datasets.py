import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
import os

dir_=os.path.dirname(__file__)
dir_=os.path.join(dir_,'datasets')
os.chdir(dir_)



class Data(Dataset):

    # Constructor
    def __init__(self, K=3, N=500):
        self.t1=pd.read_csv("pytorch_t1_df.csv")
        self.t2=pd.read_csv("pytorch_t2_df.csv")
        self.t3=pd.read_csv("pytorch_t3_df.csv")

        length=min(self.t1.shape[0],self.t2.shape[0])
        # to be flattened to 3 columns, array length needs to be divisible by 3,
        # thus we keep the maximum number of rows divisible by 3.
        n_rows_to_keep=length-length%3
        self.len=n_rows_to_keep
        self.t1=self.t1[:n_rows_to_keep]
        self.t2=self.t2[:n_rows_to_keep]
        t1_value_2d=self.t1.value.values.flatten().reshape((-1,3))
        t1_drug_2d=self.t1.drug.values.flatten().reshape((-1,3))
        t1_discharge_location_2d=self.t1.discharge_location.values.flatten().reshape((-1,3))

        t2_value_2d=self.t2.value.values.flatten().reshape((-1,3))
        t2_drug_2d=self.t2.drug.values.flatten().reshape((-1,3))
        t2_discharge_location_2d=self.t2.discharge_location.values.flatten().reshape((-1,3))

        self.t1_X=torch.from_numpy(np.concatenate((t1_value_2d,t1_drug_2d),axis=1))
        self.t1_y1=torch.from_numpy(t2_value_2d)
        self.t1_y2=torch.from_numpy(t2_drug_2d)

    # Getter
    def __getitem__(self, index):
        #TODO does not have to be t1_X and y1
        return (f"X:{self.t1_X[index]}, y-(Heart-Rate):{self.t1_y1[index]},y-(Drug):{self.t1_y2[index]}")

    # Get Length
    def __len__(self):
        return self.len


print(Data()[0])
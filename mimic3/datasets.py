import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os

dir_=os.path.dirname(__file__)
dir_=os.path.join(dir_,'datasets')
os.chdir(dir_)


class Data(Dataset):
    def __init__(self,timestep,return_feature=True):
        self.t1=pd.read_csv("pytorch_t1_df.csv")
        self.t2=pd.read_csv("pytorch_t2_df.csv")
        self.t3=pd.read_csv("pytorch_t3_df.csv")
        self.timestep=timestep
        self.return_feature=return_feature

        length=min(self.t1.shape[0],self.t2.shape[0],self.t3.shape[0])
        # to be reshaped to 3 columns, array length needs to be divisible by 3,
        # thus we keep the maximum number of rows divisible by 3.
        n_rows_to_keep=length-length%3
        self.len=n_rows_to_keep
        # all three same shape because the target of each df is selected from the next df,
        # thus should have the same number of rows.
        self.t1=self.t1[:n_rows_to_keep]
        self.t2=self.t2[:n_rows_to_keep]
        self.t3=self.t3[:n_rows_to_keep]

        #t1 features
        t1_value_2d=self.t1.value.values.reshape((-1,3))

        #t2 features
        t2_value_2d=self.t2.value.values.reshape((-1,3))
        t2_drug_2d=self.t2.drug.values.reshape((-1,3))

        #t3 features
        t3_value_2d=self.t3.value.values.reshape((-1,3))
        t3_drug_2d=self.t3.drug.values.reshape((-1,3))

        #t1 X and ys
        self.t1_X=torch.from_numpy(t1_value_2d)
        self.t1_y1=torch.from_numpy(t2_value_2d)
        self.t1_y2=torch.from_numpy(t2_drug_2d)

        #t2 X and ys
        self.t2_X=torch.from_numpy(np.concatenate((t2_value_2d,t2_drug_2d),axis=1))
        self.t2_y1=torch.from_numpy(t3_value_2d)
        self.t2_y2=torch.from_numpy(t3_drug_2d)

        #t3 X and ys
        self.t3_X=torch.from_numpy(np.concatenate((t3_value_2d,t3_drug_2d),axis=1))
        self.t3_y=torch.from_numpy(self.t3.discharge_location.values)

    def __getitem__(self, index):
        # final timestep
        if self.timestep=='t3':
            return self.t3_X[index].float(), self.t3_y[index].float()
        else:
            if self.return_feature:
                # y is features
                return eval(f"self.{self.timestep}_X[index].float()"), eval(f"self.{self.timestep}_y1[index].float()")
            # y is drugs
            return eval(f"self.{self.timestep}_X[index].float()"), eval(f"self.{self.timestep}_y2[index].float()")

    def __len__(self):
        return len(self.t1_X)

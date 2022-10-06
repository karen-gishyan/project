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


class RNNData(Dataset):
    def __init__(self,is_feature=True,timestep=1):
        self.is_feature=is_feature
        self.timestep=timestep

        self.X1_feature=torch.load('rnn/tensors/features_t1.pt')
        self.Y1_feature=self.X2_feature=torch.load('rnn/tensors/features_t2.pt')
        self.Y2_feature=self.X3_feature=torch.load('rnn/tensors/features_t3.pt')
        #TODO y may need to be 3D as well
        self.y3_feature=torch.load('rnn/tensors/labels.pt')

        self.X1_drug=torch.load('rnn/tensors/drugs_t1.pt')
        self.Y1_drug=self.X2_drug=torch.load('rnn/tensors/drugs_t2.pt')
        self.Y2_drug=self.X3_drug=torch.load('rnn/tensors/drugs_t3.pt')
        self.y3_drug=torch.load('rnn/tensors/labels.pt')

    def __getitem__(self, index):
        if self.is_feature:
            if self.timestep==1:
                return self.X1_feature[index].float(), self.Y1_feature[index].float()
            elif self.timestep==2:
                return self.X2_feature[index].float(), self.Y2_feature[index].float()
            else:
                return self.X3_feature[index].float(), self.y3_feature[index][20:30].float()
        else:
            if self.timestep==1:
                return self.X1_drug[index].float(),self.Y1_drug[index].float()
            elif self.timestep==2:
                return self.X2_drug[index].float(), self.Y2_drug[index].float()
            else:
                # for y we take the last 10 labels, but any 10 values could have been taken.
                # all have the same value.
                return self.X3_drug[index].float(), self.y3_drug[index][20:30].float()

    def __len__(self):
        # all features and drugs for all timestep have the same length
        return len(self.X1_feature)

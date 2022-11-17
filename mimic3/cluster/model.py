import os
import torch
from typing import Callable


class DistanceModel:
    def __init__(self,diagnosis,timestep) -> None:
        """Load a batch of 2D Tensors using the diagnosis and timestep.
        Tensor is padded twice, and each batch (and each feature in each batch)
        has the same number of rows.

        feature_tensors (Tensor[Tensor]):
            3D Tensor of shape(n,m,k), where
            n is is the batch size (number of admissions)
            m is the number of time series observations (rows)
            k is the number of features (columns, fixed, k=10)

        Args:
            diagnosis (str): name of the diagnosis.
            timestep (int):  timestep (stage).
        """
        self.diagnosis=diagnosis
        self.timestep=timestep
        self.feature_tensors=torch.load(f"../datasets/{diagnosis}/t{timestep}/features.pt")
        # needed when we select 'good' batches based on output.
        self.output=torch.load(f"../datasets/{diagnosis}/output.pt")

    def average_feature_time_series(self):
        self.feature_tensors=self.feature_tensors.mean(dim=1)
        return self

    def train_test(self):
        number_of_batches=self.feature_tensors.shape[0]
        train_size=round(number_of_batches*0.7)
        self.train_data=self.feature_tensors[:train_size,:]
        self.test_data=self.feature_tensors[train_size:,:]

        self.output_train=self.output[:train_size]
        self.output_test=self.output[train_size:]
        return self

    def select_good_batches_based_on_output(self):
        good_indices=(self.output_train==1).nonzero().flatten()
        self.train_data=self.train_data.index_select(0,good_indices)
        return self

    def calculate_similarity(self,similarity_function:Callable,
                                                  **kwargs):

        # reduce to a 1D vector
        self.target=self.train_data.mean(dim=0)
        similarities={}
        for i,batch in enumerate(self.test_data):
            score=similarity_function(batch,self.target,**kwargs)
            similarities.update({i:score})

        keys=[t[0] for t in sorted(similarities.items(),key=lambda dict_:dict_[1],reverse=True)]
        keys=torch.Tensor(keys[:5])

        path=f"{self.diagnosis}/distance/{similarity_function.__name__}/t{self.timestep}"
        if not os.path.exists(path): os.makedirs(path)
        torch.save(keys,os.path.join(path,'keys.pt'))

    def __call__(self,similarity_function):
        self.average_feature_time_series().train_test().select_good_batches_based_on_output()
        self.calculate_similarity(similarity_function=similarity_function)

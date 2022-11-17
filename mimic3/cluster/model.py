import os
import json
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
        #needed for selecting the sequences
        self.drug_tensors=torch.load(f"../datasets/{self.diagnosis}/t{self.timestep}/drugs.pt")
        # needed for selecting 'good' batches based on output.
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
        self.drugs_train=self.drug_tensors[:train_size,:]
        return self

    def select_good_batches_based_on_output(self):
        good_indices=(self.output_train==1).nonzero().flatten()
        self.train_data=self.train_data.index_select(0,good_indices)
        return self

    def calculate_similarity(self,similarity_function:Callable,
                                                  **kwargs):

        combined_test_similarities={}
        for i,test_batch in enumerate(self.test_data):
            similarities={}
            for j, train_batch in enumerate(self.train_data):
                score=similarity_function(test_batch,train_batch,**kwargs)
                similarities.update({j:score})

            keys=[t[0] for t in sorted(similarities.items(),key=lambda dict_:dict_[1],reverse=True)]
            keys=keys[:5]
            combined_test_similarities.update({i:keys})

        self.path=f"{self.diagnosis}/distance/{similarity_function.__name__}/t{self.timestep}"
        self.save_path=os.path.join(self.path,'train_keys.json')

        if not os.path.exists(self.path): os.makedirs(self.path)
        with open(self.save_path,'w') as file:
            json.dump(combined_test_similarities,file,indent=4)

        return self

    def get_drug_sequences(self):
        with open(self.save_path) as file:
            # train indices for each test item
            good_train_indices=json.load(file)
        combined_drugs_for_test=[]
        for _,indices in good_train_indices.items():
            good_drugs=self.drugs_train.index_select(0,torch.IntTensor(indices))
            combined_drugs_for_test.append(good_drugs)

        test_size=len(good_train_indices)
        max_number_of_drugs=good_drugs.shape[1]

        #TODO maybe drug sequences need to be transposed
        #TODO maybe storing in 2d is a better option
        # first dimension is each unique test instance, rows are the different drugs, columns
        # are the drugs lengths
        t=torch.cat(combined_drugs_for_test).view(test_size,-1,max_number_of_drugs).int()
        torch.save(t,os.path.join(self.path,'drug_sequences.pt'))

    def __call__(self,similarity_function):

        # data processing
        self.average_feature_time_series().train_test().select_good_batches_based_on_output()
        # calculation and saving
        self.calculate_similarity(similarity_function=similarity_function).get_drug_sequences()
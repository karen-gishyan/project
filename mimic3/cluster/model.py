import os
import json
import torch
import numpy as np
from scipy.stats import kstest
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from typing import Callable
from sklearn.decomposition import PCA


class DistanceModel:
    def __init__(self,diagnosis,timestep):
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
        #TODO should -1 be converted to 0 before averaging as in multistage?
        # maybe not because here not the value itself is important, but the similarity
        # train-test value similarity. If both are averaged the same, way, should be ok.
        self.feature_tensors=self.feature_tensors.mean(dim=1)
        return self

    def train_test(self):
        number_of_batches=self.feature_tensors.shape[0]
        train_size=round(number_of_batches*0.7)
        self.train_data=self.feature_tensors[:train_size,:]
        self.test_data=self.feature_tensors[train_size:,:]

        self.output_train=self.output[:train_size]
        self.output_test=self.output[train_size:]
        self.drugs_train=self.drug_tensors[:train_size,:]
        return self

    def save_test_output(self):
        """
        Output will be the same for all models and sub approaches.
        """
        path=f"{self.diagnosis}/test_output.pt"
        if not os.path.exists(path=path):
            torch.save(self.output_test,path)
        return self

    def select_good_batches_based_on_output(self):
        good_indices=(self.output_train==1).nonzero().flatten()
        self.train_data=self.train_data.index_select(0,good_indices)
        return self

    def calculate_similarity(self,similarity_function:Callable,
                                                  **kwargs):
        """
        Calculate similarity of testing batches with the training instances,
        select and save indices of most similar features.
        """
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

        # pad -1 for places where number of related training instances is less than 5
        t=pad_sequence((*combined_drugs_for_test,),batch_first=True,padding_value=-1)
        torch.save(t,os.path.join(self.path,'drug_sequences.pt'))


    def __call__(self,similarity_function):
        # data processing
        self.average_feature_time_series().train_test().save_test_output().select_good_batches_based_on_output()
        # calculation and saving
        self.calculate_similarity(similarity_function=similarity_function).get_drug_sequences()


class ClusteringModel(DistanceModel):
    def perform_clustering(self,clustering_function,**kwargs):
        self.output=clustering_function(**kwargs).fit(self.train_data.numpy())
        train_clusters=self.output.predict(self.train_data.numpy())
        self.test_clusters=self.output.predict(self.test_data.numpy())
        combined_cluster_indices={}
        for i,cluster in enumerate(self.test_clusters):
            #here no indice is better or worse, selection is based on first 5
            similar_train_batch_indices=np.where(train_clusters==cluster)[0][:5]
            similar_train_batch_indices=list(map(int,similar_train_batch_indices))
            combined_cluster_indices.update({i:similar_train_batch_indices})

        self.path=f"{self.diagnosis}/cluster/{clustering_function.__name__}/t{self.timestep}"
        self.save_path=os.path.join(self.path,'train_keys.json')

        if not os.path.exists(self.path): os.makedirs(self.path)
        #json indexes change a lot during rewriting,
        # I think it is ok as filtering is for the first 5 (there are no good and bad.)
        with open(self.save_path,'w') as file:
            json.dump(combined_cluster_indices,file,indent=4)

        return self

    def visualize_clusters(self):
        """
        Visualize test data points with respect to their clusters.
        """
        colors=np.random.rand(len(self.output.cluster_centers_),3)
        clusters=range(len(self.output.cluster_centers_))
        color_map={clusters[i]:colors[i] for i in clusters}
        Y=PCA(n_components=2).fit_transform(self.test_data)
        c_labels=[color_map[i] for i in self.test_clusters]

        ax = plt.axes()
        ax.scatter(Y[:,0],Y[:,1],c=c_labels)
        ax.set_title(f"{self.diagnosis} diagnosis cluster results with PCA: Timestep {self.timestep}")
        plt.show()

        return self

    def __call__(self,clustering_function):
        # data processing
        self.average_feature_time_series().train_test()
        # calculation and saving
        self.perform_clustering(clustering_function=clustering_function,n_clusters=3)
        # for existing cluster methods visualize only for the first timestep
        if self.timestep==1:
            self.visualize_clusters()


class DistributionModel(DistanceModel):
    def calculate_similarity_based_on_distribution(self):
        self.final_tensors=torch.load(f"../datasets/{self.diagnosis}/t{3}/features.pt")
        self.final_tensors=self.final_tensors.mean(dim=1)
        indices=range(self.train_data.shape[0])
        self.final_tensors=self.feature_tensors.index_select(0,torch.IntTensor(indices))

        combined_similarity_scores={}
        for i, data in enumerate(self.test_data):
            similarity_scores={}
            for j,final in enumerate(self.final_tensors):
                # some test scores are completely the same even though tensors are completely different
                # 0,3,8 so on
                test=kstest(data,final)
                similarity_scores.update({j:test.pvalue})

            keys=[t[0] for t in sorted(similarity_scores.items(),key=lambda dict_:dict_[1],reverse=True)]
            keys=keys[:5]
            combined_similarity_scores.update({i:keys})

        self.path=f"{self.diagnosis}/kstest/t{self.timestep}"
        self.save_path=os.path.join(self.path,'train_keys.json')

        if not os.path.exists(self.path): os.makedirs(self.path)
        with open(self.save_path,'w') as file:
            json.dump(combined_similarity_scores,file,indent=4)
        return self

    def __call__(self,):
        self.average_feature_time_series().train_test()
        self.calculate_similarity_based_on_distribution().get_drug_sequences()

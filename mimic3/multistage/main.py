import os
import yaml
from utils import DataConversion
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import FeatureDataset, DrugDataset, Model, MultiStageModel
import matplotlib.pyplot as plt

dir_=os.path.dirname(__file__)
os.chdir(dir_)

with open('../datasets/sqldata/stats.yaml') as stats, open('info.yaml') as info:
    stats=yaml.safe_load(stats)
    info=yaml.safe_load(info)

diagnoses=stats['diagnosis_for_selection']
timesteps=info['timesteps']
for diagnosis in diagnoses:
    features_datasets=[]
    drug_datasets=[]
    for t in timesteps:
        # DataConversion(diagnosis=diagnosis,timestep=t).average_save_feature_time_series().\
        #     convert_drugs_dummy_data_format()

        features_datasets.append(FeatureDataset(diagnosis=diagnosis,timestep=t))
        drug_datasets.append(DrugDataset(diagnosis=diagnosis,timestep=t))

    features_t1,features_t2, features_t3=features_datasets
    drug_t1,drug_t2, drug_t3=drug_datasets
    torch.seed()
    batch_size=30

    loader_features_t1,loader_features_t2, loader_features_t3=map(lambda i:DataLoader(i,batch_size=batch_size), features_datasets)
    loader_drug_t1,loader_drug_t2, loader_drug_t3=map(lambda i:DataLoader(i,batch_size=batch_size), drug_datasets)

    feature_model=Model(912,10)
    drug_model=Model(912,902)
    outpt_model=Model(912,1)

    multistage_model=MultiStageModel(feature_model,drug_model,outpt_model)
    optimizer = torch.optim.SGD(multistage_model.parameters(), lr=0.01)
    criterion= nn.MSELoss()
    epochs=50

    combined_loader=zip(loader_features_t1,loader_drug_t1,loader_features_t2,\
                     loader_drug_t2,loader_features_t3,loader_drug_t3)

    l11_loss,l21_loss,lout_loss=[],[],[]
    l12_loss,l22_loss=[],[]
    combined_loss=[]


    for epoch in range(epochs):
        for i,((x1_t1,y11), (x2_t1,y21),(x1_t2,y12),(x2_t2,y22), (x1_t3,y),(x2_t3,y)) \
                 in enumerate(zip(loader_features_t1,loader_drug_t1,loader_features_t2,\
                     loader_drug_t2,loader_features_t3,loader_drug_t3)):
            feature_Xt2,drug_Xt2,feature_Xt3,drug_Xt3,out=multistage_model(x1_t1,x2_t1)

            l11=criterion(feature_Xt2,y11)
            l12=criterion(drug_Xt2,y21)
            l21=criterion(feature_Xt3,y12)

            #drug_t2
            l22=criterion(drug_Xt3,y22)

            #output
            lout=criterion(out,y.view(-1,1))

            loss=(l11+l12+l21+l22+lout)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #TODO appending per batch does not work for the first diagnosis

        l11_loss.append(round(l11.item(),3))
        l12_loss.append(round(l12.item(),3))
        l21_loss.append(round(l21.item(),3))
        l22_loss.append(round(l22.item(),3))
        lout_loss.append(round(lout.item(),2))
        combined_loss.append(loss.item())

    fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5, sharey=False)
    plt.title(f"Loss Curves over 3 staged for {diagnosis} diagonis")
    ax1.plot(l11_loss)
    ax1.set_title('T1 Features')
    ax2.plot(l12_loss)
    ax2.set_title('T1 Drugs')
    ax3.plot(l21_loss)
    ax3.set_title('T2 Features')
    ax4.plot(l22_loss)
    ax4.set_title('T2 Drugs')
    ax5.plot(lout_loss)
    ax5.set_title('Output')

    fig.supxlabel('Epochs')
    fig.supylabel('MSE Loss per Epoch')
    fig.suptitle(f"{diagnosis}")
    plt.show()

#TODO train-test
#TODO experimentation with hidden layers
#TODO better increase in learning
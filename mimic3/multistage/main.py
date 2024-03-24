import os
import sys
import yaml
from utils import DataConversion,reset_weights, balance_datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import FeatureDataset, DrugDataset, Model, MultiStageModel
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

dir_=os.path.dirname(__file__)
os.chdir(dir_)

parent_parent_path=os.path.dirname(dir_)
sys.path.append(parent_parent_path)
from helpers import configure_logger

torch.set_printoptions(precision=3)

path=os.path.dirname(__file__)
logger=configure_logger(default=False,path=path)

with open('../datasets/sqldata/stats.yaml') as stats, open('info.yaml') as info:
    stats=yaml.safe_load(stats)
    info=yaml.safe_load(info)

diagnoses=stats['diagnosis_for_selection']
timesteps=info['timesteps']
for diagnosis in ['DIABETIC KETOACIDOSIS']:
    features_datasets=[]
    drug_datasets=[]
    for t in timesteps:
        # DataConversion(diagnosis=diagnosis,timestep=t).average_save_feature_time_series().\
        #     convert_drugs_dummy_data_format()

        features_datasets.append(FeatureDataset(diagnosis=diagnosis,timestep=t))
        drug_datasets.append(DrugDataset(diagnosis=diagnosis,timestep=t))

    output=features_datasets[2].Y
    features_t1,features_t2, features_t3=balance_datasets(features_datasets,output)
    drug_t1,drug_t2, drug_t3=balance_datasets(drug_datasets,output)

    torch.seed()
    batch_size=100
    k_folds=5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    f1_drugs_t2_list=[]
    f1_drugs_t3_list=[]
    f1_output_list=[]
    logger.info(f"{diagnosis}\n")
    for fold, (train_ids, test_ids) in enumerate(kfold.split(features_t1)):
        print(f"Diagnosis {diagnosis}.")
        print(f"Fold {fold}")
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        #train loaders
        loader_features_t1,loader_features_t2, loader_features_t3=\
            map(lambda i:DataLoader(i,batch_size=batch_size,sampler=train_subsampler), features_datasets)
        loader_drug_t1,loader_drug_t2, loader_drug_t3=\
            map(lambda i:DataLoader(i,batch_size=batch_size,sampler=train_subsampler), drug_datasets)

        #test loaders
        test_loader_features_t1,test_loader_features_t2, test_loader_features_t3=\
            map(lambda i:DataLoader(i,batch_size=batch_size,sampler=test_subsampler), features_datasets)
        test_loader_drug_t1,test_loader_drug_t2, test_loader_drug_t3=\
            map(lambda i:DataLoader(i,batch_size=batch_size,sampler=test_subsampler), drug_datasets)

        feature_model=Model(912,10)
        drug_model=Model(912,902)
        outpt_model=Model(912,1)

        multistage_model=MultiStageModel(feature_model,drug_model,outpt_model)
        multistage_model.apply(reset_weights)
        optimizer = torch.optim.Adamax(multistage_model.parameters(), lr=0.01)
        criterion= nn.MSELoss()
        epochs=50

        combined_loader=zip(loader_features_t1,loader_drug_t1,loader_features_t2,\
                        loader_drug_t2,loader_features_t3,loader_drug_t3)

        #batch
        drugs_t2_loss_list=[]
        drugs_t3_loss_list=[]
        output_loss_list=[]

        for epoch in range(epochs):
            for i,((features_t1,_),
                   (drugs_t1,drugs_t2),
                   (_,drugs_t3),
                   (_,output)) in enumerate(zip(
                                    loader_features_t1,
                                    loader_drug_t1,
                                    loader_drug_t2,
                                    loader_features_t3)):
                _,drugs_t2_pred,_,drugs_t3_pred,output_pred=multistage_model(features_t1,drugs_t1)

                # t2 prediction
                drugs_t2_loss=criterion(drugs_t2_pred,drugs_t2)
                # t3 prediction
                drugs_t3_loss=criterion(drugs_t3_pred,drugs_t3)
                # #output prediction
                output_loss=criterion(output,output_pred.view(-1,1))

                loss=(drugs_t2_loss+drugs_t3_loss+output_loss)/batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                drugs_t2_loss_list.append(round(drugs_t2_loss.item(),3))
                drugs_t3_loss_list.append(round(drugs_t3_loss.item(),3))
                output_loss_list.append(round(output_loss.item(),3))

        fig, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=False)
        ax1.plot(drugs_t2_loss_list)
        ax1.set_title('T1-T2 Drugs')
        ax2.plot(drugs_t3_loss_list)
        ax2.set_title('T2-T3 Drugs')
        ax3.plot(output_loss_list)
        ax3.set_title('Output')

        fig.supxlabel('Batches')
        fig.supylabel('MSE Loss per Batch')
        fig.suptitle(f"{diagnosis}")
        # plt.show()

        # Evaluation for this fold
        with torch.no_grad():
            for i,((features_t1,_),
                (drugs_t1,drugs_t2),
                (_,drugs_t3),
                (_,output)) in enumerate(zip(
                                    test_loader_features_t1,
                                    test_loader_drug_t1,
                                    test_loader_drug_t2,
                                    test_loader_features_t3)):
                _,drugs_t2_pred,_,drugs_t3_pred,output_pred=multistage_model(features_t1,drugs_t1)

                #drug_accuracy
                # #NOTE, even though it has 902 columns, still it is okay to keep the threshhold to 1/2
                drugs_t2_pred=(drugs_t2_pred>(1/2)).float()
                drugs_t3_pred=(drugs_t2_pred>(1/2)).float()
                f1_drugs_t2_list.append(f1_score(drugs_t2_pred.detach(), drugs_t2, average="samples"))
                f1_drugs_t3_list.append(f1_score(drugs_t3_pred.detach(), drugs_t3, average="samples"))

                #output
                pred = ((output_pred.data>1/2).flatten()).float()
                f1_output_list.append(f1_score(pred.detach(), output.flatten(), average="binary"))

    f1_t2_drugs_mean=round(sum(f1_drugs_t2_list)/len(f1_drugs_t2_list),2)
    f1_t3_drugs_mean=round(sum(f1_drugs_t3_list)/len(f1_drugs_t3_list),2)
    f1_output_mean=round(sum(f1_output_list)/len(f1_output_list),2)
    print("t2 drugs f1 score from folds:",f1_t2_drugs_mean)
    print("t3 drugs f1 score from folds:",f1_t3_drugs_mean)
    print("output f1 score from folds:",f1_output_mean)

    logger.info("t2 drugs f1 score from folds:{}".format(f1_t2_drugs_mean))
    logger.info("t3 drugs f1 score from folds:{}".format(f1_t3_drugs_mean))
    logger.info("output f1 score from folds:{}".format(f1_output_mean))

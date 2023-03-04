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
from sklearn.metrics import recall_score,f1_score

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
    drugs_t2_fold_accuracies=[]
    drugs_t2_fold_recall=[]
    drugs_t2_fold_f1=[]
    drugs_t3_fold_accuracies=[]
    drugs_t3_fold_recall=[]
    drugs_t3_fold_f1=[]
    output_fold_accuracies=[]
    recall_fold=[]
    f1_score_fold=[]

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
        plt.show()

        # Evaluation for this fold
        drug_t2_accuracy_list=[]
        drug_t2_recall_list=[]
        drug_t2_f1_list=[]
        drug_t3_accuracy_list=[]
        drug_t3_recall_list=[]
        drug_t3_f1_list=[]
        output_accuracy_list=[]
        recall_list=[]
        f1_score_list=[]
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
                n_cols=drugs_t2.shape[1]
                drugs_t2_pred=(drugs_t2_pred>(1/2)).float()
                drugs_t3_pred=(drugs_t2_pred>(1/2)).float()

                # number of correct per row divided by total row length, then averaged across columns
                drugs_t2_accuracy=torch.mean(torch.sum(drugs_t2_pred.detach()==drugs_t2,dim=1)/drugs_t2.shape[1]).item()
                drug_t2_accuracy_list.append(drugs_t2_accuracy)

                #NOTE for drugs, recall is high, wherever it is 1 it predicts correctly, but wherever it is 0 does not
                # always predict correctly, meaning it assigns more drugs than needed.
                recall_t2_drugs=recall_score(drugs_t2,drugs_t2_pred,average='samples')
                drug_t2_recall_list.append(recall_t2_drugs)
                f1_t2_drugs=f1_score(drugs_t2,drugs_t2_pred,average='samples')
                drug_t2_f1_list.append(f1_t2_drugs)

                drugs_t3_accuracy=torch.mean(torch.sum(drugs_t3_pred.detach()==drugs_t3,dim=1)/drugs_t3.shape[1]).item()
                drug_t3_accuracy_list.append(drugs_t3_accuracy)

                recall_t3_drugs=recall_score(drugs_t3,drugs_t3_pred,average='samples')
                drug_t3_recall_list.append(recall_t3_drugs)
                f1_t3_drugs=f1_score(drugs_t3,drugs_t3_pred,average='samples')
                drug_t3_f1_list.append(f1_t3_drugs)

                #output
                pred = ((output_pred.data>0.5).flatten()).float()
                number_of_1s_actual=sum(output.flatten()==1)
                number_of_1s_pred=sum(pred==1)
                logger.info(f"In fold {fold}, loader {i}, there are {number_of_1s_actual} 1s in actual.")
                logger.info(f"In fold {fold}, loader {i}, there are {number_of_1s_pred} 1s in pred.")

                output_accuracy = (torch.sum(pred == output).item())/output.shape[0]
                output_accuracy_list.append(output_accuracy)
                recall=recall_score(output,pred)
                recall_list.append(recall)
                f1=f1_score(output,pred)
                f1_score_list.append(f1)

        #drug mean across batches
        drugs_t2_mean_accuracy=round(100*sum(drug_t2_accuracy_list)/len(drug_t2_accuracy_list),3)
        drugs_t2_mean_recall=round(100*sum(drug_t2_recall_list)/len(drug_t2_recall_list),3)
        drugs_t2_mean_f1=round(100*sum(drug_t2_f1_list)/len(drug_t2_f1_list),3)

        drugs_t3_mean_accuracy=round(100*sum(drug_t3_accuracy_list)/len(drug_t3_accuracy_list),3)
        drugs_t3_mean_recall=round(100*sum(drug_t3_recall_list)/len(drug_t3_recall_list),3)
        drugs_t3_mean_f1=round(100*sum(drug_t3_f1_list)/len(drug_t3_f1_list),3)

        #output
        output_mean_accuracy=round(100*sum(output_accuracy_list)/len(output_accuracy_list),3)
        output_mean_recall=round(100*sum(recall_list)/len(recall_list),3)
        output_mean_f1=round(100*sum(f1_score_list)/len(f1_score_list),3)

        logger.info(f"fold:{fold}, drugs t2 mean accuracy: {drugs_t2_mean_accuracy}%")
        logger.info(f"fold:{fold}, drugs t2 mean recall: {drugs_t2_mean_recall}%")
        logger.info(f"fold:{fold}, drugs t2 mean f1: {drugs_t2_mean_f1}%")

        logger.info(f"fold:{fold}, drugs t3 mean accuracy: {drugs_t3_mean_accuracy}%")
        logger.info(f"fold:{fold}, drugs t3 mean recall: {drugs_t3_mean_recall}%")
        logger.info(f"fold:{fold}, drugs t3 mean f1: {drugs_t3_mean_f1}%")

        logger.info(f"fold:{fold}, output mean accuracy:{output_mean_accuracy}%")
        logger.info(f"fold:{fold}, mean recall:{output_mean_recall}%")
        logger.info(f"fold:{fold}, mean f1:{output_mean_f1}%\n")

        # store accuracies per fold
        drugs_t2_fold_accuracies.append(drugs_t2_mean_accuracy)
        drugs_t2_fold_recall.append(drugs_t2_mean_recall)
        drugs_t2_fold_f1.append(drugs_t2_mean_f1)

        drugs_t3_fold_accuracies.append(drugs_t3_mean_accuracy)
        drugs_t3_fold_recall.append(drugs_t3_mean_recall)
        drugs_t3_fold_f1.append(drugs_t3_mean_f1)

        output_fold_accuracies.append(output_mean_accuracy)
        recall_fold.append(output_mean_recall)
        f1_score_fold.append(output_mean_f1)


    logger.info(f"drugs t2 accuracy (folds average): {sum(drugs_t2_fold_accuracies)/len(drugs_t2_fold_accuracies)}%")
    logger.info(f"drugs t2 recall (folds average): {sum(drugs_t2_fold_recall)/len(drugs_t2_fold_recall)}%")
    logger.info(f"drugs t2 f1-score (folds average): {sum(drugs_t2_fold_f1)/len(drugs_t2_fold_f1)}%")

    logger.info(f"drugs t3 accuracy (folds average): {sum(drugs_t3_fold_accuracies)/len(drugs_t3_fold_accuracies)}%")
    logger.info(f"drugs t3 recall (folds average): {sum(drugs_t3_fold_recall)/len(drugs_t3_fold_recall)}%")
    logger.info(f"drugs t3 f1-score (folds average): {sum(drugs_t3_fold_f1)/len(drugs_t3_fold_f1)}%")

    logger.info(f"output accuracy (folds average): {sum(output_fold_accuracies)/len(output_fold_accuracies)}%")
    logger.info(f"recall (folds average): {sum(recall_fold)/len(recall_fold)}%")
    logger.info(f"f1-score (folds average): {sum(f1_score_fold)/len(f1_score_fold)}%")

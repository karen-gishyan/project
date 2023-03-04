import os
import torch
import numpy as np
from itertools import product
from torch.nn.functional import pad
from torch.utils.data import DataLoader,SubsetRandomSampler
import torch.nn as nn
import matplotlib.pyplot as plt
from evaluate import DataSet, EvaluationModel
from helpers import configure_logger
from sklearn.model_selection import KFold
from multistage.utils import reset_weights



path=os.path.dirname(__file__)
logger=configure_logger(default=False,path=path)


def remove_file_from_subdirectories(file_name):
    """
    Find the file in the subdirectories of the root folder.
    Takes user input before deleting the file.
    """
    dir_=os.path.dirname(__file__)
    os.chdir(dir_)
    for root,dirs, files in os.walk("."):
        for file in files:
            if file==file_name:
                path=os.path.join(root,file)
                print(path)
                user_input=input("should the file be removed?:")
                if str(user_input)=='yes':
                    os.remove(path)

def combine_drug_sequences(diagnosis,dir_name,method=None):
    """
     For each testing instance we generate 125 drug sequences.
    """
    cd=os.getcwd()
    sequences=[]
    for t in [1,2,3]:
        if method:
            path=os.path.join(cd,diagnosis,dir_name,method,f"t{t}","drug_sequences.pt")
        else:
            path=os.path.join(cd,diagnosis,dir_name,f"t{t}","drug_sequences.pt")
        tensor=torch.load(path)
        sequences.append(tensor)

    row_shapes=[i.shape[1] for i in sequences]
    col_shapes=[i.shape[2] for i in sequences]
    max_rows,max_cols=max(row_shapes),max(col_shapes)
    # drugs in the same t are padded to have the same number of rows and columns,
    # however both rows and columns can differ across ts depending on the method.
    # we thus pad with -1 the difference between row and maximum row, col and max col.
    for i in range(len(sequences)):
        n_rows=sequences[i].shape[1]
        n_cols=sequences[i].shape[2]
        diff_rows=max_rows-n_rows
        diff_cols=max_cols-n_cols
        # padding tuple logic is left,right, up, down, thus we pad right and down
        sequences[i]=pad(sequences[i],(0,diff_cols,0,diff_rows),value=-1)

    t1_sequence,t2_sequence,t3_sequence=sequences
    batch_size=t1_sequence.shape[0]
    # should be the same for all columns
    n_cols=t1_sequence.shape[2]
    combinations=[]
    for batch in range(batch_size):
        concat=list(product(t1_sequence[batch].tolist(), t2_sequence[batch].tolist(),t3_sequence[batch].tolist()))
        concat=torch.Tensor(np.array(concat).reshape((-1,n_cols*3)))
        combinations.append(concat)
    final_tensor=torch.cat((*combinations,),dim=0)
    if method:
        torch.save(final_tensor,os.path.join(cd,diagnosis,dir_name,method,"combined_drugs.pt"))
    else:
        torch.save(final_tensor,os.path.join(cd,diagnosis,dir_name,"combined_drugs.pt"))


    # increase the amount of test size based on the expansion of product
    test_output=torch.load(f'{diagnosis}/test_output.pt')
    expand_count=final_tensor.shape[0]/test_output.shape[0]
    test_output_expanded=torch.Tensor(np.repeat(test_output.numpy(),expand_count))
    #test output is the same for all methods, but test_output_expanded can differ
    # across methods
    if method:
         path=f'{diagnosis}/{dir_name}/{method}/test_output_expanded.pt'
    else:
        path=f'{diagnosis}/{dir_name}/test_output_expanded.pt'
    assert final_tensor.shape[0]==test_output_expanded.shape[0], \
     f"Number of rows do not match {diagnosis},{dir_name},{method}"
    torch.save(test_output_expanded,path)

def train_individual(diagnosis, dirname,method=None):
    dataset=DataSet(diagnosis,dirname,method)
    print(torch.sum(dataset.output_tensor==1))
    input_size=dataset.drug_tensor.shape[1]
    output_size=dataset.output_tensor.shape[1]
    batch_size=100
    k_folds=5
    kfold = KFold(n_splits=k_folds, shuffle=False)
    folds_accuracy_list=[]
    # 0-500, 500-1000, ...,2000-2500 ids are taken as test_ids for each fold
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"Diagnosis {diagnosis}.")
        print(f"Fold {fold}")
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        train_loader=DataLoader(dataset,batch_size=batch_size,sampler=train_subsampler)
        test_loader=DataLoader(dataset,batch_size=batch_size,sampler=test_subsampler)
        model=EvaluationModel(input_size=input_size,output_size=output_size)
        model.apply(reset_weights)
        criterion= nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        torch.seed()
        epochs=50
        total_loss=[]
        for _ in range(epochs):
            for _, (x,y) in enumerate(train_loader):
                y_pred=model(x.float())
                loss=criterion(y_pred.float(),y.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss.append(loss.item())

        plt.plot(total_loss,label=f"fold:{fold+1}")
        title=f"Diagnosis:{diagnosis}, Type:{dirname}, Method:{method}" if method else \
            f"Diagnosis:{diagnosis}, Type:{dirname}"
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend(loc="upper right")

        #pred
        output_accuracy_list=[]
        logger.info(f"{diagnosis},{dirname},{method}")
        with torch.no_grad():
            for i, (x,y) in enumerate(test_loader):
                output_pred=model(x)
                pred = ((output_pred.data>0.5).flatten()).float()
                number_of_1s_actual=sum(y.flatten()==1)
                number_of_1s_pred=sum(pred==1)
                logger.info(f"In fold {fold}, loader {i}, there are {number_of_1s_actual} 1s in actual.")
                logger.info(f"In fold {fold}, loader {i}, there are {number_of_1s_pred} 1s in pred.")
                output_accuracy = (torch.sum(pred == y.flatten()).item())/y.shape[0]
                output_accuracy_list.append(output_accuracy)

        mean_accuracy_across_batches=sum(output_accuracy_list)/len(output_accuracy_list)
        print(f"Mean Batch Accuracy {mean_accuracy_across_batches}")
        folds_accuracy_list.append(mean_accuracy_across_batches)

    plt.show()
    folds_average=sum(folds_accuracy_list)/len(folds_accuracy_list)
    logger.info(f"{diagnosis},{dirname},{method}\n")
    logger.info(f"output accuracy (folds average): {folds_average}\n")

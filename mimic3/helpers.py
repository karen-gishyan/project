import logging
import torch
import json
from torch.utils.data import DataLoader
from numpy import unique
from datasets import SplitRNNData


def configure_logger():
    logger = logging.getLogger('mimic3')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('../info.log',mode='a')
    # file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def create_split_loaders(**kwargs):
    """
    Split a feauture/drugs dataloader for timestep t into train,test, valid sets.
    kwargs of SplitRNNData  and batch size of DataLoader construct are expected.
    """
    split=['train','test','valid']
    loaders=[]
    for i in split:
        data=SplitRNNData(is_feature=kwargs['is_feature'],
                                    timestep=kwargs['timestep'],
                                    split_size=kwargs.get('split_size'),
                                    split=i)
        loaders.append(DataLoader(data,batch_size=kwargs['batch_size']))
    return loaders

def accuracy(pred_y,y,feature=False):
    if not feature:
        pred_y_max=torch.argmax(pred_y,dim=2)
        y_max=torch.argmax(y,dim=2)
        return (pred_y_max==y_max).numpy().mean()
    else:
        n_columns=pred_y.shape[2]
        accuracy_per_feature={}
        for col in range(n_columns):
            accuracy=torch.mean((pred_y[:,:,col]-y[:,:,col])**2)
            accuracy_per_feature.update({f"Col{col+1}":accuracy})
        total_accuracy=torch.mean((pred_y-y)**2)
        mean_col_accuracy={"Column mean accuracy":total_accuracy}
        return accuracy_per_feature,mean_col_accuracy



def pred_to_labels(tensor,drugs=True):
    if drugs:
        with open("../json/drug_mapping.json",'r')as file:
            mappings=json.load(file)
    else:
        with open("../json/discharge_location_mapping.json",'r')as file:
            mappings=json.load(file)

    key_list = list(mappings.keys())
    val_list = list(mappings.values())

    res=list(map(lambda i:key_list[val_list.index(i)],tensor))
    return res

def count_uniques_in_pred_and_output(pred_tensor,y_tensor):
    """
    Return dictionary of unique values and the number of unique occurences of each
    for prediction and output tensors.
    """
    vals,counts=unique(pred_tensor.detach().numpy(),return_counts=True)
    pred_counts=dict(zip(vals,counts))

    vals,counts=unique(y_tensor.numpy(),return_counts=True)
    actual_counts=dict(zip(vals,counts))

    return pred_counts, actual_counts


def convert_drugs_dummy_data_format(drug_tensor):
    with open("../json/drug_mapping.json") as file:
        drug_mappings=json.load(file)

    len_total_drugs=len(list(drug_mappings))
    dummy_format=torch.zeros(drug_tensor.shape[0],len_total_drugs)

    # for a given rows 1 will indicate that drug was given, else we leave as 0
    for i, row in enumerate(drug_tensor):
        for drug_index in row:
            if drug_index!=-1:
                dummy_format[i][drug_index.long()]=1

    return dummy_format
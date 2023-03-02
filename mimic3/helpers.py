import os
import logging
import torch
import json
from numpy import unique


def configure_logger(default=True,path=None):
    logger = logging.getLogger('mimic3')
    logger.setLevel(logging.INFO)
    if default:
        file_handler = logging.FileHandler('../info.log',mode='w')
    else:
        if path:
            file_handler = logging.FileHandler(os.path.join(path,'info.log'),mode='w')
        else:
            file_handler = logging.FileHandler('info.log',mode='w')
    # file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

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
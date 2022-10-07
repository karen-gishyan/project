import logging
from torch.utils.data import DataLoader
from datasets import SplitRNNData


def configure_logger():
    logger = logging.getLogger('mimic3')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('../info.log',mode='w')
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


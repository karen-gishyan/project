from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from models import MultiClassLogisticRegression
import numpy as np
import json
import logging

logger=logging.getLogger('comparison_logger')
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(filename="project/mimic3/optim/results/comparisons.log",mode='w'))


def remove_nan(X,y):
    temp=X.isna().any(axis=1)
    drop_indexes=temp[temp].index
    X.drop(index=drop_indexes,inplace=True)
    y.drop(index=drop_indexes,inplace=True)
    return X,y


def evaluate(X_train, X_test, y_train, y_test,save_path=None,scale=True):
    """For no scaling, use lr parameter instead of scaling_factor."""
    if not save_path:raise ValueError("save path is not provided")
    step_size=int(len(X_train)/10)
    batches=list(range(step_size,len(X_train),step_size))
    # scaling factors the same as learning rates
    scaling_factors=[0.0001,0.001,0.01,0.05,0.1]
    lr=MultiClassLogisticRegression()
    save_list=[]
    for batch_size in batches:
        for scaling_factor in scaling_factors:
            if scale:
                lr.fit(X_train,y_train,batch_size=batch_size,scaling_factor=scaling_factor)
            else:
                lr.fit(X_train,y_train,batch_size=batch_size,lr=scaling_factor)
            accuracy=round(lr.score(X_test,y_test),2)
            p,r,f1,_=precision_recall_fscore_support(y_test,lr.predict_classes(X_test),average='macro')
            custom_scd_dict={}
            custom_scd_dict.update({
                "name":"custom_scd",
                "batch_size":batch_size,
                "scaling_factor":scaling_factor,
                "accuracy":accuracy,
                "precision":p,
                "recall":r,
                "f1":f1
            })
            save_list.append(custom_scd_dict)

    with open(save_path,'w') as file:
        json.dump(save_list,file,indent=4)


def evaluate_sklearn(X_train, X_test, y_train, y_test,save_path=None,n_iter=300):
    if not save_path:raise ValueError("save path is not provided")
    step_size=int(len(X_train)/10)
    batches=list(range(step_size,len(X_train),step_size))
    classes = np.unique(y_train)
    # scaling factors the same as learning rates
    scaling_factors=[0.0001,0.001,0.01,0.05,0.1]
    save_list=[]
    for batch_size in batches:
        for scaling_factor in scaling_factors:
            scd_dict={}
            for _ in range(n_iter):
                clf=SGDClassifier(loss='log_loss',eta0=scaling_factor,learning_rate='constant')
                idx = np.random.choice(X_train.to_numpy().shape[0], batch_size)
                X_batch, y_batch = X_train.to_numpy()[idx], y_train.to_numpy()[idx]
                clf.partial_fit(X_batch,y_batch,classes)

            accuracy=round(clf.score(X_test,y_test),2)
            p,r,f1,_=precision_recall_fscore_support(y_test,clf.predict(X_test),average='macro')
            scd_dict.update({
                "name":"scd",
                "batch_size":batch_size,
                "scaling_factor":scaling_factor,
                "accuracy":accuracy,
                "precision":p,
                "recall":r,
                "f1":f1
            })
            save_list.append(scd_dict)

    with open(save_path,'w') as file:
        json.dump(save_list,file,indent=4)


def compare(path1,path2,path3):
    """Direct comparison. """
    with open(path1) as file, open(path2)as file_no_scale,open(path3) as file_sklearn:
        save_list=json.load(file)
        save_list_no_scale=json.load(file_no_scale)
        save_list_sklearn=json.load(file_sklearn)

    assert len(save_list)==len(save_list_no_scale)==len(save_list_sklearn),"Lengths are not equal"
    logger.info(path1)
    #custom scd with scaling
    res_f1=max(save_list,key=lambda d:d['f1'])
    logger.info("Max f1 scaled:{},batch_size:{},scaling_factor:{}".\
        format(res_f1['f1'],res_f1['batch_size'],res_f1['scaling_factor']))
    res_accuracy=max(save_list,key=lambda d:d['accuracy'])
    logger.info("Max accuracy scaled:{},batch_size:{},scaling_factor:{}".\
        format(res_accuracy['accuracy'],res_accuracy['batch_size'],res_accuracy['scaling_factor']))
    average_f1=sum([d['f1'] for d in save_list])/len(save_list)
    logger.info("Mean f1 scaled:{}".format(average_f1))
    average_accuracy=sum([d['accuracy'] for d in save_list])/len(save_list)
    logger.info("Mean accuracy scaled:{}\n".format(average_accuracy))

    #custom scd without scaling
    res_f1=max(save_list_no_scale,key=lambda d:d['f1'])
    logger.info("Max f1 not scaled:{},batch_size:{},scaling_factor:{}".\
        format(res_f1['f1'],res_f1['batch_size'],res_f1['scaling_factor']))
    res_accuracy=max(save_list_no_scale,key=lambda d:d['accuracy'])
    logger.info("Max accuracy not scaled:{},batch_size:{},scaling_factor:{}".\
        format(res_accuracy['accuracy'],res_accuracy['batch_size'],res_accuracy['scaling_factor']))
    average_f1=sum([d['f1'] for d in save_list_no_scale])/len(save_list_no_scale)
    logger.info("Mean f1 not scaled:{}".format(average_f1))
    average_accuracy=sum([d['accuracy'] for d in save_list_no_scale])/len(save_list_no_scale)
    logger.info("Mean accuracy not scaled:{}\n".format(average_accuracy))

    #scd
    res_f1=max(save_list_sklearn,key=lambda d:d['f1'])
    logger.info("Max f1 sklearn:{},batch_size:{},scaling_factor:{}".\
        format(res_f1['f1'],res_f1['batch_size'],res_f1['scaling_factor']))
    res_accuracy=max(save_list_sklearn,key=lambda d:d['accuracy'])
    logger.info("Max accuracy sklearn:{},batch_size:{},scaling_factor:{}".\
        format(res_accuracy['accuracy'],res_accuracy['batch_size'],res_accuracy['scaling_factor']))
    average_f1=sum([d['f1'] for d in save_list_sklearn])/len(save_list_sklearn)
    logger.info("Mean f1 sklearn:{}".format(average_f1))
    average_accuracy=sum([d['accuracy'] for d in save_list_sklearn])/len(save_list_sklearn)
    logger.info("Mean accuracy sklearn:{}\n".format(average_accuracy))

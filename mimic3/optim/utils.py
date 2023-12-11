from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from models import MultiClassLogisticRegression
import logging
import numpy as np

logger=logging.getLogger('results_logger')
logger.setLevel(logging.INFO)


def remove_nan(X,y):
    temp=X.isna().any(axis=1)
    drop_indexes=temp[temp].index
    X.drop(index=drop_indexes,inplace=True)
    y.drop(index=drop_indexes,inplace=True)
    return X,y


def evaluate(X_train, X_test, y_train, y_test,log_path=None):
    if not log_path:raise ValueError("Log path is not provided")
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(filename=log_path,mode='w'))

    step_size=int(len(X_train)/10)
    batches=list(range(step_size,len(X_train),step_size))
    # scaling factors the same as learning rates
    scaling_factors=[0.001,0.001,0.01,0.05,0.1]
    lr=MultiClassLogisticRegression()
    for batch_size in batches:
        for scaling_factor in scaling_factors:
            lr.fit(X_train,y_train,batch_size=batch_size,scaling_factor=scaling_factor)
            logger.info("Custom SCD")
            logger.info("Batch size: {},scaling_facor: {}".format(batch_size,scaling_factor))
            logger.info("Accuracy: {}".format(lr.score(X_test,y_test),2))
            p,r,f,_=precision_recall_fscore_support(y_test,lr.predict_classes(X_test),average='macro')
            logger.info("Precision: {},Recall:{}, F1:{}\n".format(p,r,f))

            clf=SGDClassifier(loss='log_loss',alpha=scaling_factor)
            idx = np.random.choice(X_train.to_numpy().shape[0], batch_size)
            X_batch, y_batch = X_train.to_numpy()[idx], y_train.to_numpy()[idx]
            classes = np.unique(y_train)
            clf.partial_fit(X_batch,y_batch,classes)
            logger.info("SCD")
            logger.info("Batch size: {},scaling_facor: {}".format(batch_size,scaling_factor))
            logger.info("Accuracy: {}".format(clf.score(X_test,y_test),2))
            p,r,f,_=precision_recall_fscore_support(y_test,clf.predict(X_test),average='macro')
            logger.info("Precision: {},Recall:{}, F1:{}\n".format(p,r,f))

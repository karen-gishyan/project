import numpy as np


class MultiClassLogisticRegression:
    """Main Training Steps:
        Linear Prediction -> Softmax Activation -> Cross Entropy Calculation -> Derivative calculation -> Update
        Sources:https://github.com/bamtak/machine-learning-implemetation-python/blob/master/Multi%20Class%20Logistic%20Regression.ipynb
    """
    def __init__(self,n_iter = 1000, threshold=1e-3):
        self.n_iter = n_iter
        self.threshold = threshold

    def fit(self,X, y, batch_size=10, lr=0.001, random_seed=4):
        """Model training."""
        np.random.seed(random_seed)
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        y = self.one_hot(y)
        X=X.to_numpy()
        self.loss = []
        self.bias = np.zeros((1, len(self.classes)))
        self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
        i = 0
        while (not self.n_iter or i < self.n_iter):
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            y_pred=self.predict(X_batch)
            self.loss.append(self.cross_entropy(y_batch,y_pred))
            # update
            dweight,dbias=self.get_gradients(y_batch,y_pred,X_batch)
            self.weights-=lr*dweight
            self.bias-=lr*dbias
            if np.abs(dweight).max() < self.threshold:
                break
            if i % 100 == 0:
                print(' Training Accuray at {} iterations is {}'.format(i, self.evaluate(X, y)))
            i +=1

    def one_hot(self, y):
        """Dummy conversion."""
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    def predict(self, X):
        """Linear prediction, followed by softmax."""
        y_pred = np.dot(X, self.weights.T)+self.bias
        return self.softmax(y_pred)

    def softmax(self,z):
        """Softmax activation."""
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

    def get_gradients(self,y,y_pred,X_batch):
        """ Calculates gradients of the loss function with respect to weights and bias."""
        dweight=-1* np.dot((y-y_pred).T,X_batch)/len(X_batch)
        dbias=-1* np.sum((y-y_pred),axis=0)/len(X_batch)
        return dweight,dbias

    def cross_entropy(self, y, y_pred):
        """Calculates cross_entropy loss."""
        # clip to avoid log of 0
        y_pred = np.clip(y_pred, 0.0000001, 1 - 0.0000001)
        ce_loss= -1 *np.sum(np.multiply(y,np.log(y_pred)))/len(y)
        return ce_loss

    def predict_classes(self, X):
        """Converts probability predictions to classes."""
        self.pred = self.predict(X)
        # vectorization is needed because for some datasets labels start from 1, instead of 0
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.pred, axis=1))

    def evaluate(self, X, y):
        """Training evaluation. y is one-hot encoded format."""
        return np.mean(self.predict_classes(X) == np.argmax(y, axis=1))

    def score(self, X_test, y_test):
        """Testing evaluation. y is a 1d vector."""
        X_test=X_test.to_numpy()
        y_test=y_test.to_numpy().flatten()
        return np.mean(self.predict_classes(X_test) ==y_test)

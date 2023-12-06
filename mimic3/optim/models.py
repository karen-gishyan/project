import numpy as np


#https://github.com/bamtak/machine-learning-implemetation-python/blob/master/Multi%20Class%20Logistic%20Regression.ipynb
class MultiClassLogisticRegression:
    """Main Training Steps:
        Linear Prediction -> Softmax Activation -> Cross Entropy Calculation -> Derivative calculation -> Update
    """
    def __init__(self,n_iter = 10000, threshold=1e-3):
            self.n_iter = n_iter
            self.threshold = threshold

    def fit(self,X, y, batch_size=1, lr=0.001, random_seed=4, verbose=False):
        np.random.seed(random_seed)
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
        i = 0
        while (not self.n_iter or i < self.n_iter):
            self.loss.append(self.cross_entropy(y, self.predict_(X)))
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.predict_(X_batch)
            update=lr*self.get_gradient(error,X_batch)
            self.weights-=update
            if np.abs(update).max() < self.threshold:
                break
            if i % 1000 == 0 and verbose:
                print(' Training Accuray at {} iterations is {}'.format(i, self.evaluate_(X, y)))
            i +=1

    def add_bias(self,X):
        return np.c_[X,np.ones(len(X))]

    def one_hot(self, y):
        """ Dummpy conversion."""
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    def predict_(self, X):
        """Linear prediction"""
        pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
        return self.softmax(pre_vals)

    def softmax(self,z):
        """Activation"""
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

    def cross_entropy(self, y, probs):
        """Loss function"""
        #TODO check implementation
        return -1 * np.mean(y * np.log(probs))

    def get_gradient(self,error,X_batch):
        return -1* np.dot(error.T,X_batch)/len(X_batch)

    def evaluate_(self, X, y):
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))

    def predict(self, X):
        return self.predict_(self.add_bias(X))

    def predict_classes(self, X):
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))

    def score(self, X, y):
        X=X.to_numpy()
        y=y.to_numpy().flatten()
        return np.mean(self.predict_classes(X) == y)

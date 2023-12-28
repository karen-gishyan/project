import numpy as np
import matplotlib.pyplot as plt



class MultiClassLogisticRegression:
    """Main Training Steps:
        Linear Prediction -> Softmax Activation -> Cross Entropy Calculation -> Derivative calculation -> Update
        Sources:https://github.com/bamtak/machine-learning-implemetation-python/blob/master/Multi%20Class%20Logistic%20Regression.ipynb
    """
    def __init__(self,n_iter = 500, threshold=0.02):
        self.n_iter = n_iter
        self.threshold = threshold

    def fit(self,X, y,lr=None,scaling_factor=None, random_seed=4,batch_size=1,n_epochs=1,momentum=0.9,
            nesterov=None,standard=True):
        """Model training.
        lr is the default update rate.
        scaling_factor is custom update.

        nesterov=True,standard=False, runs nesterov
        nesterov=False,standard=False, runs momentum
        nesterov=None, standard=True, runs standard

        """
        if nesterov is None:
            assert standard==True, "standard should be True if nesterov is False."

        if nesterov in [True,False]:
            assert standard==False, "standar cannot be True if nesterov is either True or False."

        #NOTE put nesterov=False to run with momentum only
        if not any([lr,scaling_factor]):
            raise ValueError("Either learning rate or scaling factor should be provided.")
        np.random.seed(random_seed)
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}
        y = self.one_hot(y)
        self.mean=X.mean(axis=0).reshape(-1,1)
        #NOTE normalization does not help
        self.loss = []
        self.bias = np.zeros((1, len(self.classes)))
        self.weights = np.zeros(shape=(len(self.classes),X.shape[1]))
        self.velocity_w=np.zeros(shape=(len(self.classes),X.shape[1]))
        self.velocity_b=0
        #stachastic and minibatch gd are supported
        min_cosine=1
        for _ in range(n_epochs):
            for i in range(0,len(X),batch_size):
                X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
                if nesterov:
                    self.look_ahead_weights = self.weights - momentum * self.velocity_w
                    self.look_ahead_bias = self.bias - momentum * self.velocity_b
                    y_pred=self.predict_nesterov(X_batch)
                else:
                    y_pred=self.predict(X_batch)
                loss=self.cross_entropy(y_batch,y_pred)
                self.loss.append(loss)
                # update
                dweight,dbias=self.get_gradients(y_batch,y_pred,X_batch)
                if scaling_factor:
                    cosine_similarity=np.mean(np.dot(X_batch,self.mean)/(np.linalg.norm(X_batch)*np.linalg.norm(self.mean)))
                    min_cosine=min(min_cosine,cosine_similarity)
                    lr=cosine_similarity*scaling_factor
                    if cosine_similarity<=0:
                        continue

                if standard:
                    #NOTE comment out next two lines if lr should not decay
                    # rescale=scaling_factor if scaling_factor else 0.01
                    # lr=lr * (1 / (1 + rescale * i))
                    self.weights-=lr*dweight
                    self.bias-=lr*dbias

                else:
                    #momentum
                    self.velocity_w=momentum*self.velocity_w+lr*dweight
                    self.velocity_b=momentum*self.velocity_b+lr*dbias
                    self.weights-=self.velocity_w
                    self.bias-=self.velocity_b

        if scaling_factor:
            print("Min cosine similarity {}, scaling factor {}".format(min_cosine,scaling_factor))
                # if np.abs(dweight).max() < self.threshold:
                #     break
                # if i % 100 == 0:
                #     print(' Training Accuray at {} iterations is {}'.format(i, self.evaluate(X, y)))

    def one_hot(self, y):
        """Dummy conversion."""
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

    def predict(self, X):
        """Linear prediction, followed by softmax."""
        y_pred = np.dot(X, self.weights.T)+self.bias
        return self.softmax(y_pred)

    def predict_nesterov(self,X):
        """Linear prediction, followed by softmax for the Nesterov method."""
        y_pred = np.dot(X, self.look_ahead_weights.T)+self.look_ahead_bias
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
        return np.mean(self.predict_classes(X) == np.vectorize(lambda c: self.classes[c])(np.argmax(y, axis=1)))

    def score(self, X_test, y_test):
        """Testing evaluation. y is a 1d vector."""
        X_test=X_test.to_numpy()
        y_test=y_test.to_numpy().flatten()
        return np.mean(self.predict_classes(X_test) ==y_test)

    def plot(self):
        plt.plot(self.loss)
        plt.show()
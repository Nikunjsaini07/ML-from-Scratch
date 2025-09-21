import numpy as np

class Perceptron:

    def __init__(self ,  lr = 0.01 ,epochs = 1000):
       self.epochs = epochs
       self.lr = lr
       self.weight = None
    def fit(self , X_train , y_train):
         X_train =   np.insert(X_train , 0 , 1 , axis=1)
         self.weight = np.ones(X_train.shape[1])

         for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                y_hat = 1 if np.dot(X_train[j], self.weight) > 0 else 0
                self.weight += self.lr * (y_train[j] - y_hat) * X_train[j]
                
         return   self 
    
    def predict(self, X):
        X = np.insert(X , 0 , 1 , axis=1)
        return np.where(np.dot(X, self.weight) > 0, 1, 0)
    
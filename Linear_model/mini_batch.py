import numpy as np 
import random

class Mini_batch:

    def __init__(self , batch_size = 10 , lr = 0.01 , epochs = 10):
        self.coef = None
        self.intercept = None
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self , X_train , y_train):

        #initiating the values 
        self.intercept = 0
        self.coef = np.ones(X_train.shape[1])    

        for i in range(self.epochs):

            
            for j in range(X_train.shape[0] //self.batch_size):
                idx = random.sample(range(X_train.shape[0]) ,self.batch_size )
                # It's more like the stochastic gradient descent just the difference is it uses mini-batches


                y_pred = np.dot(X_train[idx], self.coef) + self.intercept
                slope_inter = -2*np.mean(y_train[idx] - y_pred)
                slope_coef = -2 * np.dot((y_train[idx] - y_pred) , X_train[idx] ) / self.batch_size

                self.intercept -= self.lr * slope_inter
                self.coef -= self.lr * slope_coef


        print(f'Intercept: {self.intercept}, Coefficients: {self.coef}')  

    def predict(self, X_test):
        return np.dot(X_test, self.coef) + self.intercept
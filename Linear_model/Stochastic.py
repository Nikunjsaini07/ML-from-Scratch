import numpy as np 


class SGD:

    def __init__(self , lr = 0.01 , epochs = 10):
        self.coef = None
        self.intercept = None
        self.lr = lr
        self.epochs = epochs

    def fit(self , X_train , y_train):

        #initiating the values 
        self.intercept = 0
        self.coef = np.ones(X_train.shape[1])    

        for i in range(self.epochs):

            # the core idea behind this gd is that we only update for one sample at a time i.e one row of training data 
            for j in range(X_train.shape[0]):
                id = np.random.randint(0, X_train.shape[0])

                y_pred = np.dot(X_train[id], self.coef) + self.intercept
                slope_inter = -2(y_train[id] - y_pred)
                slope_coef = -2 * (y_train[id] - y_pred) * X_train[id]  

                self.intercept -= self.lr * slope_inter
                self.coef -= self.lr * slope_coef


        print(f'Intercept: {self.intercept}, Coefficients: {self.coef}')  

    def predict(self, X_test):
        return np.dot(X_test, self.coef) + self.intercept

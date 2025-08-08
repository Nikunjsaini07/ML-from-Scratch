import numpy as np 


class SM_linear():
    
    def __init__(self):
        self.m = None 
        self.b = None 

    def fit(self , X_train , y_train):

        numerator = 0
        denominator = 0

        for i in range(len(X_train)):
            numerator = numerator + (X_train[i] - np.mean(X_train) )*(y_train[i] - np.mean(y_train))
            denominator = denominator + (X_train[i] - np.mean(X_train))**2

        self.m = numerator/denominator 

        self.b = np.mean(y_train) - self.m*(np.mean(X_train))


        return self 
    
    def predict(self , x_test ):

        y_pred = self.m*x_test + self.b 
        
        return y_pred 




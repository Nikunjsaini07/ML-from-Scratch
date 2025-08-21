import numpy as np 


class Ridge_simple():
    
    def __init__(self , alpha = 0.001 ):
        self.m = None 
        self.b = None 
        self.alpha = alpha 

    def fit(self , X_train , y_train):

        numerator = 0
        denominator = 0

        for i in range(len(X_train)):
            numerator = numerator + (X_train[i] - np.mean(X_train) )*(y_train[i] - np.mean(y_train))
            denominator = denominator + (X_train[i] - np.mean(X_train))**2

        self.m = numerator/(denominator + self.alpha)

        self.b = np.mean(y_train) - self.m*(np.mean(X_train))


        return self 
    
    def predict(self , x_test ):

        y_pred = self.m*x_test + self.b 
        
        return y_pred 
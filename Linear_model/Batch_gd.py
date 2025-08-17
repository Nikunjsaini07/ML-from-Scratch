import numpy as np

class BGD:

    def __init__(self , lr=0.01 , epochs=1000):
        self.coef = None 
        self.intercept = None 
        self.lr = lr 
        self.epochs = epochs
    
    def fit(self , X_train , y_train):

        #initiating the values 
        self.intercept = 0
        self.coef = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            # updating the intercept

            y_pred = (np.dot( X_train , self.coef) + self.intercept)
            slope_i = -2*np.mean(y_train - y_pred)
            self.intercept = self.intercept - (self.lr * slope_i)

            # updating the coef..
            
            slope_coef  = -2*np.dot((y_train - y_pred), X_train) / len(X_train)
            self.coef = self.coef - (self.lr * slope_coef )



            
    def predict(self , X_test):

        return  (np.dot(X_test , self.coef )) + self.intercept
         

     
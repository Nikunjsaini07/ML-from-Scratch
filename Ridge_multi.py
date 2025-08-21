import numpy as np

class MLRidge:

    def __init__(self, alpha=0.001):
        self.coef = None
        self.intercept = None
        self.alpha = alpha

    def fit(self , X_train , y_train): 
            X_train = np.insert(X_train , 0 , 1 , axis=1)
            transpose = np.transpose(X_train)
            iden = np.identity(X_train.shape[1])
            iden[0][0] = 0
            
            inverse = np.linalg.inv(np.dot(transpose, X_train ) + self.alpha * iden)
            
            B = np.dot(inverse, np.dot(transpose, y_train))
            self.coef = B[1:]
            self.intercept = B[0]

            return self
    
    def predict(self , X_test):
         y_pred = np.dot( X_test , self.coef) + self.intercept
         return  y_pred

        
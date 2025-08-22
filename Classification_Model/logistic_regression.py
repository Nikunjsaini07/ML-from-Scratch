import numpy as np

class Logistic_R:

    def __init__(self , lr = 0.01 ,epochs = 1000):
       self.epochs = epochs
       self.weights = None
       self.lr = lr 
       

    def fit(self, X_train, Y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)  # Add bias term
        self.weights = np.zeros(X_train.shape[1])
        

        # Gradient Descent
        for _ in range(self.epochs):  # number of iterations
            z = np.dot(X_train, self.weights) 
            y_predicted = self.sigmoid(z)

          
            dw = np.dot(X_train.T, ( Y_train - y_predicted ))/X_train.shape[0]
            

            # Update weights 
            self.weights += self.lr * dw
            

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

    def probability(self, X_test):
        # Add bias column
        X_test = np.insert(X_test, 0, 1, axis=1)
        z = np.dot(X_test, self.weights)
        return self.sigmoid(z)
    
    def predict(self , X_test ,threshold = 0.5 ):   #this is used during roc auc curve 
         
         Y_pred = self.probability(X_test)

         return (Y_pred >= threshold).astype(int)  # Convert probabilities to binary predictions
  

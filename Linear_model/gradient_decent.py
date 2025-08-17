import numpy as np

class gd:


    def __init__(self ,  lr=0.01, epoch=1000 ):
        self.m = 0 
        self.b= 0
        self.lr = lr
        self.epoch = epoch


    def fit(self , X_train , Y_train ):
        
       for i in range(self.epoch):
            slope_m = -2 *np.mean((Y_train - (self.m * X_train + self.b)) * X_train)
            slope_b = -2 * np.mean((Y_train - (self.m * X_train + self.b)))

            self.m = self.m - self.lr*slope_m
            self.b = self.b - self.lr*slope_b
            # printing epoch at every 100 
            if i % 100 == 0:
                loss = np.mean((Y_train - (self.m*X_train + self.b))**2)
                print(f"Epoch {i}, m={self.m}, b={self.b}, loss={loss}")
                
       print(self.m , self.b )  
       

    def predict(self , x_test ):

        return self.m *x_test + self.b 

           
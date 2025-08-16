import numpy as np

class gd:


    def __init__(self ,  lr , epoch ):
        self.m = 0 
        self.b= 0
        self.lr = lr
        self.epoch = epoch


    def fit(self , X_train , Y_train ):
        
       for i in range(self.epoch):
            slope_m = -2 *np.sum((Y_train - (self.m * X_train + self.b)).X_train)
            slope_b = -2 * np.sum((Y_train - (self.m * X_train + self.b)))

            self.m = self.m - self.lr*slope_m
            self.b = self.b - self.lr*slope_b

            print(self.m , self.b )
       

    def predict(self , x_test ):

        return self.m *x_test + self.b 

           
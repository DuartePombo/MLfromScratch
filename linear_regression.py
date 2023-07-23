import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr #learning rate
        self.n_iters = n_iters # number of iterations in the gradient descent
        self.weights = None # the beta coefficients related with X
        self.bias = None # the constant or beta1

    def fit(self, X, y):

        # The formulas are:

        # weights_new = weights - LearningRate * derivativeWeights 
        # bias_new = bias - LearningRate * DerivativeBias

        # The derivative of the cost function in relation to the weights (betas) = dw = (1/N) * np.dot(2xi, y_hat - yi)  , where the 2 is just a scaling factor we will ommit
        # The derivative of the cost function in relation to the bias (constant)    = db = (1/N) * np.sum(2 * (y_hat - yi))  , where the 2 is just a scaling factor we will ommit

        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias #the linear reg. Here we do X * weights because we want to get weights for each feature
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) #the derivative of the weights. Here we have to transpose the X and do X.T * weights because we want a value for each prediction.
                                                                #So this is along the other axis, that is why we need to be careful and use the transpose
            
            db = (1/n_samples) * np.sum(y_predicted-y)

            self.weights = self.weights - self.lr * dw #updating the new betas corresponding to X
            self.bias = self.bias - self.lr * db       #updating the new constant



    def predict(self,X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

        

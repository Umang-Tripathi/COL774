# Imports - you can add any other permitted libraries
import numpy as np


# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):

        self.theta = None
        self.time_interval = 0.02

        pass
    
    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Gradient Descent.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target values.

        learning_rate : float
            The learning rate to use in the update rule.
            
        Returns
        -------
        List of Parameters: numpy array of shape (n_iter, n_features,)
            The list of parameters obtained after each iteration of Gradient Descent.
        """

        # Added intercept term (1) so that feature matrix X is m x 2
        X = np.hstack((np.ones((m, 1)), X)) 

        # Initialize theta to zeros
        self.theta = np.zeros((X.shape[1], 1))

        # Number of training examples
        m = len(y)
        n = X.shape[1]

        # List to store the parameters after each iteration
        theta_list = []
        J_theta_list = []
        gradient_list = []
        prev_J_theta = float('inf')

        start_time = time.time()
        while True:
            # Compute the predictions : ( h_theta(x(i)) )
            predictions = X.dot(self.theta)

            # Compute the error : ( y(i) - h_theta(x(i)) )
            error = predictions - y

            # Compute the gradient : gradient = (1/m) sum ( h_theta(x(i)) - y(i) ) * x(i)
            gradient = (1 / m) * X.T.dot(error)

            # Update theta :  theta(t+1) = theta(t) - learning_rate * gradient
            self.theta = self.theta - learning_rate * gradient

            # Compute J_theta
            J_theta = (1 / (2 * m)) * np.sum(error ** 2)
            if ( time.time() - start_time > self.time_interval):
                J_theta_list.append()
                start_time = time.time()
                

            # Store the theta
            theta_list.append(self.theta)

            # Check convergence condition
            if abs(J_theta - prev_J_theta) < 1e-9:
                break

            # Update prev_J_theta
            prev_J_theta = J_theta

        return theta_list

        pass
    
    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : numpy array of shape (n_samples,)
            The predicted target values.
        """

        y_pred = X.dot(self.theta)

        return y_pred
    
    
    




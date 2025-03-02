# Imports - you can add any other permitted libraries
import numpy as np


# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):

        self.theta = None # Parameters of the model

        self.max_iter = 1000000 # Maximum number of iterations

        self.tol = 1e-7 # Convergence criterion

        self.iterations = None # Number of iterations until convergence

        pass
    
    def fit(self, X, y, learning_rate=1.0):
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
        List of Parameters: numpy array of shape (n_iter, n_features + 1,)
            The list of parameters obtained after each iteration of Gradient Descent.
        """

        m, n = X.shape
        X = np.c_[np.ones(m), X]  # Add intercept term
        self.theta = np.zeros(n + 1)  # Initialize parameters

        params_list = []
        params_list.append(self.theta.copy())
        prev_cost = float('inf')

        
        Itr = 0
        while Itr < self.max_iter:
            Itr += 1

            predictions = X @ self.theta  # Compute predictions
            error = predictions - y
            gradient = (X.T @ error) / m  # Compute gradient
            self.theta -= learning_rate * gradient  # Update parameters

            # Compute cost
            cost = (error @ error) / (2 * m)
            params_list.append(self.theta.copy())

            # Check convergence
            if abs(prev_cost - cost) < self.tol:
                break
            prev_cost = cost

        self.iterations = Itr

        return np.array(params_list)
        
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

        m = X.shape[0]
        X = np.c_[np.ones(m), X]  # Add intercept term


        return X @ self.theta  # Compute predictions

        pass






# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LogisticRegressor:
    # Assume Binary Classification
    def __init__(self):

        self.theta = None # Parameters of the model

        self.mean = None # Mean of each feature

        self.std = None # Standard deviation of each feature

        pass

    def _sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def _normalize(self, X , mean=None, std=None):

        if mean is None or std is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            self.mean = mean
            self.std = std

        return (X - mean) / std
    
    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Newton's Method.
        Remember to normalize the input data X before fitting the model.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target labels - 0 or 1.
        
        learning_rate : float
            The learning rate to use in the update rule.
        
        Returns
        -------
        List of Parameters: numpy array of shape (n_iter, n_features+1,)
            The list of parameters obtained after each iteration of Newton's Method.
        """

        

        # Normalize the input data

        X = self._normalize(X, self.mean, self.std)

        # Add intercept term
        X = np.hstack([np.ones((X.shape[0], 1)), X])


        M, N = X.shape
        self.theta = np.zeros(N)

        theta_history = []
        max_iter = 1000000
        tol = 1e-7


        for _ in range(max_iter):
  
            z = X @ self.theta

            # Probability of y=1 given X: h = 1 / (1 + exp(-z))
            h = self._sigmoid(z)

            # Gradient of the log-likelihood: X.T * (y - h)
            gradient = X.T @ (y - h)

            # Diagonal matrix R of the probabilities h
            R = np.diag(h * (1 - h))

            # Hessian matrix H = X.T * R * X
            # H[i][j] = d^2 ( h ) / d ( theta_i ) d ( theta_j )
            # H[i][j] = -X.T[i] * R * X[j] if i != j
            H = X.T @ R @ X 

            # Newton’s update step: θ_new = θ + H^(-1) * gradient
            try:
                delta_theta = learning_rate*np.linalg.inv(H) @ gradient

            except np.linalg.LinAlgError:
                
                H = X.T @ R @ X + 1e-9 * np.eye(N)
                delta_theta = learning_rate*np.linalg.inv(H) @ gradient

                break

            self.theta += delta_theta
            theta_history.append(self.theta.copy())

            # Convergence check
            if np.linalg.norm(delta_theta) < tol:
                print(_)
                break

        return np.array(theta_history)


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
            The predicted target label.
        """

        X= self._normalize(X)

        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term


        # P = 1 / (1 + exp(-theta(T).X))
        probabilities = self._sigmoid(X @ self.theta)

        y_pred = (probabilities >= 0.5).astype(int)
    
        return y_pred
        pass
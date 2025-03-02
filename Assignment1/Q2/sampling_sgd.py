import numpy as np
from collections import deque

def generate(N, theta, input_mean, input_sigma, noise_sigma):
    """
    Generate normally distributed input data and target values
    Note that we have 2 input features
    Parameters
    ----------
    N : int
        The number of samples to generate.
        
    theta : numpy array of shape (3,)
        The true parameters of the linear regression model.
        
    input_mean : numpy array of shape (2,)
        The mean of the input data.
        
    input_sigma : numpy array of shape (2,)
        The standard deviation of the input data.
        
    noise_sigma : float
        The standard deviation of the Gaussian noise.
        
    Returns
    -------
    X : numpy array of shape (N, 2)
        The input data.
        
    y : numpy array of shape (N,)
        The target values.
    """

 
    x1 = np.random.normal(loc=input_mean[0], scale=input_sigma[0], size=N)
    x2 = np.random.normal(loc=input_mean[1], scale=input_sigma[1], size=N)


    X = np.column_stack((x1, x2))

    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=noise_sigma, size=N)

    # Compute target values: y = theta_0 + theta_1 * x1 + theta_2 * x2 + noise
    y = theta[0] + theta[1] * x1 + theta[2] * x2 + noise

    return X, y

class StochasticLinearRegressor:
    def __init__(self):

        self.theta = None
        self.learned_thetas = None
        self.theta_itr_history = None
        self.batch_sizes = [1, 80, 8000, 800000]
        self.tol = {1: 1e-4, 80: 1e-4, 8000: 1e-6, 800000: 1e-7}
        self.max_epochs = {1: 10, 80: 100, 8000: 1000, 800000: 10000}

        # self.batch_sizes = [1, 8, 80 , 800]
        # self.tol = {1: 1e-3, 8: 1e-4, 80: 1e-4, 800: 1e-7}
        # self.max_epochs = {1: 10, 8: 100, 80: 1000, 800: 10000}

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
        List of Parameters: numpy array of shape (n_iter, n_features+1,)
            The list of parameters obtained after each iteration of Gradient Descent.
        """
        
        m = X.shape[0]
        X = np.c_[np.ones(m), X]

        n_samples, n_features = X.shape

        learned_thetas = np.zeros((len(self.batch_sizes), n_features))

        theta_history = []
        theta_itr_history = []

        for batch_size in self.batch_sizes:

            # Initialize parameters to zeros
            theta = np.zeros(n_features)
            prev_theta = np.ones(n_features) 
            
            theta_history_batch = []
            theta_itr_history_batch = []


            epoch = 0
            rolling_window = deque(maxlen=10)  # Rolling window of size 10

            while epoch < self.max_epochs[batch_size]:
                # if(epoch % (max(1,int(batch_size/100))) == 0):
                #     print(f"Epoch: {epoch} for {batch_size}")
                prev_theta = theta.copy()  # Store previous theta

                indices = np.random.permutation(n_samples)  # Shuffle data
                
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    X_batch, y_batch = X[batch_indices], y[batch_indices]

                    # Compute gradient: delta(J_theta) = (1/batch_size) * X_batch^T (X_batch θ - y_batch)
                    gradient = (1 / batch_size) * X_batch.T @ (X_batch @ theta - y_batch)
                    
                    # Update parameters
                    theta -= learning_rate * gradient
                    theta_itr_history_batch.append(theta.copy())
                
                theta_history_batch.append(theta.copy())
                
                rolling_window.append(np.linalg.norm(theta - prev_theta, ord=2))
                if len(rolling_window) == 10 and np.mean(rolling_window) < self.tol[batch_size]:
                    break
                epoch += 1

            #print(f"\nBatch size: {batch_size}\n Epochs: {epoch} \n Theta: {theta}\n")
            
            theta_history.append(np.array(theta_history_batch))
            theta_itr_history.append(np.array(theta_itr_history_batch))

            learned_thetas[self.batch_sizes.index(batch_size)] = theta
        
        self.learned_thetas = learned_thetas
        self.theta = learned_thetas[-1]
        self.theta_itr_history = theta_itr_history

        return (theta_history[0], theta_history[1], theta_history[2], theta_history[3])

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
        X = np.c_[np.ones(m), X]

        y_pred_list = np.array([X @ theta for theta in self.learned_thetas])

        return y_pred_list
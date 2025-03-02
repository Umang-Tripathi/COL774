# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.


class GaussianDiscriminantAnalysis:
    # Assume Binary Classification
    def __init__(self):

        self.mu = None # Mean of each class

        self.sigma = None # Covariance matrix for classes

        self.classes_ = None # Class labels

        self.class_prior = None  # Prior probabilities of each class

        self.mean = None # Mean of each feature

        self.std = None # Standard deviation of each feature

        pass
    def normalize_features(self, X, means=None, stds=None):
        """
        Normalize features using mean and standard deviation.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
        
        Returns
        -------
        X_norm : numpy array of shape (n_samples, n_features)
            The normalized input data.
        """
        if means is None or stds is None:
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)

        return (X - means) / stds 

    def fit(self, X, y, assume_same_covariance=False):
        """
        Fit the Gaussian Discriminant Analysis model to the data.
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
        Parameters: 
            If assume_same_covariance = True - 3-tuple of numpy arrays mu_0, mu_1, sigma 
            If assume_same_covariance = False - 4-tuple of numpy arrays mu_0, mu_1, sigma_0, sigma_1
            The parameters learned by the model.
        """ 

        # Store class labels

        self.classes_ = np.unique(y)

        # Normalize features

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        X_norm = self.normalize_features(X)
        X = X_norm

        # Number of samples for each class(here only two classes - 0 , 1 )

        M_classes = [np.sum(y == c) for c in self.classes_] # Number of samples for each class

        mu = np.zeros((len(self.classes_), X.shape[1])) # Mean of each class
        
        M = X.shape[0] # Total number of samples

        # Prior probabilities of each class

        self.class_prior = [M_classes[i] / M for i in range(len(self.classes_))]


        """
        From Lecture notes :
            (For only 2 classes) ( 0 and 1 )

            mu_0 = 1/M0 * sum(x_i) where y_i = 0
            mu_1 = 1/M1 * sum(x_i) where y_i = 1
          
            sigma_0 = 1/M0 * sum((x_i - mu_0) * (x_i - mu_0)^T) where y_i = 0
            sigma_1 = 1/M1 * sum((x_i - mu_1) * (x_i - mu_1)^T) where y_i = 1
        
            If same covariance matrix is assumed, then 
            sigma = 1/N * sum((x_i - mu_y(i)) * (x_i - mu_y(i))^T) 
        """

        for idx, c in enumerate(self.classes_):
            mu[idx] = np.sum(X[y == c], axis=0) / M_classes[idx] # Mean of each class

        



        if assume_same_covariance:
            
            sigma = np.zeros((X.shape[1], X.shape[1]))

            for idx, c in enumerate(self.classes_):
                sigma += np.sum([np.outer(x - mu[idx], x - mu[idx]) for x in X[y == c]], axis=0)

            sigma /= M



        else :
            
            sigma = np.zeros((len(self.classes_), X.shape[1], X.shape[1]))

            for idx, c in enumerate(self.classes_):
                sigma[idx] = np.sum([np.outer(x - mu[idx], x - mu[idx]) for x in X[y == c]], axis=0) / M_classes[idx]
        

     



        if ( assume_same_covariance ):

            self.sigma = np.array([sigma for _ in range(len(self.classes_))])
            self.mu = mu

            mu_0 = mu[0]
            mu_1 = mu[1]

            return (mu_0, mu_1, sigma)
        
        else :

            self.sigma = sigma
            self.mu = mu
            

            mu_0 = mu[0]
            mu_1 = mu[1]
            sigma_0 = sigma[0]
            sigma_1 = sigma[1]


            return (mu_0, mu_1, sigma_0, sigma_1)
            
            
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
        # Normalize features
        X = self.normalize_features(X, self.mean, self.std)
        
        # Initialize array to store probabilities
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, len(self.classes_)))
        
        # Calculate probability for each class
        for idx, c in enumerate(self.classes_):

            diff = X - self.mu[idx]
            inv_sigma = np.linalg.inv(self.sigma[idx])
            exponent = -0.5 * np.sum((diff @ inv_sigma) * diff,axis=1)
            norm_const = -0.5 * np.log(np.linalg.det(self.sigma[idx]))
            probs[:, idx] = exponent + norm_const + np.log(self.class_prior[idx])
        
        # Return class with highest probability
        return np.argmax(probs, axis=1)
    
    def pdf(self, x, class_idx):
        """
        Calculate the Gaussian probability density function for a given point and class.
        
        Parameters:
        x (np.array): The input point.
        class_idx (int): The class index.
        
        Returns:
        float: The probability density value.
        """

        mu = self.mu[class_idx]
        sigma = self.sigma[class_idx]
        size = len(x)
        
        det = np.linalg.det(sigma)
        norm_const = 1.0/(np.power((2*np.pi), float(size)/2) * np.power(det, 1.0/2))
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)
        result = np.power(np.e, -0.5*( x_mu*inv*x_mu.T ))
        
        return norm_const * result
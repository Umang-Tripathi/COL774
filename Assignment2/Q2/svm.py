import cvxopt
import numpy as np
cvxopt.solvers.options['show_progress'] = False

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):

        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.w = None
        self.b = None
        self.kernel = None
        self.gamma = None

        pass

    def linear_kernel(self, X):
        
        return X @ X.T  
    
    def gaussian_kernel(self,X,Z,gamma):

        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
        Z_norm = np.sum(Z**2, axis=1).reshape(1, -1)
        sq_dist = X_norm + Z_norm - 2 * np.dot(X, Z.T)
        return np.exp(-gamma * sq_dist)  
    
    def fit(self, X, y, kernel = 'linear', C = 1.0, gamma = 0.001):
        '''
        Learn the parameters from the given training data
        Classes are 0 or 1
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            kernel: str
                The kernel to be used. Can be 'linear' or 'gaussian'
                
            C: float
                The regularization parameter
                
            gamma: float
                The gamma parameter for gaussian kernel, ignored for linear kernel
        '''
        N, D = X.shape
        y = y.astype(float)
        y = 2 * y - 1
        y = y.reshape(-1, 1) 
        
        self.kernel = kernel
        self.gamma = gamma
        
  
        if kernel == 'linear':
            K = self.linear_kernel(X)
        elif kernel == 'gaussian':
            K = self.gaussian_kernel(X,X, gamma)
        else:
            raise ValueError("Unsupported kernel. Choose 'linear' or 'gaussian'.")
        
       
        P = cvxopt.matrix((y @ y.T) * K) 
        q = cvxopt.matrix(-np.ones(N))
        
        G = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
        
        A = cvxopt.matrix(y.T)
        b = cvxopt.matrix(0.0)
        

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])
        

        support_vector_idx = alpha > 1e-5
        self.alpha = alpha[support_vector_idx]
        self.support_vectors = X[support_vector_idx]
        self.support_vector_labels = y[support_vector_idx].flatten()
        
        if kernel == 'linear':
            # weight vector w
            self.w = np.sum(self.alpha[:, None] * self.support_vector_labels[:, None] * self.support_vectors, axis=0)
            
            # bias term b
            self.b = np.mean(
                self.support_vector_labels - np.dot(self.support_vectors, self.w)
            )
        else:
            self.w = None  
            K_sv = K[support_vector_idx][:, support_vector_idx] 
            self.b = np.mean(
                self.support_vector_labels - np.sum(
                    self.alpha[:, None] * self.support_vector_labels[:, None] * K_sv, axis=0
                )
            )

    def predict(self, X):
        '''
        Predict the class of the input data
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)
        '''

        if self.kernel == 'linear':
            return (np.dot(X, self.w) + self.b > 0).astype(int)
        elif self.kernel == 'gaussian':
            K = self.gaussian_kernel(X, self.support_vectors, self.gamma)
            return (np.sum(self.alpha * self.support_vector_labels * K, axis=1) + self.b > 0).astype(int) 
             
        pass
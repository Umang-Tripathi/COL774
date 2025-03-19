import cvxopt
import numpy as np

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):
        self.alpha = None
        self.sv = None
        self.sv_y = None
        self.w = None
        self.b = None
        self.kernel = None
        self.gamma = None
        self.C = None
        self.X = None
        self.top_alpha_indices = None
        pass
        
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
        
        self.kernel = self.linear_kernel if kernel == 'linear' else lambda X, Z: self.gaussian_kernel(X, Z, gamma)
        self.gamma = gamma
        self.C = C
        self.X = X
        
        N, D = X.shape
        y = (2*y) - 1 
        
        K = self.kernel(X, X)
        
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(N))
        G = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
        A = cvxopt.matrix(y.astype(float), (1, N))
        b = cvxopt.matrix(0.0)
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        alpha = np.ravel(sol['x'])
        

        sv = alpha > 1e-5
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        self.top_alpha_indices = np.argsort(alpha)[-5:][::-1]
        

        if kernel == 'linear':
            self.w = np.sum(self.alpha[:, None] * self.sv_y[:, None] * self.sv, axis=0)
            self.b = np.mean(self.sv_y - np.dot(self.sv, self.w))
        else:
            self.w = None
            self.b = np.mean(self.sv_y - np.sum(self.alpha * self.sv_y * self.kernel(self.sv, self.sv), axis=1))
        
        pass

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
        if self.kernel == self.linear_kernel:
            return (np.dot(X, self.w) + self.b > 0).astype(int)
        else:
            K = self.kernel(X, self.sv)
            return (np.sum(self.alpha * self.sv_y * K, axis=1) + self.b > 0).astype(int)
        
        pass
    
    def linear_kernel(self, X, Z):
        return np.dot(X,Z.T)
    
    def gaussian_kernel(self,X,Z,gamma):
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
        Z_norm = np.sum(Z**2, axis=1).reshape(1, -1)
        sq_dist = X_norm + Z_norm - 2 * np.dot(X, Z.T)
        return np.exp(-gamma * sq_dist)


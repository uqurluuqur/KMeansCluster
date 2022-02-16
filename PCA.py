# Principal Component Analysis
import numpy as  np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self,X):
        # mean
        
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance matrix
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # sort eigenvactors
        eigenvectors = eigenvectors.T
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]
        # store first n eigenvectors
        self.components = eigenvectors[:self.n_components]

    def transform(self,X):
        print(self.mean,"aa",X)
        X = X - self.mean
        transformed_X = np.dot(X,self.components.T)
        return transformed_X


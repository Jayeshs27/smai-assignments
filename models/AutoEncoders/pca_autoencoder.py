import numpy as np

class PcaAutoencoder:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.eigenvectors = eigenvectors[:, :self.n_components]
    
    def encode(self, X):
        X_centered = X - self.mean
        X_encoded = np.dot(X_centered, self.eigenvectors)
        return X_encoded
    
    def forward(self, X):
        X_encoded = self.encode(X)
        X_reconstructed = np.dot(X_encoded, self.eigenvectors.T)
        X_reconstructed += self.mean
        return X_reconstructed
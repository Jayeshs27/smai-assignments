import numpy as np

class PrincipalComponentAnalysis():
    def __init__(self, num_components:int):
        self.num_components = num_components
        self.principal_components = None
        self.transformed_data = None
        self.explained_variance_ratio = None
        self.mean = None
        self.threshold = 1.1

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        data = data - self.mean
        cov_mat = np.cov(data, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eig(cov_mat)

        eigen_values = eigen_values.real
        eigen_vectors = eigen_vectors.real
        sorted_indices = np.argsort(eigen_values)[::-1]
        sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
        self.principal_components = sorted_eigen_vectors[:, :self.num_components]
        total_variance = np.sum(eigen_values)
        self.explain_variance_ratio = eigen_values / total_variance

    def transform(self, data):
        data = data - np.mean(data, axis=0)
        self.transformed_data = data @ self.principal_components
        return self.transformed_data

    def variance_ratio(self):
        return self.explain_variance_ratio[:self.num_components]
    
    def inverse_transform(self, data):
        original_data = data @ self.principal_components.T 
        original_data += self.mean
        return original_data

    def checkPCA(self, original_data) -> bool:
        if self.transformed_data is None:
            return False
        reconstructed_data = self.inverse_transform(self.transformed_data)
        print(reconstructed_data)
        print(original_data)
        difference = abs(original_data - reconstructed_data)
        print(difference)
        return (difference < self.threshold).all()
    
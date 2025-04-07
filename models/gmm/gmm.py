import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixtureModel():
    def __init__(self, num_components:int):
        self.num_components = num_components
        self.means = None
        self.covariances = None
        self.mixing_coefficients = None
        self.membership_probabilities = None
    
    def fit(self, X:np.ndarray):
        num_samples, num_features = X.shape
        self.means = X[np.random.choice(num_samples, self.num_components, replace=False)]
        self.covariances = np.array([np.eye(num_features)] * self.num_components)
        self.mixing_coefficients = np.ones(self.num_components) / self.num_components
        
        log_likelihood = 0
        while True:
            prev_log_likelihood = log_likelihood
            self.membership_probabilities = np.zeros((num_samples, self.num_components))

            log_probs = np.zeros((num_samples, self.num_components))
            # print(self.covariances)
            for clt_index in range(self.num_components):
                log_prob = multivariate_normal(self.means[clt_index], self.covariances[clt_index]).logpdf(X)
                log_probs[:, clt_index] = np.log(self.mixing_coefficients[clt_index]) + log_prob

            mx_log_probs = np.max(log_probs, axis=1, keepdims=True)
            log_probs -= mx_log_probs
            probs = np.exp(log_probs)
            self.membership_probabilities = probs / np.sum(probs, axis=1, keepdims=True)

            total_probs = np.sum(self.membership_probabilities, axis=0)
            total_probs = np.where(total_probs < 1e-6, 1e-6, total_probs)
            self.means = np.dot(self.membership_probabilities.T, X) / total_probs[:, None]
            self.mixing_coefficients = total_probs / num_samples

            for clt_index in range(self.num_components):
                self.cov_mat = np.dot(self.membership_probabilities[:,clt_index] * (X - self.means[clt_index]).T, (X - self.means[clt_index])) / total_probs[clt_index]
                self.covariances[clt_index] = self.cov_mat + np.eye(num_features) * 1e-6

            log_likelihood = self.getLikelihood(X)
            if abs(prev_log_likelihood - log_likelihood) < 1e-6:
                break

    def getParams(self):
        return self.means, self.convariances, self.mixing_coefficients

    def getMembership(self):
        return self.membership_probabilities
    
    def getLikelihood(self, X:np.ndarray):
        log_probs = np.zeros((len(X), self.num_components))
        for clt_index in range(self.num_components):
            log_prob_ic = np.log(self.mixing_coefficients[clt_index]) + multivariate_normal(self.means[clt_index], self.covariances[clt_index]).logpdf(X)
            log_probs[:, clt_index] = log_prob_ic

        mx_log_probs = np.max(log_probs, axis=1, keepdims=True)
        log_probs -= mx_log_probs
        probs = np.exp(log_probs)
        log_likelihood = np.sum(mx_log_probs.squeeze() + np.log(np.sum(probs, axis=1)))
        return log_likelihood
    
    # def score_samples(self, X: np.ndarray):
    #     log_probs = np.zeros((len(X), self.num_components))
    #     for clt_index in range(self.num_components):
    #         log_prob_ic = np.log(self.mixing_coefficients[clt_index]) + multivariate_normal(
    #             self.means[clt_index], self.covariances[clt_index]).logpdf(X)
    #         log_probs[:, clt_index] = log_prob_ic

    #     mx_log_probs = np.max(log_probs, axis=1, keepdims=True)
    #     log_probs -= mx_log_probs
    #     probs = np.exp(log_probs)
    #     log_probabilities = mx_log_probs.squeeze() + np.log(np.sum(probs, axis=1))
    #     return log_probabilities
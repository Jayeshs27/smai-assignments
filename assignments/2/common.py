#### Common Imports and functions ####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys, os

from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/knn")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/k-means")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/pca")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/gmm")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../performance-measures")))


from knn import kNNClassifier
from k_means import kMeansClustering
from pca import PrincipalComponentAnalysis
from gmm import GaussianMixtureModel
from evaluation import model_evaluation


def read_word_embeddings_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_feather('../../data/external/word-embeddings.feather')
    # df dim : 200 x 2 where second coloumn is 512 dim vector embedding
    data = df.to_numpy()
    embeddings = np.vstack(data[:,-1])
    labels = np.vstack(data[:,:1])
    embeddings = embeddings.astype(np.float64)
    return embeddings, labels


def z_score_normalization(data:np.ndarray) -> np.ndarray:
    std_dev = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    std_dev = np.where(std_dev < 1e-6, 1e-6, std_dev)
    normalized_data = (data - mean) / std_dev
    return normalized_data

def plot_scree_plot(data: np.ndarray, n_components:int, outPath:str):
    pca = PrincipalComponentAnalysis(num_components=n_components)
    pca.fit(data)
    variance_ratio = pca.variance_ratio()
    cum_variance_ratio = np.cumsum(variance_ratio)

    plt.figure(figsize=(15, 12))
    plt.plot(np.arange(1, n_components + 1), cum_variance_ratio, 'o-', color='b', ms=6, label='Explained Variance')
    plt.xticks(np.arange(1, n_components + 1, 1))
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Proportion of explained variance')
    # plt.xticks(np.arange(1, n_components + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outPath)
    # plt.show()

def print_clusters(labels: np.ndarray, cluster_assignments: np.ndarray):
    clusters = defaultdict(list)
    for label, cluster in zip(labels, cluster_assignments):
        clusters[cluster].append(label)
    
    for cluster in sorted(clusters.keys()):
        print(f"Cluster {int(cluster)}:")
        print(", ".join(str(item[0]) for item in clusters[cluster]))
        print("-" * 80)

def plot_k_vs_cost(embeddings:np.ndarray, outPath:str) -> None:
    k_max=21
    costs=[]
    for i in range(1, k_max):
        kmeans = kMeansClustering(k=i)
        kmeans.fit(embeddings)
        cost = kmeans.getCost()
        costs.append(float(cost))

    k_values = [int(i) for i in range(1, k_max)]
    plt.figure(figsize=(8,6))
    plt.plot(k_values, costs, marker='o', ms=5)
    plt.xticks(np.arange(min(k_values), max(k_values)+1, 1))  # Step size 1 for integers
    plt.grid(False)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares(WCSS)')
    plt.savefig(outPath)
    # plt.show()

def calculate_aic_and_bic_score(num_components:int, X:np.ndarray, log_likelihood:float) -> tuple[float, float]:
    num_samples, num_features = X.shape
    num_parameters = num_components * (num_features + ((num_features * (num_features + 1)) // 2)) + num_components - 1
    aic_score = 2 * num_parameters - 2 * log_likelihood
    bic_score = num_parameters * np.log(num_samples) - 2 * log_likelihood
    return aic_score, bic_score

def plot_AIC_BIC_vs_k(embeddings:np.ndarray, outPath:str) -> None:
    k_max = 21
    aic_scores=[]
    bic_scores=[]
    cluster_values=np.arange(1,k_max)
    for i in range(1,k_max):
        gmm = GaussianMixtureModel(num_components=i)
        gmm.fit(X=embeddings)
        aic, bic = calculate_aic_and_bic_score(i, embeddings, gmm.getLikelihood(embeddings))
        aic_scores.append(aic)
        bic_scores.append(bic)
        likelihood = gmm.getLikelihood(embeddings)
        # print(i, aic, bic, likelihood)

        ### Inbuilt GMM class
        # gmm = GaussianMixture(n_components=i).fit(embeddings)
        # aic = gmm.aic(embeddings)
        # bic = gmm.bic(embeddings)   
        # likelihood = np.sum(gmm.score_samples(embeddings))
        # print(i, aic, bic, likelihood)
        # aic_scores.append(aic)
        # bic_scores.append(bic)

    plt.figure(figsize=(8, 6))

    plt.plot(cluster_values, aic_scores, label='AIC score', marker='o')
    plt.plot(cluster_values, bic_scores, label='BIC score', marker='o')

    plt.title('AIC and BIC v/s Number of clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('AIC/BIC scores')
    plt.legend()
    plt.savefig(outPath)
    # plt.show()
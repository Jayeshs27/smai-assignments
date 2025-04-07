from common import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

embeddings, labels = read_word_embeddings_data()

# ### 6.1 : K-Means based on 2D Visualization(on k2)
k2 = 4
kmeans = kMeansClustering(k=k2)
kmeans.fit(features=embeddings)
cluster_assignments = kmeans.predict(data_features=embeddings) 
print(f"6.1 kMeans clustering using k2={k2}")
print_clusters(labels=labels, cluster_assignments=cluster_assignments)
print("")

# ### 6.2 : PCA + K-Means Clustering
# plot_scree_plot(embeddings, n_components=150, outPath='figures/embeddings_scree_plot.png')
opt_dims = 110
pca = PrincipalComponentAnalysis(num_components=opt_dims)
pca.fit(embeddings)
transform_feats = pca.transform(embeddings)

# plot_k_vs_cost(embeddings=embeddings, outPath='figures/pca_clustering_wcss_vs_k.png')
k_kmeans3 = 8
kmeans = kMeansClustering(k=k_kmeans3)
kmeans.fit(features=transform_feats)
cluster_assignments = kmeans.predict(data_features=transform_feats) 
print(f"6.2 kMeans clustering using k_kmeans3={k_kmeans3}")
print_clusters(labels=labels, cluster_assignments=cluster_assignments)
print("")

# ### 6.3 : GMM based on 2D Visualization (on k2)
gmm = GaussianMixtureModel(num_components=k2)
gmm.fit(X=embeddings)
cluster_memberships = gmm.getMembership()
cluster_assignments = np.argmax(cluster_memberships, axis=1)
print(f"6.3 GMM clustering using k2={k2}")
print_clusters(labels=labels, cluster_assignments=cluster_assignments)
print("")


# 6.4 : PCA + GMMs
# plot_AIC_BIC_vs_k(embeddings=transform_feats, outPath='figures/pca_gmm_aic_bic_vs_k.png')
k_gmm3 = 5
gmm = GaussianMixtureModel(num_components=k_gmm3)
gmm.fit(X=embeddings)
cluster_memberships = gmm.getMembership()
cluster_assignments = np.argmax(cluster_memberships, axis=1)
print(f"6.4 GMM clustering using k_gmm3={k_gmm3}")
print_clusters(labels=labels, cluster_assignments=cluster_assignments)
print("")


from common import *
from sklearn.mixture import GaussianMixture



embeddings, labels = read_word_embeddings_data()

gmm = GaussianMixtureModel(num_components=10)
gmm.fit(X=embeddings)

gmm_skl = GaussianMixture(n_components=10)
gmm_skl.fit(X=embeddings)

### plotting AIC,BIC v/s K 
# plot_AIC_BIC_vs_k(embeddings=embeddings,outPath='figures/gmm_aic_bic_vs_k.png')

# ################### GMM using k_gmm1 ##################
k_gmm1 = 2
gmm = GaussianMixtureModel(num_components=k_gmm1)
gmm.fit(X=embeddings)
likelihood = gmm.getLikelihood(embeddings)
membership_mat = gmm.getMembership()
cluster_assignments = np.argmax(membership_mat, axis=1)

print(f"4.2 GMM clustering using k_gmm1={k_gmm1}")
print_clusters(labels=labels, cluster_assignments=cluster_assignments)
print("")

## testing on 2D Dataset
# df = pd.read_csv('test.csv')
# data = df.to_numpy()
# labels = data[:,-1]
# data = data[:, :2]

# num_clusters = 3
# gmm = GaussianMixtureModel(num_components=num_clusters)
# gmm.fit(X=data)
# membership_probs = gmm.getMembership()
# print(membership_probs)l
# print(gmm.getLikelihood(data))

# ##  plotting the clusters with max prob

# cluster_assigns = np.argmax(membership_probs, axis=1)
# plt.figure(figsize=(8, 6))
# for j in range(num_clusters):  # assuming k=10
#     cluster_points = data[cluster_assigns == j]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {j}')

# plt.title('K-Means Clustering')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# # plt.savefig(f'gmm_clustering_k={num_clusters}.png')
# plt.show()






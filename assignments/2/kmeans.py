from common import *

df = pd.read_feather('../../data/external/word-embeddings.feather')
# df dim : 200 x 2 where second coloumn is 512 dim vector embedding
data = df.to_numpy()
embeddings = np.vstack(data[:,-1])
labels = np.vstack(data[:,:1])
embeddings = embeddings.astype(np.float64)

### 3.1 WCSS v/s k plot
# plot_k_vs_cost(embeddings=embeddings, outPath='figures/wcss_vs_k_plot.png')

#### Kmeans for Optimal Number of Clusters
k_kmeans1 = 7
kmeans = kMeansClustering(k=k_kmeans1)
kmeans.fit(embeddings)
cluster_assigments = kmeans.predict(embeddings)
print(f"3.2 kMeans clustering using k_kmeans1={k_kmeans1}")
print_clusters(labels=labels, cluster_assignments=cluster_assigments)
print("")



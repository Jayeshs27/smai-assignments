from common import *

def plot_2d_projection(transformed_data):
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[:,0], transformed_data[:,1], label='Trasformed Points')
    plt.xlabel('Transformed Feature-1')
    plt.ylabel('Transformed Feature-2')
    plt.savefig('figures/pca_2d_plot.png')
    # plt.show()

def plot_3d_projection(transformed_data):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(transformed_data[:,0], transformed_data[:,1], transformed_data[:,2], marker='o', s=10)
    ax.set_xlabel('Transformed Feature-1')
    ax.set_ylabel('Transformed Feature-2')
    ax.set_zlabel('Transformed Feature-3')
    plt.savefig('figures/pca_3d_plot.png')
    # plt.show()

embeddings, labels = read_word_embeddings_data()

### Transformation to 2-Dim
pca = PrincipalComponentAnalysis(num_components=2)
pca.fit(embeddings)
trans_data_2d = pca.transform(embeddings)
if pca.checkPCA(embeddings):
    print("Data Transformed into 2D Dimensions Successfully")
    plot_2d_projection(transformed_data=trans_data_2d)
else:
    print("Failed to Transform data in 2D Dimensions")


### Transformation to 3-Dim
pca = PrincipalComponentAnalysis(num_components=3)
pca.fit(embeddings)
trans_data_3d = pca.transform(embeddings)
if pca.checkPCA(embeddings):
    print("Data Transformed into 3D Dimensions Successfully")
    plot_3d_projection(transformed_data=trans_data_3d)
else:
    print("Failed to Transform data in 3D Dimensions")

# k2 = 4
# kmeans = kMeansClustering(k=k2)
# kmeans.fit(features=trans_data_2d)
# cluster_assignments = kmeans.predict(data_features=trans_data_2d) 
# plt.figure(figsize=(8, 6))
# # Scatter plot with different colors for each cluster
# for j in range(k2):  # assuming k=10
#     cluster_points = trans_data_2d[cluster_assignments == j]
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {j}')

# plt.title('K-Means Clustering')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.savefig(f'figures/pca_2d_proj_visualizing_clusters.png')
# plt.show()
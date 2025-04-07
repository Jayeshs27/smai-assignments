import scipy.cluster.hierarchy  as hc
import matplotlib.pyplot as plt
from common import *


def plot_dendrogram(data:np.ndarray, dist_metric, dist_method):
    linkage_matrix = hc.linkage(y=data, method=dist_method, metric=dist_metric)
    fig = plt.figure(figsize=(15, 12))
    dgm = hc.dendrogram(linkage_matrix)
    plt.subplots_adjust(left=0.10, right=0.90, top=0.90, bottom=0.10)
    plt.tight_layout()
    plt.savefig(f'figures/hc_dgm_metric={dist_metric}_method={dist_method}')
    # plt.show()

embeddings, labels = read_word_embeddings_data()

# dist_metrics=['euclidean', 'cosine', 'minkowski']
# dist_methods=['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

# for mthd in dist_methods:
#     if mthd == 'centroid' or mthd == 'median' or mthd == 'ward':
#         plot_dendrogram(data=embeddings, dist_metric='euclidean', dist_method=mthd)
#     else:
#         for met in dist_metrics:
#             plot_dendrogram(data=embeddings, dist_metric=met, dist_method=mthd)

k_best1 = 7
k_best2 = 5

linkage_matrix = hc.linkage(y=embeddings, method='ward', metric='euclidean')

clusters_hc_kbest1 = hc.fcluster(linkage_matrix, t=k_best1, criterion='maxclust')
clusters_hc_kbest2 = hc.fcluster(linkage_matrix, t=k_best2, criterion='maxclust')

print(f"8 Hierarchical Clustering with k_best1={k_best1}")
print_clusters(labels=labels, cluster_assignments=clusters_hc_kbest1)
print("")
print(f"8 Hierarchical Clustering with k_best2={k_best2}")
print_clusters(labels=labels, cluster_assignments=clusters_hc_kbest2)
print("")


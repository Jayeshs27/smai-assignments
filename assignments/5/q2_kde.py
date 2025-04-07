from common import *
from kde_model import KDE

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/gmm")))
from gmm import GaussianMixtureModel

def generate_circle_data(num_points, radius, noise, center=(0, 0)):
    angles = 2 * np.pi * np.random.rand(num_points) 
    radii = radius * np.sqrt(np.random.rand(num_points)) 
    x = radii * np.cos(angles) + noise * np.random.randn(num_points) + center[0]
    y = radii * np.sin(angles) + noise * np.random.randn(num_points) + center[1]
    return x, y

def plot_gmm(model, X, num_components):
    membership_probs = model.getMembership()
    labels = np.argmax(membership_probs, axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))
    for cluster in range(num_components):
        cluster_points = X[labels == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'Cluster {cluster}')
        
    ax.set_title("GMM Clusters")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'figures/gmm_clusters_plot_components={num_components}.png')
    plt.show()


def plot_synthetic_dataset(x_large, y_large, x_small, y_small):
    plt.figure(figsize=(10, 10))
    plt.scatter(x_large, y_large, color='gray', alpha=0.5, s=10)
    plt.scatter(x_small, y_small, color='gray', alpha=0.5, s=10)
    plt.axis('equal')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Synthetic Dataset with Two Circles")
    plt.tight_layout()
    plt.savefig('figures/synthetic_dataset.png')
    plt.show()

## 2.2 Synthetic Dataset Generation

x_large, y_large = generate_circle_data(num_points=3000, radius=2, noise=0.2)
x_small, y_small = generate_circle_data(num_points=500, radius=0.2, noise=0.02, center=(1,1))
data = np.vstack((np.column_stack((x_large, y_large)), np.column_stack((x_small, y_small))))
plot_synthetic_dataset(x_large, y_large, x_small, y_small)

## 2.3 KDE v/s GMM

# KDE on Synthetic Dataset
kde = KDE(bandwidth=1.0, kernel='triangular')
kde.fit(data)
kde.visualize() 

# GMM on Synthetic Dataset
gmm = GaussianMixtureModel(num_components=10)
gmm.fit(data)
plot_gmm(gmm, data, num_components=10)





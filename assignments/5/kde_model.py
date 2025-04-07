from common import *

class KDE:
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.data = None
        
    def box_kernel(self, distances):
        return np.where(np.abs(distances) <= 1, 0.5, 0)

    def gaussian_kernel(self, distances):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * distances ** 2)

    def triangular_kernel(self, distances):
        return np.maximum(0, 1 - np.abs(distances))

    def _kernel(self, distances):
        if self.kernel == 'box':
            return self.box_kernel(distances)
        elif self.kernel == 'gaussian':
            return self.gaussian_kernel(distances)
        elif self.kernel == 'triangular':
            return self.triangular_kernel(distances)
        else:
            raise ValueError("Kernel type not supported.")

    def predict(self, x):
        x = np.array(x)
        n, d = self.data.shape
        distances = np.linalg.norm((self.data - x) / self.bandwidth, axis=1)
        kernel_values = self._kernel(distances)
        density = np.sum(kernel_values) / (n * self.bandwidth ** d)
        return density

    
    def fit(self, data):
        self.data = np.array(data)

    def visualize(self):
        x_min, y_min = self.data.min(axis=0) - 1
        x_max, y_max = self.data.max(axis=0) + 1
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), 
                                     np.linspace(y_min, y_max, 100))
        
        density_grid = np.zeros(x_grid.shape)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                density_grid[i, j] = self.predict([x_grid[i, j], y_grid[i, j]])
                
        plt.figure(figsize=(12, 8))
        plt.contourf(x_grid, y_grid, density_grid, cmap='viridis')
        plt.scatter(self.data[:, 0], self.data[:, 1], color='red', s=10, label='Data points')
        plt.colorbar(label='Density')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title(f"KDE Density Estimation (Kernel: {self.kernel})")
        plt.tight_layout()
        plt.savefig(f"figures/kde_visualisation.png")
        plt.show()


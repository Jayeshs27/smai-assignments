import numpy as np
from collections import Counter


class distanceMethod:
    valid_methods=['euclidean','manhattan','cosine']
    def __init__(self, method='euclidean'):
        if method not in self.valid_methods:
            raise ValueError("Not a valid distance method")
        self.method = method

    def calculate_distance(self, x_train, x_test):
        if self.method == 'euclidean':
            distance = np.sqrt(np.sum((x_train - x_test) ** 2, axis=1))
            return distance
        elif self.method == 'manhattan':
            distance = np.sum(np.abs(x_train - x_test), axis=1)
            return distance
        else:
            x_train_norm = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
            x_test_norm = x_test / np.linalg.norm(x_test)
            distance = 1 - np.dot(x_train_norm, x_test_norm)
            return distance

class kNNClassifier:
    x_train = None
    y_train = None

    def __init__(self, k:int=1, distance_method:str='euclidean'):
        self.k = k
        self.distance_method = distance_method

    def fit(self, x_train, y_train) -> None:
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test) -> str:
        dist_method = distanceMethod(method=self.distance_method)

        distance = dist_method.calculate_distance(self.x_train, x_test)
        indices = np.argsort(distance)
        labels = self.y_train[indices[:self.k]]

        # labels_list = labels.tolist()
        counts = Counter(labels)
        most_common_label = counts.most_common(1)[0][0]

        return most_common_label

# import numpy as np
# from collections import Counter

# class distanceMethod:
#     valid_methods = ['euclidean', 'manhattan', 'cosine']
    
#     def __init__(self, method='euclidean'):
#         if method not in self.valid_methods:
#             raise ValueError("Not a valid distance method")
#         self.method = method

#     def calculate_distance(self, x_train, x_test):
#         if self.method == 'euclidean':
#             distance = np.sqrt(np.sum((x_train[:, np.newaxis] - x_test) ** 2, axis=2))
#         elif self.method == 'manhattan':
#             distance = np.sum(np.abs(x_train[:, np.newaxis] - x_test), axis=2)
#         else:
#             x_train_norm = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
#             x_test_norm = x_test / np.linalg.norm(x_test, axis=1, keepdims=True)
#             distance = 1 - np.dot(x_train_norm, x_test_norm.T)
#         return distance


# class kNNClassifier:
#     x_train = None
#     y_train = None

#     def __init__(self, k: int = 1, distance_method: str = 'euclidean'):
#         self.k = k
#         self.distance_method = distance_method

#     def fit(self, x_train, y_train) -> None:
#         self.x_train = x_train
#         self.y_train = y_train

#     def predict(self, X_test) -> np.ndarray:
#         dist_method = distanceMethod(method=self.distance_method)

#         # Calculate distances between each x_test and all x_train
#         distances = dist_method.calculate_distance(self.x_train, X_test)

#         # Find the k nearest labels for each test sample
#         nearest_neighbors = np.argsort(distances, axis=0)[:self.k, :]

#         # Get the labels of the k nearest neighbors
#         nearest_labels = self.y_train[nearest_neighbors]

#         # Determine the most common label for each test sample
#         y_pred = np.array([Counter(nearest_labels[:, i]).most_common(1)[0][0] for i in range(X_test.shape[0])])

#         return y_pred

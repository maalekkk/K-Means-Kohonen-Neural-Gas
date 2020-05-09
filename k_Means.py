import warnings
import numpy as np
from copy import deepcopy, copy


def euclidean_distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


class KMeans:

    def __init__(self, centroids):
        self.centroids = centroids  # list of centroids

    def quantization_error(self, dataset, clusters):
        quantization_error = 0
        for i in range(len(dataset)):
            index_of_centroid = int(clusters[i])
            quantization_error += euclidean_distance(dataset[i], self.centroids[index_of_centroid], None)
        quantization_error /= len(dataset)
        return quantization_error

    def fit(self, dataset):
        num_of_classes = len(self.centroids)
        clusters = np.zeros(len(dataset))
        # Centroids history
        centroids_history = [copy(self.centroids)]
        # To store quantization errors
        quantization_errors = []
        # Flag to 1st iteration
        first_iteration = True
        # Loop will run till the error becomes zero
        error = 1
        while error > 0.005:
            # Assigning each value to its closest cluster
            for i in range(len(dataset)):
                distances = euclidean_distance(dataset[i], self.centroids)
                cluster = np.argmin(distances)
                clusters[i] = cluster
            # Only one call in first iteration
            if first_iteration:
                first_iteration = False
                quantization_error = self.quantization_error(dataset, clusters)
                quantization_errors.append(quantization_error)
            # Storing the old centroid values
            old_centroids = deepcopy(self.centroids)
            # Finding the new centroids by taking the average value
            for i in range(num_of_classes):
                points = [dataset[j] for j in range(len(dataset)) if clusters[j] == i]
                if len(points) != 0:
                    self.centroids[i] = np.mean(points, axis=0)

            # Calculate quantization error
            quantization_error = self.quantization_error(dataset, clusters)
            quantization_errors.append(quantization_error)
            # Store actual error = difference in two iterations
            error = euclidean_distance(self.centroids, old_centroids, None)
            # Store actual position of centroids
            centroids_history.append(copy(self.centroids))
        return centroids_history, quantization_errors

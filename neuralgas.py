import numpy as np
from copy import deepcopy


def euclidean_distance(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


class NeuralGas:
    def __init__(self, weights):
        self.weights = weights  # list of weights

    def quantization_error(self, dataset):
        quantization_error = 0
        for i in range(len(dataset)):
            closest_weights, index = self.find_closest(self.weights, dataset[i])
            quantization_error += euclidean_distance(dataset[i], closest_weights, None)
        quantization_error /= len(dataset)
        return quantization_error

    def find_closest(self, weights, x):
        weights_vector = weights[0]
        index = 0
        dis = euclidean_distance(weights_vector, x, 0)
        for i in range(len(weights)):
            if euclidean_distance(self.weights[i], x, 0) < dis:
                dis = euclidean_distance(self.weights[i], x, 0)
                weights_vector = self.weights[i]
                index = i
        return weights_vector, index

    def fit(self, dataset, l_rate=0.5, neighbour_rate=0.7):
        # Store index of neuron
        index = np.arange(len(self.weights))
        # Store weights
        weights_history = [deepcopy(self.weights)]
        # To store quantization errors
        quantization_errors = []
        # Calculate first quantization error before 1st iteration
        quantization_error = self.quantization_error(dataset)
        quantization_errors.append(quantization_error)
        # Error = different between last and actual weights position
        error = 1
        while error > 0.005:
            np.random.shuffle(dataset)
            old_weights = deepcopy(self.weights)
            for x in dataset:
                distances = np.zeros(len(self.weights))
                for i in range(len(self.weights)):
                    distances[i] = euclidean_distance(x, self.weights[i], None)
                self.weights = [x for _, x in sorted(zip(distances, self.weights))]
                index = [x for _, x in sorted(zip(distances, index))]
                old_weights = [x for _, x in sorted(zip(distances, old_weights))]
                for i in range(len(self.weights)):
                    h = np.exp(-i / neighbour_rate)
                    for j in range(len(self.weights[i])):
                        self.weights[i][j] += (l_rate * h * (
                                x[j] - self.weights[i][j]))

            # Calculate quantization error
            quantization_error = self.quantization_error(dataset)
            quantization_errors.append(quantization_error)

            # Save new weights and new error
            weights_history.append(deepcopy([x for _, x in sorted(zip(index, self.weights))]))
            error = euclidean_distance(np.array(self.weights), old_weights, None)

            # Changing coefficients
            l_rate *= 0.8
            neighbour_rate *= 0.8
        return np.array(weights_history), quantization_errors

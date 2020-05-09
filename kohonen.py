import numpy as np
from copy import deepcopy


def euclidean_distance(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)


class Kohonen:
    def __init__(self, weights):
        self.weights = weights  # list of weights
        self.amount_of_win = np.full(len(self.weights), 1)  # for Conscience Winner Takes All

    def quantization_error(self, dataset, mechanism_of_conscience):
        quantization_error = 0
        for i in range(len(dataset)):
            closest_weights, index = self.find_closest(self.weights, dataset[i], mechanism_of_conscience, True)
            quantization_error += euclidean_distance(dataset[i], closest_weights, None)
        quantization_error /= len(dataset)
        return quantization_error

    def find_closest(self, weights, x, mechanism_of_conscience, quantization_error=False):
        weights_vector = weights[0]
        index = 0
        dis = euclidean_distance(weights_vector, x, 0)
        if not quantization_error and mechanism_of_conscience:
            dis *= self.amount_of_win[0]
        for i in range(len(weights)):
            tmp = euclidean_distance(self.weights[i], x, 0)
            if not quantization_error and mechanism_of_conscience:
                tmp *= self.amount_of_win[i]
            if tmp < dis:
                dis = tmp
                weights_vector = self.weights[i]
                index = i
        if mechanism_of_conscience:
            self.amount_of_win[index] += 1
        return weights_vector, index

    def fit_wta(self, dataset, mechanism_of_conscience, l_rate=0.5):
        # Store weights
        weights_history = [deepcopy(self.weights)]
        # Error = different between last and actual weights position
        error = 1
        # To store quantization errors
        quantization_errors = []
        # Calculate first quantization error before 1st iteration
        quantization_error = self.quantization_error(dataset, mechanism_of_conscience)
        quantization_errors.append(quantization_error)

        while error > 0.005:
            np.random.shuffle(dataset)
            old_weights = deepcopy(self.weights)
            for x in dataset:
                closest_weights, index = self.find_closest(self.weights, x, mechanism_of_conscience)
                for i in range(len(closest_weights)):
                    closest_weights[i] += l_rate * (x[i] - closest_weights[i])

            # Calculate quantization error
            quantization_error = self.quantization_error(dataset, mechanism_of_conscience)
            quantization_errors.append(quantization_error)

            # Save new weights and new error
            weights_history.append(deepcopy(self.weights))
            error = euclidean_distance(np.array(self.weights), old_weights, None)

            # Changing coefficients
            l_rate *= 0.8
        return np.array(weights_history), quantization_errors

    def fit_wtm(self, dataset, l_rate=0.5, neighbour_rate=0.7):
        # Store weights
        weights_history = [deepcopy(self.weights)]
        # Error = different between last and actual weights position
        error = 1
        # To store quantization errors
        quantization_errors = []
        # Calculate first quantization error before 1st iteration
        quantization_error = self.quantization_error(dataset, False)
        quantization_errors.append(quantization_error)

        while error > 0.005:
            np.random.shuffle(dataset)
            old_weights = deepcopy(self.weights)
            for x in dataset:
                closest_weights, index = self.find_closest(self.weights, x, False)
                for i in range(len(self.weights)):
                    rho = np.abs(i - index)
                    if rho <= 1:
                        h = np.exp((-rho ** 2) / (2 * neighbour_rate ** 2))
                        for j in range(len(self.weights[i])):
                            self.weights[i][j] += (l_rate * h * (
                                    x[j] - self.weights[i][j]))

            # Calculate quantization error
            quantization_error = self.quantization_error(dataset, False)
            quantization_errors.append(quantization_error)

            # Save new weights and new error
            weights_history.append(deepcopy(self.weights))
            error = euclidean_distance(np.array(self.weights), old_weights, None)

            # Changing coefficients
            l_rate *= 0.8
            neighbour_rate *= 0.8
        return np.array(weights_history), quantization_errors

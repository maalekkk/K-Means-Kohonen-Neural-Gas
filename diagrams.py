import csv

from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
from scipy.spatial.qhull import Voronoi

import k_Means as kMn
import kohonen as khn
import neuralgas as n_gas
import initialize as init


def clear_plot(ax):
    ax.cla()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(color='black', linestyle='-', linewidth=0.3)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')


def draw_weights_position(i, ax, dataset, weights, kohonen, title):
    clear_plot(ax)
    ax.set_title(title)
    label = 'x\nIteration: {0}'.format(i)
    ax.set_xlabel(label)
    ax.set_ylabel('y')
    ax.scatter(dataset[:, 0], dataset[:, 1], c='black')
    ax.scatter(weights[i][:, 0], weights[i][:, 1], c='red')
    if kohonen:
        ax.plot(weights[i][:, 0], weights[i][:, 1], c='red')
    return ax


def error_plot(errors_history, x_label, y_label, title, file_name):
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(np.arange(len(errors_history)), errors_history)
    ax.grid()
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    fig.savefig(file_name)


def centroids_position_gif(weights_history, dataset, file_name, kohonen, title):
    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, draw_weights_position, frames=len(weights_history), fargs=(ax, dataset, weights_history,
                                                                                         kohonen, title))
    anim.save(file_name, dpi=100, writer='imagemagick')


def graph_with_change_of_quantization_error(figure_type, algorithm_name, kohonen_method=""):
    dataset_one_fig = init.generate_data(figure_type, 220, False)
    dataset_two_fig = init.generate_data(figure_type, 220, True)
    fig_1, ax_1 = plt.subplots()
    fig_2, ax_2 = plt.subplots()
    ax_1.set(xlabel='iteration', ylabel='quantization error',
             title=algorithm_name + kohonen_method + ' - quantization error\none figure')
    ax_1.grid()
    ax_2.set(xlabel='iteration', ylabel='quantization error',
             title=algorithm_name + kohonen_method + ' - quantization error\ntwo figures')
    ax_2.grid()
    if algorithm_name == "k-Means":
        for i in range(2, 21, 2):
            centroids_1 = init.generate_centroids_from_data(dataset_one_fig, i)
            centroids_2 = init.generate_centroids_from_data(dataset_two_fig, i)
            k_means_1 = kMn.KMeans(centroids_1)
            k_means_2 = kMn.KMeans(centroids_2)
            centroids_history_1, quantization_errors_1 = k_means_1.fit(dataset_one_fig)
            centroids_history_2, quantization_errors_2 = k_means_2.fit(dataset_two_fig)
            ax_1.plot(np.arange(len(quantization_errors_1)), quantization_errors_1, label=str(i) + ' neurons')
            ax_2.plot(np.arange(len(quantization_errors_2)), quantization_errors_2, label=str(i) + ' neurons')
    elif algorithm_name == "Kohonen":
        if kohonen_method == "WTA":
            for i in range(2, 21, 2):
                centroids_1 = init.generate_centroids_from_data(dataset_one_fig, i)
                centroids_2 = init.generate_centroids_from_data(dataset_two_fig, i)
                kohonen_1 = khn.Kohonen(centroids_1)
                kohonen_2 = khn.Kohonen(centroids_2)
                centroids_history_1, quantization_errors_1 = kohonen_1.fit_wta(dataset_one_fig, False)
                centroids_history_2, quantization_errors_2 = kohonen_2.fit_wta(dataset_one_fig, False)
                ax_1.plot(np.arange(len(quantization_errors_1)), quantization_errors_1, label=str(i) + ' neurons')
                ax_2.plot(np.arange(len(quantization_errors_2)), quantization_errors_2, label=str(i) + ' neurons')
        elif kohonen_method == "CWTA":
            for i in range(2, 21, 2):
                centroids_1 = init.generate_centroids_from_data(dataset_one_fig, i)
                centroids_2 = init.generate_centroids_from_data(dataset_two_fig, i)
                kohonen_1 = khn.Kohonen(centroids_1)
                kohonen_2 = khn.Kohonen(centroids_2)
                centroids_history_1, quantization_errors_1 = kohonen_1.fit_wta(dataset_one_fig, True)
                centroids_history_2, quantization_errors_2 = kohonen_2.fit_wta(dataset_one_fig, True)
                ax_1.plot(np.arange(len(quantization_errors_1)), quantization_errors_1, label=str(i) + ' neurons')
                ax_2.plot(np.arange(len(quantization_errors_2)), quantization_errors_2, label=str(i) + ' neurons')
        elif kohonen_method == "WTM":
            for i in range(2, 21, 2):
                centroids_1 = init.generate_centroids_from_data(dataset_one_fig, i)
                centroids_2 = init.generate_centroids_from_data(dataset_two_fig, i)
                kohonen_1 = khn.Kohonen(centroids_1)
                kohonen_2 = khn.Kohonen(centroids_2)
                centroids_history_1, quantization_errors_1 = kohonen_1.fit_wtm(dataset_one_fig)
                centroids_history_2, quantization_errors_2 = kohonen_2.fit_wtm(dataset_one_fig)
                ax_1.plot(np.arange(len(quantization_errors_1)), quantization_errors_1, label=str(i) + ' neurons')
                ax_2.plot(np.arange(len(quantization_errors_2)), quantization_errors_2, label=str(i) + ' neurons')
    elif algorithm_name == "NeuralGas":
        for i in range(2, 21, 2):
            centroids_1 = init.generate_centroids_from_data(dataset_one_fig, i)
            centroids_2 = init.generate_centroids_from_data(dataset_two_fig, i)
            neural_gas_1 = n_gas.NeuralGas(centroids_1)
            neural_gas_2 = n_gas.NeuralGas(centroids_2)
            centroids_history_1, quantization_errors_1 = neural_gas_1.fit(dataset_one_fig)
            centroids_history_2, quantization_errors_2 = neural_gas_2.fit(dataset_two_fig)
            ax_1.plot(np.arange(len(quantization_errors_1)), quantization_errors_1, label=str(i) + ' neurons')
            ax_2.plot(np.arange(len(quantization_errors_2)), quantization_errors_2, label=str(i) + ' neurons')
    ax_1.legend()
    ax_2.legend()
    fig_1.savefig('results/task1,2/1/' + algorithm_name + kohonen_method + '_quant_errors_one_figure.png')
    fig_2.savefig('results/task1,2/1/' + algorithm_name + kohonen_method + '_quant_errors_two_figures.png')


def compare_neurons(neuron1, neuron2):
    if len(neuron1) is not len(neuron2):
        return False
    if neuron1[0] == neuron2[0] and neuron1[1] == neuron2[1]:
        return True
    return False


def table_with_avg_of_quantization_error(algorithm_name, two_fig, dataset):
    with open('results/task1,2/2/' + algorithm_name + ('_two' if two_fig is True else '_one') + '_table.txt',
              'w',
              newline='') as fileOut:
        writer_1 = csv.writer(fileOut, delimiter=' ',
                              quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer_1.writerow(
            ['method', 'weights_type', 'l_rate', 'n_rate', 'avg_of_final_quantization_error', 'standard_deviation',
             'minimal_error', 'avg_number_of_inactive_neurons',
             'standard_deviation'])

        num_of_classes = 20

        if algorithm_name == "k-Means":
            for i in range(10):
                print(i, '-ta iteracja k-Means')
                random_centroids = bool(np.random.randint(0, 2))
                final_quantization_errors = []
                inactive_neurons_history = []
                for j in range(100):
                    inactive_neurons = 0
                    if not random_centroids:
                        centroids = init.generate_centroids_from_data(dataset, num_of_classes)
                    else:
                        centroids = init.generate_random_centroids(num_of_classes)
                    k_means = kMn.KMeans(centroids)
                    centroids_history, quantization_errors = k_means.fit(dataset)
                    for k in range(len(centroids)):
                        if compare_neurons(centroids_history[0][k], centroids_history[len(centroids_history) - 1][k]):
                            inactive_neurons += 1
                    inactive_neurons_history.append(inactive_neurons)
                    final_quantization_errors.append(quantization_errors[len(quantization_errors) - 1])
                writer_1.writerow(
                    ['empty', 'random' if random_centroids is True else 'dataset', 'empty', 'empty',
                     np.average(final_quantization_errors), np.std(final_quantization_errors),
                     np.min(final_quantization_errors), np.average(inactive_neurons_history),
                     np.std(inactive_neurons_history)])

        elif algorithm_name == "Kohonen":
            for i in range(10):
                print(i, '-ta iteracja Kohonen')
                l_rate = np.random.uniform(0.001, 1)
                neighbour_rate = np.random.uniform(0.001, 1)
                random_weights = bool(np.random.randint(0, 2))
                method = np.random.randint(0, 3)
                final_quantization_errors = []
                inactive_neurons_history = []
                for j in range(100):
                    inactive_neurons = 0
                    if not random_weights:
                        weights = init.generate_centroids_from_data(dataset, num_of_classes)
                    else:
                        weights = init.generate_random_centroids(num_of_classes)
                    kohonen = khn.Kohonen(weights)
                    if method == 0:
                        weights_history, quantization_errors = kohonen.fit_wtm(dataset, l_rate, neighbour_rate)
                    elif method == 1:
                        weights_history, quantization_errors = kohonen.fit_wta(dataset, True, l_rate)
                    elif method == 2:
                        weights_history, quantization_errors = kohonen.fit_wta(dataset, False, l_rate)
                    for k in range(len(weights)):
                        if compare_neurons(weights_history[0][k],
                                           weights_history[len(weights_history) - 1][k]):
                            inactive_neurons += 1
                    inactive_neurons_history.append(inactive_neurons)
                    final_quantization_errors.append(quantization_errors[len(quantization_errors) - 1])
                writer_1.writerow(
                    [('WTM' if method == 0 else ('CWTA' if method == 1 else 'WTA')),
                     'random' if random_weights is True else 'dataset', l_rate,
                     neighbour_rate if method == 0 else 'empty',
                     np.average(final_quantization_errors), np.std(final_quantization_errors),
                     np.min(final_quantization_errors), np.average(inactive_neurons_history),
                     np.std(inactive_neurons_history)])

        elif algorithm_name == "NeuralGas":
            for i in range(10):
                print(i, '-ta iteracja NeuralGas')
                l_rate = np.random.uniform(0.001, 1)
                neighbour_rate = np.random.uniform(0.001, 1)
                random_weights = bool(np.random.randint(0, 2))
                final_quantization_errors = []
                inactive_neurons_history = []
                for j in range(100):
                    inactive_neurons = 0
                    if not random_weights:
                        weights = init.generate_centroids_from_data(dataset, num_of_classes)
                    else:
                        weights = init.generate_random_centroids(num_of_classes)
                    neural_gas = n_gas.NeuralGas(weights)
                    weights_history, quantization_errors = neural_gas.fit(dataset, l_rate, neighbour_rate)
                    for k in range(len(weights)):
                        if compare_neurons(weights_history[0][k],
                                           weights_history[len(weights_history) - 1][k]):
                            inactive_neurons += 1
                    inactive_neurons_history.append(inactive_neurons)
                    final_quantization_errors.append(quantization_errors[len(quantization_errors) - 1])
                writer_1.writerow(
                    [algorithm_name, 'random' if random_weights is True else 'dataset', l_rate, neighbour_rate,
                     np.average(final_quantization_errors), np.std(final_quantization_errors),
                     np.min(final_quantization_errors), np.average(inactive_neurons_history),
                     np.std(inactive_neurons_history)])


def generate_positions_of_centroids(num_of_classes, centroids_history,
                                    ax):
    centroids = []
    for i in range(num_of_classes):
        centroids.append(list())
        for j in range(len(centroids_history)):
            centroids[i].append(list())
            centroids[i][j].append(centroids_history[j][i][0])
            centroids[i][j].append(centroids_history[j][i][1])
    centroids = np.array(centroids)
    for i in range(num_of_classes):
        ax.scatter(centroids[i][:, 0], centroids[i][:, 1], s=8)
        ax.plot(centroids[i][:, 0], centroids[i][:, 1], )
    ax.axis('equal')


def positions_of_centroids(figure_type, two_fig, algorithm_name, num_of_classes, random_centroids, kohonen_method=""):
    if two_fig:
        dataset = init.generate_data(figure_type, 220, True)
    else:
        dataset = init.generate_data(figure_type, 220, False)

    fig_1, ax_1 = plt.subplots()
    fig_2, ax_2 = plt.subplots()
    ax_1.set(xlabel='x', ylabel='y',
             title=algorithm_name + kohonen_method + ' - positions of centroids\n' + (
                 'two' if two_fig is True else 'one') + ' figure\n' + str(
                 num_of_classes) + ' centroids')
    ax_2.set(xlabel='x', ylabel='y',
             title=algorithm_name + kohonen_method + ' - assignment to centroids\n' + (
                 'two' if two_fig is True else 'one') + ' figure\n' + str(
                 num_of_classes) + ' centroids')
    if random_centroids:
        centroids = init.generate_random_centroids(num_of_classes)
    else:
        centroids = init.generate_centroids_from_data(dataset, num_of_classes)
    ax_1.scatter(dataset[:, 0], dataset[:, 1], c='black', s=8)
    if algorithm_name == "k-Means":
        k_means_1 = kMn.KMeans(centroids)
        centroids_history, quantization_errors = k_means_1.fit(dataset)
        generate_positions_of_centroids(num_of_classes, centroids_history, ax_1)
        my_diagram(centroids_history, ax_2, dataset)
    elif algorithm_name == "Kohonen":
        if kohonen_method == "WTA":
            kohonen_1 = khn.Kohonen(centroids)
            centroids_history_1, quantization_errors_1 = kohonen_1.fit_wta(dataset, False)
            generate_positions_of_centroids(num_of_classes, centroids_history_1, ax_1)
            my_diagram(centroids_history_1, ax_2, dataset)
        elif kohonen_method == "CWTA":
            kohonen_1 = khn.Kohonen(centroids)
            centroids_history_1, quantization_errors_1 = kohonen_1.fit_wta(dataset, True)
            generate_positions_of_centroids(num_of_classes, centroids_history_1, ax_1)
            my_diagram(centroids_history_1, ax_2, dataset)
        elif kohonen_method == "WTM":
            kohonen_1 = khn.Kohonen(centroids)
            centroids_history_1, quantization_errors_1 = kohonen_1.fit_wtm(dataset)
            generate_positions_of_centroids(num_of_classes, centroids_history_1, ax_1)
            my_diagram(centroids_history_1, ax_2, dataset)
    elif algorithm_name == "NeuralGas":
        neural_gas_1 = n_gas.NeuralGas(centroids)
        centroids_history_1, quantization_errors_1 = neural_gas_1.fit(dataset)
        generate_positions_of_centroids(num_of_classes, centroids_history_1, ax_1)
        my_diagram(centroids_history_1, ax_2, dataset)

    fig_1.savefig('results/task1,2/3/' + algorithm_name + kohonen_method + '_' + str(
        num_of_classes) + '_centroids_positions_' + ('two' if two_fig is True else 'one') + '_figure.png')
    fig_2.savefig('results/task1,2/3/' + algorithm_name + kohonen_method + '_' + str(
        num_of_classes) + '_assignment_to_centroids_' + ('two' if two_fig is True else 'one') + '_figure.png')

#
# def voronoi_diagram(centroids_history_1, centroids_history_2, ax_3, ax_4, dataset_one_fig, dataset_two_fig):
#     final_centroids_1 = centroids_history_1[len(centroids_history_1) - 1]
#     final_centroids_2 = centroids_history_2[len(centroids_history_2) - 1]
#     diagram_voronoi_1 = Voronoi(final_centroids_1)
#     diagram_voronoi_2 = Voronoi(final_centroids_2)
#     voronoi_plot_2d(diagram_voronoi_1, ax_3)
#     voronoi_plot_2d(diagram_voronoi_2, ax_4)
#     ax_3.scatter(dataset_one_fig[:, 0], dataset_one_fig[:, 1], c='black', s=6)
#     ax_4.scatter(dataset_two_fig[:, 0], dataset_two_fig[:, 1], c='black', s=6)


def my_diagram(centroids_history, ax, dataset):
    final_centroids_1 = centroids_history[len(centroids_history) - 1]
    clusters_1 = np.zeros(len(dataset))
    for i in range(len(dataset)):
        distances = kMn.euclidean_distance(dataset[i], final_centroids_1)
        cluster = np.argmin(distances)
        clusters_1[i] = cluster
    for i in range(len(final_centroids_1)):
        points = [dataset[j] for j in range(len(dataset)) if clusters_1[j] == i]
        if len(points) != 0:
            x, y = [], []
            for j in range(len(points)):
                x.append(points[j][0])
                y.append(points[j][1])
            ax.scatter(x, y, s=10)
    ax.scatter(final_centroids_1[:, 0], final_centroids_1[:, 1], c='black', s=25)
    ax.axis('equal')

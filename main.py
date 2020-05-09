import csv

import initialize as init
import k_Means as kMn
import kohonen as khn
import neuralgas as n_gas
import diagrams
import imageCompression as imgComp

# WarmUp 1
# k_Means = kMn.KMeans(centroids)
# centroids_history, quantization_errors = k_Means.fit(dataset, num_of_classes)
# diagrams.error_plot(quantization_errors, 'iteration', 'quantization error', 'k-means', 'kMeans.png')
# diagrams.centroids_position_gif(centroids_history, dataset, 'kMeans.gif', False, 'k-Means - weights positions')

# WarmUp 2
# kohonen = khn.Kohonen(random_centroids)
# centroids_history, quantization_errors = kohonen.fit_wta(dataset, mechanism_of_conscience=False)
# diagrams.error_plot(quantization_errors, 'iteration', 'quantization error', 'kohonen', 'kohonenWta.png')
# diagrams.centroids_position_gif(centroids_history, dataset, 'kohonenWta.gif', True, 'Kohonen WTA - weights positions')

# kohonen = khn.Kohonen(random_centroids) centroids_history, quantization_errors = kohonen.fit_wta(dataset,
# mechanism_of_conscience=True) diagrams.error_plot(quantization_errors, 'iteration', 'quantization error',
# 'kohonen', 'kohonenCwta.png') diagrams.centroids_position_gif(centroids_history, dataset, 'kohonenCwta.gif', True,
# 'Kohonen CWTA - weights positions')

# kohonen = khn.Kohonen(centroids)
# centroids_history, quantization_errors = kohonen.fit_wtm(dataset)
# diagrams.error_plot(quantization_errors, 'iteration', 'quantization error', 'kohonen', 'kohonenWtm.png')
# diagrams.centroids_position_gif(centroids_history, dataset, 'kohonenWtm.gif', True, 'Kohonen WTM - weights positions')

# WarmUp 3
# neural_gas = n_gas.NeuralGas(centroids)
# centroids_history, quantization_errors = neural_gas.fit(dataset)
# diagrams.error_plot(quantization_errors, 'iteration', 'quantization error', 'neuralGas', 'neuralGas.png')
# diagrams.centroids_position_gif(centroids_history, dataset, 'neuralGas.gif', False, 'Neural Gas - weights positions')


# dataset_one_figure = init.generate_data("square", 220, False)
# dataset_two_figure = init.generate_data("square", 220, True)

# Task 1

# Point 1
# diagrams.graph_with_change_of_quantization_error('square', 'Kohonen', 'WTM')
# diagrams.graph_with_change_of_quantization_error('square', 'Kohonen', 'WTA')
# diagrams.graph_with_change_of_quantization_error('square', 'Kohonen', 'CWTA')
# diagrams.graph_with_change_of_quantization_error('square', 'NeuralGas')

# Point 2
# diagrams.table_with_avg_of_quantization_error('Kohonen', False, dataset_one_figure)
# diagrams.table_with_avg_of_quantization_error('Kohonen', True, dataset_two_figure)
# diagrams.table_with_avg_of_quantization_error('NeuralGas', False, dataset_one_figure)
# diagrams.table_with_avg_of_quantization_error('NeuralGas', True, dataset_two_figure)

# Point 3
# diagrams.positions_of_centroids('square', True, 'Kohonen', 2, True, 'WTM')
# diagrams.positions_of_centroids('square', False, 'Kohonen', 2, True, 'WTM')
# diagrams.positions_of_centroids('square', True, 'Kohonen', 10, True, 'WTM')
# diagrams.positions_of_centroids('square', False, 'Kohonen', 10, True, 'WTM')
# diagrams.positions_of_centroids('square', True,  'NeuralGas', 2, True)
# diagrams.positions_of_centroids('square', False,  'NeuralGas', 2, True)
# diagrams.positions_of_centroids('square', True, 'NeuralGas', 10, True)
# diagrams.positions_of_centroids('square', False, 'NeuralGas', 10, True)


# Task 2
# Point 1
# diagrams.graph_with_change_of_quantization_error('square', 'k-Means')

# Point 2
# diagrams.table_with_avg_of_quantization_error('k-Means', False, dataset_one_figure)
# diagrams.table_with_avg_of_quantization_error('k-Means', True, dataset_two_figure)

# Point 3
# diagrams.positions_of_centroids('square', True, 'k-Means', 2, True)
# diagrams.positions_of_centroids('square', False, 'k-Means', 2, True)
# diagrams.positions_of_centroids('square', True, 'k-Means', 10, True)
# diagrams.positions_of_centroids('square', False, 'k-Means', 10, True)

# Task 3
with open('results/task3/quantization_table.txt',
          'w',
          newline='') as fileOut:
    writer_1 = csv.writer(fileOut, delimiter=' ',
                          quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    writer_1.writerow(
        ['name', 'final_error'])

    for i in range(1, 4, 1):
        print('Photo: ', i)
        centroids = 4
        while centroids < 257:
            error = imgComp.compress_photo('photo' + str(i) + '.png', centroids)
            writer_1.writerow([str(i) + '_' + str(centroids), error])
            centroids *= 2

    # print('Obrazek 1')
    # error = imgComp.compress_photo('photo1.png', 4)
    # writer_1.writerow(['1_4', error])
    # error = imgComp.compress_photo('photo1.png', 8)
    # writer_1.writerow(['1_8', error])
    # error = imgComp.compress_photo('photo1.png', 16)
    # writer_1.writerow(['1_16', error])
    # error = imgComp.compress_photo('photo1.png', 32)
    # writer_1.writerow(['1_32', error])
    # error = imgComp.compress_photo('photo1.png', 64)
    # writer_1.writerow(['1_64', error])
    # error = imgComp.compress_photo('photo1.png', 128)
    # writer_1.writerow(['1_128', error])
    # error = imgComp.compress_photo('photo1.png', 256)
    # writer_1.writerow(['1_256', error])
    #
    # print('Obrazek 2')
    # error = imgComp.compress_photo('photo2.png', 4)
    # writer_1.writerow(['2_4', error])
    # error = imgComp.compress_photo('photo2.png', 8)
    # writer_1.writerow(['2_8', error])
    # error = imgComp.compress_photo('photo2.png', 16)
    # writer_1.writerow(['2_16', error])
    # error = imgComp.compress_photo('photo2.png', 32)
    # writer_1.writerow(['2_32', error])
    # error = imgComp.compress_photo('photo2.png', 64)
    # writer_1.writerow(['2_64', error])
    # error = imgComp.compress_photo('photo2.png', 128)
    # writer_1.writerow(['2_128', error])
    # error = imgComp.compress_photo('photo2.png', 256)
    # writer_1.writerow(['2_256', error])
    #
    # print('Obrazek 3')
    # error = imgComp.compress_photo('photo3.png', 4)
    # writer_1.writerow(['3_4', error])
    # error = imgComp.compress_photo('photo3.png', 8)
    # writer_1.writerow(['3_8', error])
    # error = imgComp.compress_photo('photo3.png', 16)
    # writer_1.writerow(['3_16', error])
    # error = imgComp.compress_photo('photo3.png', 32)
    # writer_1.writerow(['3_32', error])
    # error = imgComp.compress_photo('photo3.png', 64)
    # writer_1.writerow(['3_64', error])
    # error = imgComp.compress_photo('photo3.png', 128)
    # writer_1.writerow(['3_128', error])
    # error = imgComp.compress_photo('photo3.png', 256)
    # writer_1.writerow(['3_256', error])
    #
    # print('Obrazek 4')
    # error = imgComp.compress_photo('photo4.png', 4)
    # writer_1.writerow(['4_4', error])
    # error = imgComp.compress_photo('photo4.png', 8)
    # writer_1.writerow(['4_8', error])
    # error = imgComp.compress_photo('photo4.png', 16)
    # writer_1.writerow(['4_16', error])
    # error = imgComp.compress_photo('photo4.png', 32)
    # writer_1.writerow(['4_32', error])
    # error = imgComp.compress_photo('photo4.png', 64)
    # writer_1.writerow(['4_64', error])
    # error = imgComp.compress_photo('photo4.png', 128)
    # writer_1.writerow(['4_128', error])
    # error = imgComp.compress_photo('photo4.png', 256)
    # writer_1.writerow(['4_256', error])

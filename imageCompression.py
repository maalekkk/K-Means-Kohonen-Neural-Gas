import numpy as np
from PIL import Image
from k_Means import KMeans
import initialize as init


def compress_photo(filename, num_of_centroids, dimension=3):
    img = Image.open(filename)
    pixel_array = np.asarray(img)
    image_height = img.height
    image_width = img.width
    pixel_np1 = np.reshape(pixel_array, (image_height * image_width, dimension))
    pixel_np = pixel_np1.copy()
    new_pixel = np.zeros((image_width * image_height, dimension), dtype=np.uint8)
    centroids = init.generate_centroids_from_data_more_dimensions(pixel_np, num_of_centroids, dimension)
    km = KMeans(centroids)
    centroids_history, error_history = km.fit(new_pixel)
    print(error_history)
    centers = np.asarray(centroids_history[-1], dtype=np.uint8)
    for p_idx, point in enumerate(pixel_np):
        distances = np.zeros(num_of_centroids)
        for n_idx, node in enumerate(centers):
            distances[n_idx] = np.linalg.norm(point - node)
        winner = np.argmin(distances)
        new_pixel[p_idx] = centers[winner]
    pixel_centroids_reshaped = np.reshape(new_pixel, (image_height, image_width, 3))
    compressed_im = Image.fromarray(pixel_centroids_reshaped)
    compressed_im.save('results/task3/' + filename[:-4] + '_' + str(num_of_centroids) + '_centroids.png')
    np.set_printoptions(precision=10)
    print(str(num_of_centroids), ' -> ', str(error_history[-1]))
    return error_history[-1]

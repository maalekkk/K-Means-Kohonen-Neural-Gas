import numpy as np


def generate_circle_points(circle_x, circle_y, circle_r, num_of_points):
    x, y = [], []
    for i in range(num_of_points):
        alpha = 2 * np.math.pi * np.random.random()
        r = circle_r * np.math.sqrt(np.random.random())
        x.append(r * np.math.cos(alpha) + circle_x)
        y.append(r * np.math.sin(alpha) + circle_y)
    return x, y


def generate_square_points(left_top_x, left_top_y, side_length, num_of_points):
    x, y = [], []
    for i in range(num_of_points):
        x.append(left_top_x + np.random.uniform(0, side_length))
        y.append(left_top_y - np.random.uniform(0, side_length))
    return x, y


def generate_rectangle_points(left_top_x, left_top_y, x_side_length, y_side_length, num_of_points):
    x, y = [], []
    for i in range(num_of_points):
        x.append(left_top_x + np.random.uniform(0, x_side_length))
        y.append(left_top_y - np.random.uniform(0, y_side_length))
    return x, y


def generate_data(fig_type, fig_num_of_pts, two_fig):
    dataset = []
    if fig_type == "circle":
        x, y, r = 3, 0, 2
        x1, y1 = generate_circle_points(x, y, r, fig_num_of_pts)
        if two_fig:
            x, y, r = -3, 0, 2
            x2, y2 = generate_circle_points(x, y, r, fig_num_of_pts)
            dataset = np.array(list(zip(x1 + x2, y1 + y2)))
        else:
            dataset = np.array(list(zip(x1, y1)))
    elif fig_type == "square":
        left_top_x, left_top_y, side_length = -6, 2, 4
        x1, y1 = generate_square_points(left_top_x, left_top_y, side_length, fig_num_of_pts)
        if two_fig:
            left_top_x, left_top_y, side_length = 2, 2, 4
            x2, y2 = generate_square_points(left_top_x, left_top_y, side_length, fig_num_of_pts)
            dataset = np.array(list(zip(x1 + x2, y1 + y2)))
        else:
            dataset = np.array(list(zip(x1, y1)))
    elif fig_type == "rectangle":
        left_top_x, left_top_y, x_side_length, y_side_length = -6, 3, 4, 6
        x1, y1 = generate_rectangle_points(left_top_x, left_top_y, x_side_length, y_side_length,
                                           fig_num_of_pts)
        if two_fig:
            left_top_x, left_top_y, x_side_length, y_side_length = 2, 3, 4, 6
            x2, y2 = generate_rectangle_points(left_top_x, left_top_y, x_side_length, y_side_length,
                                               fig_num_of_pts)
            dataset = np.array(list(zip(x1 + x2, y1 + y2)))
        else:
            dataset = np.array(list(zip(x1, y1)))
    return dataset


def generate_centroids_from_data(dataset, num_of_classes):
    dataset_sorted = sorted(dataset, key=lambda x: (x[0], x[1]))
    temp_size = np.math.floor(len(dataset_sorted) / num_of_classes)
    centroids = []
    for i in range(num_of_classes):
        random_number = np.math.floor(np.random.uniform(i * temp_size, (i + 1) * temp_size))
        centroids.append(list())
        centroids[i].append(dataset_sorted[random_number][0])
        centroids[i].append(dataset_sorted[random_number][1])
    return np.array(centroids)


def generate_centroids_from_data_more_dimensions(dataset, num_of_classes, dimension):
    dataset_sorted = sorted(dataset, key=lambda x: (x[0], x[1]))
    temp_size = np.math.floor(len(dataset_sorted) / num_of_classes)
    centroids = []
    for i in range(num_of_classes):
        random_number = np.math.floor(np.random.uniform(i * temp_size, (i + 1) * temp_size))
        centroids.append(list())
        for j in range(dimension):
            centroids[i].append(dataset_sorted[random_number][j])
    return np.array(centroids)


def generate_random_centroids(num_of_classes):
    centroids = []
    for i in range(num_of_classes):
        centroids.append(list())
        centroids[i].append(np.random.uniform(-10, 10))
        centroids[i].append(np.random.uniform(-10, 10))
    return np.array(centroids)

import scipy as sp
import cv2
import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from Feature import Feature


def symmetry_factor(scan_line, index, width, kernel):
    left = scan_line[index - width:index].astype(float)
    right = np.flipud(scan_line[index:index + width]).astype(float)
    diff = np.abs(left - right)
    return diff.dot(kernel)  # +255- np.std(left)+255 - np.std(right)#np.average(diff)


def scan_whole_line(whole_image, height_index=400, width=100):
    scan_line = whole_image[height_index, :]
    X = range(width, len(scan_line) - width)
    kernel = cv2.getGaussianKernel(width * 2, 0)[width:] * 2
    return np.squeeze(np.array([symmetry_factor(scan_line, index, width, kernel) for index in X]))


def unique_rows(a, orderBy):
    order = np.lexsort(orderBy.T)
    a = a[order]
    orderBy = orderBy[order]
    diff = np.diff(orderBy, axis=0)
    ui = np.ones(len(orderBy), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


def create_symmetry_map(image, width, verbose=False):
    print width
    img = np.zeros(image.shape[1] - width * 2)
    for num in xrange(1, image.shape[0]):
        results = scan_whole_line(image, num, width)
        img = np.vstack((img, results))
    img = img.astype(np.float32)
    kernel_size = 11
    if verbose:
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.show()
    img = cv2.GaussianBlur(img, sigmaX=0, ksize=(kernel_size, 1))
    img = cv2.Sobel(img, cv2.CV_32F, 2, 0, ksize=kernel_size)
    padding = np.zeros((img.shape[0], width))

    if verbose:
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')

        plt.show()
    return np.hstack((padding, img, padding))


def process_image(image, w):
    processed_map_horizontal = create_symmetry_map(image.res_grayscale_image, w)

    processed_map_vertical = create_symmetry_map(image.res_grayscale_image.T, w)

    processed_map_horizontal[processed_map_horizontal < 0] = 0
    processed_map_vertical[processed_map_vertical < 0] = 0
    processed_map = processed_map_horizontal * processed_map_vertical.T
    processed_map = processed_map.astype(np.float32)
    processed_map[processed_map < np.max(processed_map) * 0.0] = 0

    neighborhood_size = 5
    threshold = 0

    data = processed_map

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))

    return xy


def process_features(features, width, image):
    assert width in [np.power(2, i) for i in xrange(0, 10)]
    feature_vectors = []

    for feature_y, feature_x in features:
        scan_line = image.res_grayscale_image[feature_y, :]
        index = feature_x

        left = scan_line[index - width:index].astype(float)
        right = np.flipud(scan_line[index:index + width]).astype(float)

        avg_line = (left + right) / 2
        avg_value = np.average(avg_line)

        feature_vector = avg_line < avg_value

        feature_vector = sp.ndimage.zoom(feature_vector, 128 / float(width), order=0).astype(int)

        feature_vectors.append(Feature(feature_vector, feature_x, feature_y, image, image.id_building, width))

    return feature_vectors


class FeatureExtractor:
    def extractFeatures(self, image):  # returns Feature[] for one image

        features_all = []
        i = 7
        while True:
            w = np.power(2, i)
            if image.res_grayscale_image.shape[0] < 2 * w:
                break

            xy = process_image(image, w)
            features = process_features(xy, w, image)

            if len(features) == 0:
                break

            features_all.extend(features)
            i = i + 1;

        print "Feature count: " + str(len(features_all))
        return features_all  # , feature_map

    def extractFeaturesMulti(self, images):  # returns Feature[] for set of images
        feature_list = []

        for image in images:
            features = self.extractFeatures(image)
            feature_list.extend(features);

        return feature_list;

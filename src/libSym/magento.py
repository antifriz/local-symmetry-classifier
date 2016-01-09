import scipy as sp
import sklearn
import sklearn.naive_bayes
import itertools
import cv2
import numpy as np
import random
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from sklearn.neighbors import KNeighborsClassifier


def _preprocess_image(image, max_width=640, max_height=480):
    if type(image) is str:
        image = cv2.imread(image)
    elif type(image) is not np.ndarray:
        raise Exception('Passed obj should be string or numpy array')
            
    factor = 1
    print image.shape[0]
    if image.shape[0] > max_height:
        factor = max_height / float(image.shape[0])
    if image.shape[1] > max_width:
        f2 = max_width / float(image.shape[1])
        factor = f2 if f2 < factor else factor
    # print 'factor',factor
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor,
                       interpolation=cv2.INTER_CUBIC) if factor < 1 else image
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _find_best_k(harris_output, desired_feature_count, propose_k=0.1, tolerance=0.1):
    min_k = 0
    k = propose_k
    max_k = 1
    max_val = harris_output.max()
    while True:
        feature_count = np.sum(harris_output > k * max_val)

        # print min_k,k,max_k,feature_count
        if feature_count > desired_feature_count * (1 + tolerance):
            min_k = k
        elif feature_count < desired_feature_count * (1 - tolerance):
            max_k = k
        else:
            break
        if max_k - min_k < 0.001:
            break
        k = (max_k + min_k) / 2
    return k


def _extract_significant_points_in_image(preprocessed_image, desired_point_count=1000):
    image = cv2.GaussianBlur(preprocessed_image, sigmaX=0, ksize=(3, 3))
    image = np.float32(image)
    harris_data = cv2.cornerHarris(image, 2, 3, 0.04)

    k = _find_best_k(harris_data, desired_point_count)

    indices_all = np.dstack(np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0])))
    return indices_all[harris_data > k * harris_data.max()]


def _process_descriptors(scan_line, index, width):
    left = scan_line[index - width:index].astype(float)
    right = np.flipud(scan_line[index:index + width]).astype(float)
    avg_line = (left + right) / 2
    raw = (avg_line < np.average(avg_line)).astype(float)
    return sp.ndimage.interpolation.zoom(raw, 128 / float(width), order=3, mode='constant')


def _symmetry_score_for_pixel(scan_line, index, width, kernel):
    left = scan_line[index - width:index].astype(float)
    right = np.flipud(scan_line[index:index + width]).astype(float)
    diff = np.abs(left - right)
    return diff.dot(kernel)


def _unique_rows(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


def create_descriptors(image):
    preprocessed_image = _preprocess_image(image)
    indices = _extract_significant_points_in_image(preprocessed_image)
    preprocessed_image = cv2.GaussianBlur(preprocessed_image, sigmaX=0, ksize=(3, 3))
    descriptors = []
    for x, y in indices:
        for w in np.logspace(3, 7, 9, base=2).astype(np.int):
            if preprocessed_image.shape[1] - x < w or x < w:
                continue
            scan_line = preprocessed_image[y, :]
            val = _symmetry_score_for_pixel(scan_line, x, w, cv2.getGaussianKernel(w * 2, 0)[w:] * 2)  # np.ones(w)/w)
            if val < 16:
                descriptors.append(_process_descriptors(scan_line, x, w))
    for x, y in indices:
        cv2.circle(preprocessed_image, (x, y), 10, 255)
    # plt.figure(figsize=(20,20)),plt.imshow(gray, cmap='gray'),plt.show()
    descriptors = list(_unique_rows(np.array(descriptors)))
    random.seed(42)
    random.shuffle(descriptors)
    descriptors = np.array(descriptors[:400])
    return descriptors


class MagentoClassifier(object):
    def __init__(self, n_neighbors=15, weights='distance'):
        self._classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        self._train_image_ids = []

    def fit(self, filenames, labels):
        descriptors_all = []
        self._train_image_ids = []
        for filename, cnt in zip(filenames, labels):
            #print filename
            descriptors = create_descriptors(filename)
            print descriptors.shape
            descriptors_all += list(descriptors)
            self._train_image_ids += list(np.ones(descriptors.shape[0]) * cnt)

        descriptors_all = np.array(descriptors_all)
        self._train_image_ids = np.array(self._train_image_ids)
        self._classifier.fit(descriptors_all, self._train_image_ids)

    def predict(self, filename, method='default'):
        descriptors = create_descriptors(filename)
        if method == 'default':
            return int(sp.stats.mstats.mode(self._classifier.predict(descriptors))[0][0])
        elif method == 'strict':
            pp = self._classifier.predict_proba(descriptors)
            chances = np.zeros(pp.shape[1])
            for p in pp:
                if np.max(p) > 0.5:
                    chances[np.argmax(p)] += 1000.0 / len(self._train_image_ids[self._train_image_ids == np.argmax(p)])
            np.argmax(chances)
        else:
            raise Exception('Unknown predict method')

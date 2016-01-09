import scipy as sp
import sklearn
import sklearn.naive_bayes
import itertools
import cv2
import numpy as np
import random
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.spatial.distance import euclidean
from scipy.ndimage.interpolation import zoom
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, NearestNeighbors


def _preprocess_image(image, max_width=640, max_height=480):
    if type(image) is str:
        image = cv2.imread(image)
    elif type(image) is not np.ndarray:
        raise Exception('Passed obj should be string or numpy array')

    factor = 1
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


#
# def _extract_significant_points_in_image(preprocessed_image, desired_point_count=1000):
#     image = cv2.GaussianBlur(preprocessed_image, sigmaX=0, ksize=(3, 3))
#     image = np.float32(image)
#     harris_data = cv2.cornerHarris(image, 2, 3, 0.04)
#
#     k = _find_best_k(harris_data, desired_point_count)
#
#     indices_all = np.dstack(np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0])))
#     return indices_all[harris_data > k * harris_data.max()]
#

# def _process_descriptors(scan_line, index, width):
#     left = scan_line[index - width:index].astype(float)
#     right = np.flipud(scan_line[index:index + width]).astype(float)
#     avg_line = (left + right) / 2
#     raw = (avg_line < np.average(avg_line)).astype(float)
#     return sp.ndimage.interpolation.zoom(raw, 128 / float(width), order=3, mode='constant')


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


# def create_descriptors(image):
#     preprocessed_image = _preprocess_image(image)
#     indices = _extract_significant_points_in_image(preprocessed_image)
#     preprocessed_image = cv2.GaussianBlur(preprocessed_image, sigmaX=0, ksize=(3, 3))
#     descriptors = []
#     for x, y in indices:
#         for w in np.logspace(3, 7, 9, base=2).astype(np.int):
#             if preprocessed_image.shape[1] - x < w or x < w:
#                 continue
#             scan_line = preprocessed_image[y, :]
#             val = _symmetry_score_for_pixel(scan_line, x, w, cv2.getGaussianKernel(w * 2, 0)[w:] * 2)  # np.ones(w)/w)
#             if val < 16:
#                 descriptors.append(_process_descriptors(scan_line, x, w))
#     for x, y in indices:
#         cv2.circle(preprocessed_image, (x, y), 10, 255)
#     # plt.figure(figsize=(20,20)),plt.imshow(gray, cmap='gray'),plt.show()
#     descriptors = list(_unique_rows(np.array(descriptors)))
#     random.seed(42)
#     random.shuffle(descriptors)
#     descriptors = np.array(descriptors[:400])
#     return descriptors
#
#
# class MagentoClassifier(object):
#     def __init__(self, n_neighbors=10, weights='distance'):
#         self._classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
#         self._train_image_ids = []
#
#     def fit(self, filenames, labels):
#         descriptors_all = []
#         self._train_image_ids = []
#         for filename, cnt in zip(filenames, labels):
#             descriptors = create_descriptors(filename)
#             descriptors_all += list(descriptors)
#             self._train_image_ids += list(np.ones(descriptors.shape[0]) * cnt)
#
#         descriptors_all = np.array(descriptors_all)
#         self._train_image_ids = np.array(self._train_image_ids)
#         self._classifier.fit(descriptors_all, self._train_image_ids)
#
#     def predict(self, filename, method='default'):
#         descriptors = create_descriptors(filename)
#         if method == 'default':
#             return int(sp.stats.mstats.mode(self._classifier.predict(descriptors))[0][0])
#         elif method == 'strict':
#             pp = self._classifier.predict_proba(descriptors)
#             chances = np.zeros(pp.shape[1])
#             for p in pp:
#                 if np.max(p) > 0.5:
#                     chances[np.argmax(p)] += 1000.0 / len(self._train_image_ids[self._train_image_ids == np.argmax(p)])
#             np.argmax(chances)
#         else:
#             raise Exception('Unknown predict method')
#
#
#


def _extract_significant_points_in_image(image, desired_point_count=1000):
    """

    :type image: Image
    """
    processed_image = image.get_processed()
    processed_image = np.float32(processed_image)
    harris_data = cv2.cornerHarris(processed_image, 2, 3, 0.04)

    k = _find_best_k(harris_data, desired_point_count)

    indices_all = np.dstack(np.meshgrid(np.arange(0, processed_image.shape[1]), np.arange(0, processed_image.shape[0])))
    return indices_all[harris_data > k * harris_data.max()]


def _process_descriptors(scan_line, index, width):
    """

    :type scan_line: np.ndarray
    :type index: int
    :type width: int
    """
    left = scan_line[index - width:index].astype(float)
    right = np.flipud(scan_line[index:index + width]).astype(float)
    avg_line = (left + right) / 2
    raw = (avg_line < np.average(avg_line)).astype(float)
    return sp.ndimage.interpolation.zoom(raw, 128 / float(width), order=3, mode='constant')


class MagentoClassifier(object):
    def __init__(self, n_neighbors=10, weights='distance'):
        self._classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        self._buildings = None
        """
        :type _buildings: list[Buildings]|None
        """

    def fit(self, buildings):
        """

        :type buildings: list[Building]
        """

        self._buildings = buildings
        features_all = [feature for building in buildings
                        for image in building.get_images()
                        for feature in
                        image.get_features()]

        descriptors_all = np.array([feature.get_descriptor() for feature in features_all])
        assert len(descriptors_all.shape) == 2

        classes_all = np.array([feature.get_image().get_building().get_identifier() for feature in features_all])
        assert len(classes_all.shape) == 1

        assert descriptors_all.shape[0] == classes_all.shape[0]

        self._classifier.fit(descriptors_all, classes_all)

    # todo
    def predict(self, image):
        """

        :type image: Image
        """

        features_all = image.get_features()

        descriptors_all = np.array([feature.get_descriptor() for feature in features_all])
        assert len(descriptors_all.shape) == 2
        distances, matches = self._classifier.kneighbors(descriptors_all, n_neighbors=1, return_distance=True)
        self.show_match(image, matches, distances)
        # descriptors = create_descriptors(filename)
        # if method == 'default':
        #    return int(sp.stats.mstats.mode(self._classifier.predict(descriptors))[0][0])
        # elif method == 'strict':
        #    pp = self._classifier.predict_proba(descriptors)
        #    chances = np.zeros(pp.shape[1])
        #    for p in pp:
        #        if np.max(p) > 0.5:
        #            chances[np.argmax(p)] += 1000.0 / len(self._train_image_ids[self._train_image_ids == np.argmax(p)])
        #    np.argmax(chances)
        # else:
        #    raise Exception('Unknown predict method')

    def show_match(self, image_test, matches,distances):
        """
        :type image_test: Image
        :type matches: np.array
        :type distances: np.array
        """
        image_train = self._buildings[0].get_images()[0]
        """
        :type image_train: Image
        """
        image_train_rgb = image_train.get_rgb()
        image_test_rgb = image_test.get_rgb()

        offset = image_test_rgb.shape[1]
        size = offset + image_train_rgb.shape[1]

        showoff = np.zeros((Image.DEFAULT_HEIGHT, size, 3), np.uint8)

        print showoff.shape, image_test_rgb.shape
        showoff[0:image_test_rgb.shape[0], 0:image_test_rgb.shape[1], :] = image_test_rgb
        showoff[0:image_train_rgb.shape[0], 0 + offset:image_train_rgb.shape[1] + offset, :] = image_train_rgb

        raw_showoff = showoff.copy()
        aa = showoff.copy()
        for feature, match,distance in zip(image_test.get_features(), matches,distances):
            showoff2 = raw_showoff.copy()
            xy1 = feature.get_xy()
            other_feature = image_train.get_features()[match]
            xy2 = other_feature.get_xy()
            xy2 = xy2 + [offset, 0]  # do not += !!!

            score___ = (1 - (feature._score * other_feature._score) / (15.0 ** 2))
            rating = int(score___ * 128 + 127)
            distance_ = 1 / (distance + 1) * 10
            print distance, distance_,rating,feature._score, other_feature._score
            cv2.line(showoff, tuple(xy1), tuple(xy2), (0, 0, rating), distance_)
            cv2.line(showoff2, tuple(xy1), tuple(xy2), (0, 0, rating), distance_)
            cv2.circle(showoff,  tuple(xy1), feature._w, (0, 0, 255), thickness=1)
            cv2.circle(showoff, tuple(xy2), other_feature._w, (0, 0, 255), thickness=1)
            cv2.circle(showoff2,  tuple(xy1), feature._w, (0, 0, 255), thickness=1)
            cv2.circle(showoff2, tuple(xy2), other_feature._w, (0, 0, 255), thickness=1)


            cv2.imshow("", showoff2)

            def get_stripe(img,f):
                return cv2.resize(img[f._y - 5:f._y + 5+1, f._x - f._w:f._x + f._w + 1], (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)

            first_stripe = get_stripe(image_test.get_processed(),feature)
            second_stripe = get_stripe(image_train.get_processed(),other_feature)
            cv2.imshow("A", first_stripe)
            cv2.imshow("B", second_stripe)
            cv2.imshow("", aa)
            aa = cv2.pyrDown(aa)
            cv2.waitKey()

        cv2.imshow("", showoff)
        cv2.waitKey()


class Feature(object):
    def __init__(self, image, score, x, y, w):
        self._image = image
        self._x = x
        self._y = y
        self._w = w
        self._score = score
        self._descriptor = None
        self._locality = None

    def get_descriptor(self):
        if self._descriptor is None:
            self._calculate_descriptor()
        return self._descriptor

    def _calculate_descriptor(self):
        self._descriptor = _process_descriptors(self.get_image().get_processed()[self._y, :], self._x, self._w)

    def get_image(self):
        """

        :rtype: Image
        """
        return self._image

    def draw_me(self, image):
        """

        :type image: np.ndarray
        """
        cv2.circle(image, (self._x, self._y), self._w, (0, 0, 255), thickness=1)
        # cv2.circle(image, (self._x, self._y), 1, (0, 0, 255), thickness=2)

    def show(self, title=""):
        img = self.get_image().get_rgb().copy()
        self.draw_me(img)
        cv2.imshow(title if title is not "" else self.__repr__(), img)
        cv2.waitKey()

    def __repr__(self):
        return "{xyw=(" + str(self._x) + "," + str(self._y) + "," + str(self._w) + "), score=" + str(
                self._score) + ", locality=" + str(self._locality) + "}"

    def get_xy(self):
        return np.array([self._x, self._y])


class Image(object):
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    WS = np.logspace(3, 7, 9, base=2).astype(np.int)[2:]

    def __init__(self, image_path):
        self._image_path = image_path
        self._image_gray = None
        self._image_rgb = None
        self._image_processed = None
        self._features = None
        self._building = None

    def get_gray(self):
        """

        :rtype: np.ndarray
        """
        if self._image_gray is None:
            self._lazy_load()
        return self._image_gray

    def get_rgb(self):
        """

        :rtype: np.ndarray
        """
        if self._image_rgb is None:
            self._lazy_load()
        return self._image_rgb

    def _lazy_load(self):
        image = self._letterbox_image(cv2.imread(self._image_path))

        if len(image.shape) != 3:
            raise Exception('Image ' + self._image_path + ' is not RGB')

        self._image_rgb = image
        self._image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _letterbox_image(image):
        factor = 1
        if image.shape[0] > Image.DEFAULT_HEIGHT:
            factor = Image.DEFAULT_HEIGHT / float(image.shape[0])
        if image.shape[1] > Image.DEFAULT_WIDTH:
            f2 = Image.DEFAULT_WIDTH / float(image.shape[1])
            factor = f2 if f2 < factor else factor
        image = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC) if factor < 1 else image
        return image

    def get_features(self):
        """

        :rtype: list[Feature]
        """
        if self._features is None:
            self._extract_features()
        return self._features

    def _extract_features(self):
        """
        :rtype: list[Feature]
        """
        indices = _extract_significant_points_in_image(self)

        image_matrix = self.get_processed()
        features = []

        for w in Image.WS:
            features_in_level = []
            xys_in_level = []
            for x, y in indices:
                if image_matrix.shape[1] - x < w or x < w:
                    continue
                scan_line = image_matrix[y, :]
                score = _symmetry_score_for_pixel(scan_line, x, w,
                                                  cv2.getGaussianKernel(w * 2, 0)[w:] * 2)  # np.ones(w)/w)
                if score < 16:
                    feature = Feature(self, score, x, y, w)
                    features.append(feature)
                    features_in_level.append(feature)
                    xys_in_level.append((x, y))
            xys_in_level = np.array(xys_in_level)
            for feature, xy in zip(features_in_level, xys_in_level):
                min_distance = Image.DEFAULT_WIDTH + Image.DEFAULT_HEIGHT
                for other_feature, other_xy in zip(features_in_level, xys_in_level):
                    if other_feature._score > feature._score:
                        distance = euclidean(xy, other_xy)
                        if distance < min_distance:
                            min_distance = distance
                feature._locality = min_distance

        self._features = [feature for feature in features if feature._locality > 10]
        # for feature in self._features:
        #    print feature

    def set_building(self, building):
        """

        :type building: Building
        """
        self._building = building

    def get_building(self):
        """

        :rtype: Building
        """
        return self._building

    def get_processed(self):
        """

        :rtype: np.ndarray
        """
        if self._image_processed is None:
            self._image_processed = cv2.GaussianBlur(self.get_gray(), sigmaX=0, ksize=(3, 3))
        return self._image_processed

    def show(self, title=""):
        cv2.imshow(title if title is not "" else self._image_path, self.get_rgb())
        cv2.waitKey()

    def show_features(self, title=""):
        img = self.get_rgb().copy()
        [feature.draw_me(img) for feature in self.get_features()]
        cv2.imshow(title if title is not "" else self.__repr__(), img)
        cv2.waitKey()


class Building(object):
    def __init__(self, identifier, images):
        """

        :type identifier: int
        :type images: list[Image]
        """
        self._images = images
        for image in images:
            image.set_building(self)

        self._identifier = identifier
        self._index = None

    def get_images(self):
        """

        :rtype: list[Image]
        """
        return self._images

    def set_identifier(self, identifier):
        """

        :type identifier: int
        """
        self._identifier = identifier

    def get_identifier(self):
        """

        :rtype: int
        """
        return self._identifier

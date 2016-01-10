import scipy as sp
import sklearn
import sklearn.naive_bayes
import itertools
import cv2
import numpy as np
import random
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.spatial.distance import euclidean, cdist
from scipy.ndimage.interpolation import zoom
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, NearestNeighbors

from matplotlib import pyplot as plt

import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology



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


def _symmetry_score_for_pixel(pyramid_level, y, x, w, kernel):
    """

    :type pyramid_level: PyramidLevel
    """
    scan_line = pyramid_level.get_data()[y, :]
    left = scan_line[x - w:x].astype(float)
    right = np.flipud(scan_line[x:x + w]).astype(float)
    diff = np.abs(left - right)
    return diff.dot(kernel)


def _unique_rows(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


def detect_local_minima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr == 0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min - eroded_background
    return np.where(detected_minima)


def _get_left(scan_line, x, w):
    return np.flipud(scan_line[:,x - w:x]).astype(float)


def _get_right(scan_line, x, w):
    return scan_line[:,x:x + w].astype(float)


def _process_descriptors(pyramid_level, feature):
    """

    :type pyramid_level: PyramidLevel
    :type feature:Feature
    :rtype: np.array
    """
    x = feature._x
    y = feature._y
    w = Feature.WIDTH
    h = Feature.HEIGHT

    angles,weights_all,_ = pyramid_level.get_data()
    scan_lines = angles[y - h: y + h + 1, :]
    weights = weights_all[y - h: y + h + 1, :]
    left,right = _get_left(scan_lines,x,w),_get_right(scan_lines,x,w)
    left_w,right_w= _get_left(weights,x,w),_get_right(weights,x,w)

    #kernel = cv2.getGaussianKernel(w * 2, 0)[:w] * 2
    #c1, _ = np.meshgrid(kernel, np.zeros(h * 2 + 1))

    sum = left - right
    sum = np.minimum(sum,360-sum)
    sum_w = left_w - right_w
    sum_w = np.minimum(sum_w,360-sum_w)

    avg_line = (sum) / 2
    avg_line_w = (sum_w) / 2


    np.histogram(avg_line,bins=8,weights=sum_w)


    raw = (avg_line < np.average(avg_line)).astype(float)
    # raw *= c1
    ravel = np.ravel(raw)
    #assert ravel.shape[0] == (h * 2 + 1) * w, "" + str(ravel.shape) + " " + str((h * 2 + 1) * w)
    # np.append(ravel,feature.get_global_xy_w()[0])
    return ravel
    # return sp.ndimage.interpolation.zoom(raw, 128 / float(w), order=3, mode='constant')


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
        features_all = [feature for building in buildings for feature in building.get_all_features()]

        descriptors_all = np.array([feature.get_descriptor() for feature in features_all])
        return
        assert len(descriptors_all.shape) == 2

        classes_all = np.array(
                [feature.get_pyramid_level().get_image().get_building().get_identifier() for feature in features_all])
        assert len(classes_all.shape) == 1

        assert descriptors_all.shape[0] == classes_all.shape[0]

        self._classifier.fit(descriptors_all, classes_all)

    # todo
    def predict(self, image):
        """

        :type image: Image
        """
        N_NEIGHBORS = 10

        features_all = image.get_all_features()

        descriptors_all = np.array([feature.get_descriptor() for feature in features_all])
        assert len(descriptors_all.shape) == 2
        distances, matches = self._classifier.kneighbors(descriptors_all, n_neighbors=N_NEIGHBORS, return_distance=True)
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

    def show_match(self, image_test, matches, distances):
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

        showoff[0:image_test_rgb.shape[0], 0:image_test_rgb.shape[1], :] = image_test_rgb
        showoff[0:image_train_rgb.shape[0], 0 + offset:image_train_rgb.shape[1] + offset, :] = image_train_rgb

        only_circles = showoff.copy()
        raw_showoff = showoff.copy()
        for feature in image_test.get_all_features():
            xy1, w1 = feature.get_global_xy_w()
            cv2.circle(only_circles, tuple(xy1), w1, (0, 0, 255), thickness=1)
        for feature in image_train.get_all_features():
            xy2, w2 = feature.get_global_xy_w()
            xy2 = xy2 + [offset, 0]  # do not += !!!
            cv2.circle(only_circles, tuple(xy2), w2, (0, 0, 255), thickness=1)
        cv2.imshow("ftrs", only_circles)
        cv2.waitKey()

        matching_score = 0
        for feature, matchs, distancess in zip(image_test.get_all_features(), matches, distances):
            xy1, w1 = feature.get_global_xy_w()

            best_other_feature = image_train.get_all_features()[matchs[0]]
            xy2, w2 = best_other_feature.get_global_xy_w()
            matching_score += (xy1 - xy2) ** 2  # + (w1-w2)**2
        print "", np.sqrt(np.average(matching_score)), " +- ", np.sqrt(np.std(matching_score))

        FACTOR = 3
        for feature, matchs, distancess in zip(image_test.get_all_features(), matches, distances):
            showoff2 = raw_showoff.copy()
            xy1, w1 = feature.get_global_xy_w()

            # if all(distances>FACTOR):
            #     continue

            cv2.circle(showoff, tuple(xy1), w1, (0, 0, 255), thickness=1)
            cv2.circle(showoff2, tuple(xy1), w1, (0, 0, 255), thickness=1)

            feature.show("test")
            for idx, (match, distance) in enumerate(zip(matchs, distancess)):
                # if distance>FACTOR:
                #     continue
                other_feature = image_train.get_all_features()[match]
                xy2_, w2 = other_feature.get_global_xy_w()
                xy2 = xy2_ + [offset, 0]  # do not += !!!
                other_feature.show("train")
                print distance

                score___ = (1 - (feature._score * other_feature._score) / (15.0 ** 2))
                rating = int(score___ * 128 + 127)
                distance_ = int(1 / (distance + 1) * 10)
                # print distance, distance_, rating, feature._score, other_feature._score
                cv2.line(showoff, tuple(xy1), tuple(xy2), (0, 0, rating), distance_)
                cv2.line(showoff2, tuple(xy1), tuple(xy2), (0, 0, rating), distance_)
                cv2.circle(showoff, tuple(xy2), w2, (0, 0, 255), thickness=1)
                cv2.circle(showoff2, tuple(xy2), w2, (0, 0, 255), thickness=1)

                # if (xy2_-xy1).dot(xy2_-xy1) < 1000:
                cv2.imshow("", showoff2)
                k = cv2.waitKey(0)
                if k == 27:  # wait for ESC key to exit
                    cv2.destroyAllWindows()
                    exit(0)
                    # def get_stripe(img, f):
                    #     return cv2.resize(img[f._y - 5:f._y + 5 + 1, f._x - f._w:f._x + f._w + 1], (0, 0), fx=10, fy=10,
                    #                       interpolation=cv2.INTER_NEAREST)
                    #
                    # first_stripe = get_stripe(image_test.get_processed(), feature)
                    # second_stripe = get_stripe(image_train.get_processed(), other_feature)
                    # cv2.imshow("A", first_stripe)
                    # cv2.imshow("B", second_stripe)

        cv2.imshow("", showoff)
        cv2.imwrite("../data/outputs/jej/_" + str(random.random()) + "cool.jpg", showoff,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.waitKey()


class Feature(object):
    WIDTH = 8 # actual width -> 2*WIDTH
    HEIGHT = 4  # actual height -> 2*HEIGHT + 1

    def __init__(self, pyramid_level, score, x, y):
        self._pyramid_level = pyramid_level
        self._x = x
        self._y = y
        self._score = score
        self._descriptor = None
        self._locality = None
        self._calculate_descriptor()

    def get_descriptor(self):
        """

        :rtype: np.array
        """
        if self._descriptor is None:
            self._calculate_descriptor()
        return self._descriptor

    def _calculate_descriptor(self):
        self._descriptor = _process_descriptors(self.get_pyramid_level(), self)

    def get_pyramid_level(self):
        """

        :rtype: PyramidLevel
        """
        return self._pyramid_level

    def draw_me(self, image):
        """

        :type image: np.ndarray
        """
        cv2.circle(image, (self._x, self._y), self._w, (0, 0, 255), thickness=1)
        # cv2.circle(image, (self._x, self._y), 1, (0, 0, 255), thickness=2)

    def show(self, title=""):
        # img = self.get_pyramid_level().get_data().copy()
        # self.draw_me(img)
        # cv2.imshow(title if title is not "" else self.__repr__(), img)
        # cv2.waitKey()
        matrix = self.get_pyramid_level().get_data()
        neigh = 4
        scan_lines = matrix[self._y - neigh: self._y + neigh + 1, :]

        area = scan_lines[:, self._x - self._w:self._x + self._w]

        left = area[:, :area.shape[1] / 2]
        right = area[:, area.shape[1] / 2:]

        right = np.fliplr(right)

        aaaaaa = (left / 2 + right / 2)

        # cv2.imshow("sl",cv2.resize(scan_lines[:,x-w:x+w],dsize=(0,0),fx=10,fy=10,interpolation=cv2.INTER_NEAREST))
        # k = cv2.waitKey(0)
        # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()
        #     exit(0)


        avg = np.average(area)
        # print avg
        # avg = np.tile(avg, (area.shape[1], 1)).T
        # print avg



        area2 = (np.bitwise_not(area > avg)).astype(np.uint8) * 255

        resize = lambda img: cv2.resize(img, dsize=(0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)

        descriptor = self.get_descriptor()
        desc = np.reshape(descriptor, (area.shape[0], len(descriptor) / area.shape[0]))
        desc = (desc - desc.min()) / (desc.min() - desc.max()) * 255
        desc = desc.astype(np.uint8)

        area = cv2.cvtColor(area, cv2.COLOR_GRAY2BGR)
        area2 = cv2.cvtColor(area2, cv2.COLOR_GRAY2BGR)
        desc = cv2.cvtColor(desc.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        padding = np.zeros((area.shape[0], 5, 3))
        padding[:] = (0, 0, 255)

        large_image = np.hstack((padding, area2, padding, desc))

        large_image[:area.shape[0], :area.shape[1], :] = area

        cv2.imshow(title, resize(area))
        cv2.imshow(title + "-aaaaa", resize(aaaaaa))
        cv2.imshow(title + "-binary", resize(large_image))
        # cv2.imshow(title +"-descriptor", resize(desc))
        # k = cv2.waitKey(0)
        # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()
        #     exit(0)

    def __repr__(self):
        return "{xyw=(" + str(self._x) + "," + str(self._y) + "," + str(self._w) + "), score=" + str(
                self._score) + ", locality=" + str(self._locality) + "}"

    def get_global_xy_w(self):
        scale = self.get_pyramid_level().get_scale()
        return np.array([int(round(self._x / scale)), int(round(self._y / scale))]), int(round(self._w / scale))


class PyramidLevel(object):
    def __init__(self, image, matrix, scale):
        """

        :type image: Image
        :type matrix: np.ndarray
        :type scale: float
        """
        self._image = image
        self._grayscale = matrix
        self._data = None
        self._allowed_area = None
        self._features = None
        self._scale = scale

    def get_parent_image(self):
        return self._image

    def get_data(self):
        if self._data is None:
            self._generate_data()
        return self._data

    def get_allowed_area(self):
        if self._allowed_area is None:
            self._generate_data()
        return self._allowed_area

    def _generate_data(self):
        NORM_LIMIT = 250
        SURPRESSING_FACTOR = 0.5

        print "Hello"
        img = self._grayscale

        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        sobel = 1j * sobel_x + sobel_y  # so the angles are relative to the y-axis

        sobel_norms = np.sqrt((sobel * sobel.conjugate()).real)

        sobel_norms *= float(SURPRESSING_FACTOR) / NORM_LIMIT
        sobel_norms[sobel_norms > 1] = 1

        sobel_angles = np.angle(sobel, deg=True).real


        allowed_area = sobel_norms == 1

        ones = np.ones((Feature.HEIGHT * 2 + 1, Feature.WIDTH - 1))
        zeros = ones * 0
        left_test = np.hstack((ones, zeros))
        right_test = np.hstack((zeros, ones))

        allowed_area = morphology.binary_dilation(allowed_area, structure=left_test) & morphology.binary_dilation(
                allowed_area, structure=right_test)

        self._data = (sobel_angles, sobel_norms, allowed_area)
        # plt.subplot(231), plt.imshow(self._grayscale, cmap='gray')
        # plt.subplot(232), plt.imshow(self._allowed_area, cmap='gray')
        # plt.subplot(233), plt.imshow(sobel_norms, cmap='gray')
        # plt.subplot(234), plt.imshow(sobel_angles, cmap='gray')
        # plt.subplot(235), plt.imshow(sobel_x, cmap='gray')
        # plt.subplot(236), plt.imshow(sobel_y, cmap='gray'), plt.show()
        # #
        # print np.min(sobel_x), np.max(sobel_x), np.min(sobel_y), np.max(sobel_y)
        #
        # self._matrix
        # exit(0)

    def get_scale(self):
        return self._scale

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
        w = Feature.WIDTH
        h = Feature.HEIGHT

        DISALLOWED_AREA_CONSTANT = 10e10
        DIFFERENCE_THRESHOLD = 1

        w = Feature.WIDTH
        h = Feature.HEIGHT

        angles, norms, allowed_area = self.get_data()

        heatmap = np.ones(angles.shape) * DISALLOWED_AREA_CONSTANT

        # cv2.imshow("nja", img)
        for y in range(h,angles.shape[0]-h):
            scan_line = angles[y-h:y+h+1, :]
            weight_line = norms[y-h:y+h+1, :]
            for x in range(w, angles.shape[1] - w):
                if allowed_area[y, x]:
                    trace_scan_line = _get_left(scan_line, x, w) + _get_right(scan_line, x, w)
                    trace_scan_line = np.minimum(trace_scan_line, 360 - trace_scan_line)
                    trace_weight_line = _get_left(weight_line, x, w) + _get_right(weight_line, x, w)
                    weight_line_sum = np.sum(trace_weight_line)

                    trace_weight_line /= weight_line_sum
                    trace_scan_line *= trace_weight_line
                    trace_scan_line = trace_scan_line.flatten()
                    dot = np.sqrt(trace_scan_line.dot(trace_scan_line)) / weight_line_sum
                    heatmap[y, x] = dot
                    # if dot <= DIFFERENCE_THRESHOLD:
                    #     img = cv2.cvtColor(pyramid_level._grayscale.copy(),cv2.COLOR_GRAY2BGR)
                    #     cv2.rectangle(img,(x-w,y-h),(x+w,y+h+1),color=(0,0,255))
                    #     cv2.imshow("jej",img)
                    #     # cv2.imshow("jej",pyramid_level._grayscale[y-h:y+h,x-w:x+w])
                    #     cv2.waitKey()
                    #    #plt.subplot(122),plt.imshow(scan_line[:,x-w:x+w],'gray'),plt.show()
                    #     print "score", dot, x, y
                    #     print np.round(_get_left(scan_line, x, w).astype(np.int))
                    #     print np.round(_get_left(weight_line, x, w),2)
                    #     print np.round(-_get_right(scan_line, x, w).astype(np.int))
                    #     print np.round(_get_right(weight_line, x, w),2)

        filtered_heatmap = heatmap[heatmap != DISALLOWED_AREA_CONSTANT]
        print np.max(filtered_heatmap), np.min(filtered_heatmap), np.average(filtered_heatmap), np.std(
            filtered_heatmap), heatmap.shape
        # heatmap_to_showoff = heatmap.copy()
        # heatmap_to_showoff[heatmap==DISALLOWED_AREA_CONSTANT] =np.average(heatmap_to_showoff[heatmap_to_showoff!=DISALLOWED_AREA_CONSTANT])
        # plt.imshow(heatmap_to_showoff, cmap='gray'), plt.show()

        minimums = np.array(detect_local_minima(heatmap)).T
        #img = cv2.cvtColor(pyramid_level._grayscale, cv2.COLOR_GRAY2BGR)

        min_vals = heatmap[minimums[:, 0], minimums[:, 1]]
        argsort_indices = np.argsort(min_vals)
        minimums = minimums[argsort_indices]
        min_vals = min_vals[argsort_indices]
        minimums = minimums[min_vals < DIFFERENCE_THRESHOLD]
        min_vals= min_vals[min_vals < DIFFERENCE_THRESHOLD]

        #img[minimums[:, 0], minimums[:, 1]] = (0, 0, 255)

        #b, g, r = cv2.split(img)
        #img = cv2.merge([r, g, b])
        #plt.subplot(121), plt.imshow(img), plt.subplot(122), plt.imshow(allowed_area, 'gray'), plt.show()
        #print minimums.shape

        features_in_level = []
        for x, y in zip(minimums,min_vals):
            feature = Feature(self, min_vals, x, y)
            features_in_level.append(feature)
        if len(minimums) == 0:
            self._features = []
            return
        self._features= features_in_level
        # distance_matrix = cdist(minimums, minimums, 'euclidean')
        #
        # for i, (feature, xy) in enumerate(zip(features_in_level, minimums)):
        #     min_distance = Image.DEFAULT_WIDTH + Image.DEFAULT_HEIGHT
        #     for j, (other_feature, other_xy) in enumerate(zip(features_in_level, minimums)):
        #         if other_feature._score > feature._score:
        #             distance = distance_matrix[i, j]
        #             if distance < min_distance:
        #                 min_distance = distance
        #     feature._locality = min_distance
        # self._features = [feature for feature in features if feature._locality > 5]
        # print len(features), len(self._features)

    def show_features(self, title=""):
        img = self.get_data().copy()
        [feature.draw_me(img) for feature in self.get_features()]
        cv2.imshow(title if title is not "" else self.__repr__(), img)
        cv2.waitKey()

    def get_image(self):
        """

        :rtype: Image
        """
        return self._image


class Image(object):
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    PYRAMID_RATIO = np.sqrt(2)
    PYRAMID_LIMIT = 10

    def __init__(self, image_path):
        self._image_path = image_path
        self._image_gray = None
        self._image_rgb = None
        self._image_processed = None
        self._building = None
        self._pyramid = None

    def get_gray(self):
        """

        :rtype: np.ndarray
        """
        if self._image_gray is None:
            self._lazy_load()
        return self._image_gray

    def get_pyramid(self):
        """

        :rtype: PyramidLevel
        """
        if self._pyramid is None:
            self._init_pyramid()
        return self._pyramid

    def _init_pyramid(self):
        self._pyramid = []
        current = self.get_processed()
        scale = 1.0
        while current.shape[0] > Image.PYRAMID_LIMIT and current.shape[1] > Image.PYRAMID_LIMIT:
            self._pyramid.append(PyramidLevel(self, current, scale))
            current = Image.pyr_down(current)
            scale /= 2

    @staticmethod
    def pyr_down(image):
        """

        :param image: np.ndarray
        :rtype: np.ndarray
        """
        return cv2.pyrDown(image)  # , dstsize=(
        # int(image.shape[0] / Image.PYRAMID_RATIO), int(image.shape[1] / Image.PYRAMID_RATIO)))

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

    def get_all_features(self):
        """

        :rtype: list[Feature]
        """
        return [feature for pyramid_level in self.get_pyramid() for feature in pyramid_level.get_features()]


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

    def get_all_features(self):
        """

        :rtype: list[Feature]
        """
        return [feature for image in self.get_images() for feature in image.get_all_features()]

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
    scan_line = pyramid_level.get_matrix()[y, :]
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


def _extract_significant_points_in_image(pyramid_level):
    """

    :type pyramid_level: PyramidLevel
    """
    processed_image = pyramid_level.get_matrix()
    desired_point_count = int(0.04 * processed_image.shape[0] * processed_image.shape[1])
    processed_image = np.float32(processed_image)
    harris_data = cv2.cornerHarris(processed_image, 2, 3, 0.04)

    k = _find_best_k(harris_data, desired_point_count)

    indices_all = np.dstack(np.meshgrid(np.arange(0, processed_image.shape[1]), np.arange(0, processed_image.shape[0])))
    return indices_all[harris_data > k * harris_data.max()]


def _process_descriptors(pyramid_level, feature):
    """

    :type pyramid_level: PyramidLevel
    :type feature:Feature
    :rtype: np.array
    """
    x = feature._x
    y= feature._y
    w = feature._w
    neigh = 4
    matrix = pyramid_level.get_matrix()
    scan_lines = matrix[y - neigh: y + neigh + 1, :]
    left = scan_lines[:, x - w:x].astype(float)

    # cv2.imshow("sl",cv2.resize(scan_lines[:,x-w:x+w],dsize=(0,0),fx=10,fy=10,interpolation=cv2.INTER_NEAREST))
    # k = cv2.waitKey(0)
    # if k == 27:         # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    #     exit(0)

    kernel = cv2.getGaussianKernel(w * 2, 0)[:w] * 2
    c1,_ =  np.meshgrid(kernel,np.zeros(neigh*2+1))


    right = np.fliplr(scan_lines[:, x:x + w]).astype(float)
    avg_line = (left + right) / 2
    raw = (avg_line < np.average(avg_line)).astype(float)
    #raw *= c1
    ravel = np.ravel(raw)
    assert ravel.shape[0] == (neigh*2+1)*w, ""+str(ravel.shape)+" "+str((neigh*2+1)*w)
    #np.append(ravel,feature.get_global_xy_w()[0])
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

        features_all = image.get_all_features()

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
        cv2.imshow("ftrs",only_circles)
        #cv2.waitKey()

        matching_score = 0
        for feature, matchs, distancess in zip(image_test.get_all_features(), matches, distances):
            xy1, w1 = feature.get_global_xy_w()

            best_other_feature = image_train.get_all_features()[matchs[0]]
            xy2, w2 = best_other_feature.get_global_xy_w()
            matching_score +=(xy1-xy2)**2# + (w1-w2)**2
        print "",np.sqrt(np.average(matching_score))," +- ", np.sqrt(np.std(matching_score))

        FACTOR = 3
        for feature, matchs, distancess in zip(image_test.get_all_features(), matches, distances):
            showoff2 = raw_showoff.copy()
            xy1, w1 = feature.get_global_xy_w()

            if all(distances>FACTOR):
                continue

            cv2.circle(showoff, tuple(xy1), w1, (0, 0, 255), thickness=1)
            cv2.circle(showoff2, tuple(xy1), w1, (0, 0, 255), thickness=1)

            feature.show("test")
            for idx, (match,distance) in enumerate(zip(matchs,distancess)):
                if distance>FACTOR:
                    continue
                other_feature = image_train.get_all_features()[match]
                xy2_, w2 = other_feature.get_global_xy_w()
                xy2 = xy2_ + [offset, 0]  # do not += !!!
                other_feature.show("train"+str(idx))
                print distance

                score___ = (1 - (feature._score * other_feature._score) / (15.0 ** 2))
                rating = int(score___ * 128 + 127)
                distance_ = int(1 / (distance + 1) * 10)
                #print distance, distance_, rating, feature._score, other_feature._score
                cv2.line(showoff, tuple(xy1), tuple(xy2), (0, 0, rating), distance_)
                cv2.line(showoff2, tuple(xy1), tuple(xy2), (0, 0, rating), distance_)
                cv2.circle(showoff, tuple(xy2), w2, (0, 0, 255), thickness=1)
                cv2.circle(showoff2, tuple(xy2), w2, (0, 0, 255), thickness=1)

                #if (xy2_-xy1).dot(xy2_-xy1) < 1000:
            cv2.imshow("", showoff2)
            k = cv2.waitKey(1)
            if k == 27:         # wait for ESC key to exit
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
        cv2.imwrite("../data/outputs/jej/_"+str(random.random())+"cool.jpg",showoff,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #cv2.waitKey()


class Feature(object):
    def __init__(self, pyramid_level, score, x, y, w):
        self._pyramid_level = pyramid_level
        self._x = x
        self._y = y
        self._w = w
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
        # img = self.get_pyramid_level().get_matrix().copy()
        # self.draw_me(img)
        # cv2.imshow(title if title is not "" else self.__repr__(), img)
        # cv2.waitKey()
        matrix = self.get_pyramid_level().get_matrix()
        neigh = 4
        scan_lines = matrix[self._y - neigh: self._y + neigh + 1, :]

        area = scan_lines[:, self._x - self._w:self._x + self._w]
        avg = np.average(area)
        #print avg
        #avg = np.tile(avg, (area.shape[1], 1)).T
        #print avg

        area2 = (np.bitwise_not(area>avg)).astype(np.uint8)*255

        resize = lambda img: cv2.resize(img, dsize=(0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)


        descriptor = self.get_descriptor()
        desc = np.reshape(descriptor,(area.shape[0],len(descriptor)/area.shape[0] ))
        desc = (desc - desc.min()) /(desc.min() - desc.max())*255
        desc = desc.astype(np.uint8)


        area = cv2.cvtColor(area,cv2.COLOR_GRAY2BGR)
        area2 = cv2.cvtColor(area2,cv2.COLOR_GRAY2BGR)
        desc = cv2.cvtColor(desc.astype(np.uint8)*255,cv2.COLOR_GRAY2BGR)
        padding = np.zeros((area.shape[0],5,3))
        padding[:] = (0,0,255)



        large_image = np.hstack((area2,padding,desc))

        large_image[:area.shape[0],:area.shape[1],:] = area

        cv2.imshow(title, resize(area))
        cv2.imshow(title+"-binary", resize(large_image))
        #cv2.imshow(title +"-descriptor", resize(desc))
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
        self._matrix = matrix
        self._features = None
        self._scale = scale

    def get_parent_image(self):
        return self._image

    def get_matrix(self):
        return self._matrix

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
        w = 16
        kernel = cv2.getGaussianKernel(w * 2, 0)[w:] * 2

        indices = _extract_significant_points_in_image(self)

        image_matrix = self.get_matrix()
        features = []

        features_in_level = []
        xys_in_level = []
        for x, y in indices:
            if image_matrix.shape[1] - x < w or x < w:
                continue
            if image_matrix.shape[0] <=y+4 or y<4: #todo
                continue

            scores = [_symmetry_score_for_pixel(self, yy, x, w, kernel) for yy in range(y-4,y+4+1)]  # np.ones(w)/w)
            #print np.array(scores)
            score = np.average(scores)
            #print score
            # score += _symmetry_score_for_pixel(self,y+1, x, w, kernel)# np.ones(w)/w)
            # score += _symmetry_score_for_pixel(self,y-1, x, w, kernel)# np.ones(w)/w)

            if score < 16:  # 32*3:
                feature = Feature(self, score, x, y, w)
                features.append(feature)
                features_in_level.append(feature)
                xys_in_level.append((x, y))
                # none = self.get_matrix()[y - 5:y + 5 + 1, x - w: x + w + 1]
                # cv2.imshow("ads", cv2.resize(none, (0, 0), fx=10, fy=10,
                #                             interpolation=cv2.INTER_NEAREST))
                # cv2.waitKey()
        if len(xys_in_level) == 0:
            self._features = []
            return
        distance_matrix = cdist(xys_in_level, xys_in_level, 'euclidean')

        # scores = np.array(map(lambda x: x._score,features))


        for i, (feature, xy) in enumerate(zip(features_in_level, xys_in_level)):
            min_distance = Image.DEFAULT_WIDTH + Image.DEFAULT_HEIGHT
            for j, (other_feature, other_xy) in enumerate(zip(features_in_level, xys_in_level)):
                if other_feature._score > feature._score:
                    distance = distance_matrix[i, j]
                    if distance < min_distance:
                        min_distance = distance
            feature._locality = min_distance
        self._features =[feature for feature in features if feature._locality > 2]
        print len(features), len(self._features)

    def show_features(self, title=""):
        img = self.get_matrix().copy()
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

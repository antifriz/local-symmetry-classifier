import os
from multiprocessing import Pool
from scipy.spatial.distance import euclidean, cdist

import gc
import scipy as sp
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from sklearn.neighbors import KNeighborsClassifier
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

from os import walk
from os.path import join, basename
from scipy import ndimage
LOG_LEVEL = 4
CPU_COUNT = 8
SHOW_DETECTIONS = False


def detect_local_minima(arr):
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = morphology.binary_erosion(
            background, structure=neighborhood, border_value=1)
    detected_minima = local_min - eroded_background
    return np.where(detected_minima)


def _get_left(scan_line, x, w):
    return np.flipud(scan_line[:, x - w:x]).astype(float)


def _get_right(scan_line, x, w):
    return scan_line[:, x:x + w].astype(float)


def _print_sth(label, strng):
    print "[" + label + "]" + (' ' * (8 - len(label))) + "| " + str(strng)


def print_err(str):
    if LOG_LEVEL >= 1:
        _print_sth("ERR", str)


def print_warn(str):
    if LOG_LEVEL >= 2:
        _print_sth("WARNING", str)


def print_info(str):
    if LOG_LEVEL >= 3:
        _print_sth("INFO", str)


def print_verbose(str):
    if LOG_LEVEL >= 4:
        _print_sth("VERBOSE", str)


def print_result(str):
    if LOG_LEVEL >= 0:
        _print_sth("RESULT", str)

METHODS = ['mode', 'mode50', 'sum', 'mode2times']

def predict(data):
    try:
        image, clf, progress, method = data  #type: Image,MagentoClassifier,float,str
        progress_str = str(int(progress * 100))
        perc_str = "[" + str(" " * (3 - len(progress_str)) + progress_str) + "%] | "
        prefix = " " * len(perc_str)
        print_info(perc_str + "Predicting for image " + str(image))


        features_all = image.get_all_features() #type: list[Feature]



        descriptors_all = np.array([feature.get_descriptor() for feature in features_all])
        assert len(descriptors_all.shape) == 2
        if SHOW_DETECTIONS:
            clf.show_match(image, descriptors_all)
        # descriptors = create_descriptors(filename)
        predicted_proba = clf._classifier.predict_proba(descriptors_all)
        argmax_proba = np.argmax(predicted_proba,axis=1)
        max_proba = np.max(predicted_proba,axis=1)
        argsorted_proba = np.argsort(predicted_proba,axis=1)
        secondmax_proba = np.sort(predicted_proba,axis=1)[:,-2]
        calc_mode = lambda x: int(clf._classes[int(sp.stats.mstats.mode(x)[0][0])])


        mode_method = calc_mode(argmax_proba)

        sum_method = int(clf._buildings[np.argmax(np.sum(predicted_proba,axis=0))].get_identifier())

        trimmed = argmax_proba[max_proba >= 0.5]
        if trimmed.shape[0]>0:
            mode_50_method = calc_mode(trimmed)
        else:
            mode_50_method = mode_method
            print_info(prefix+" fallback to mode method")

        mode_2_times = calc_mode(argmax_proba[max_proba>=2*secondmax_proba])


        if method == 'mode':
            i = mode_method
        elif method == 'mode50':
            i = mode_50_method
        elif method == 'sum':
            i = sum_method
        elif method == 'mode2times':
            i = mode_2_times
        else:
            raise Exception()
        res = np.array([mode_method,mode_50_method,sum_method,mode_2_times])
        name = lambda x:filter(lambda b: b.get_identifier()==x,clf._buildings)[0].get_name()
        real_i = image.get_building().get_identifier()
        if i != real_i:
            hit_by = list(np.array(METHODS)[np.where(res == real_i)])
            print_result(perc_str + "MISS "+name(i)+" =/= "+str(image)+" " + str( "hit by " + ', '.join(hit_by) if len(hit_by) > 0 else ""))

            print_info(perc_str + name(mode_method) + " <- mode method") if i !=mode_method else None
            print_info(perc_str + name(sum_method) + " <- sum method") if i !=sum_method else None
            print_info(perc_str + name(mode_50_method) + " <- mode 50 method") if i !=mode_50_method else None
            print_info(perc_str + name(mode_2_times) + " <- mode 2 times method") if i !=mode_2_times else None
            #clf.show_match(image, descriptors_all)
        else:
            print_result(prefix + "HIT")
        # print_result(("HIT  " if i == image.get_building().get_identifier() else "MISS ") + " " + str(image))
        return res
    except Exception as e:
        raise e
    # clf.append(result)
    # elif method == 'strict':
    #    pp = self._classifier.predict_proba(descriptors)
    #    chances = np.zeros(pp.shape[1])
    #    for p in pp:
    #        if np.max(p) > 0.5:
    #            chances[np.argmax(p)] += 1000.0 / len(self._train_image_ids[self._train_image_ids == np.argmax(p)])
    #    np.argmax(chances)
    # else:
    #    raise Exception('Unknown predict method')
    real = image.get_building().get_identifier()
    # print_info("["+str(round(float(idx)/len(images),2))+"%] "+("HIT  " if real == result else "MISS ")+"| Image " + str(image) + " predicted: " + str(result) + ", real: " + str(
    #         real))


class MagentoClassifier(object):
    N_NEIGHBORS = 10
    KNN_WEIGHTS = 'distance'

    @staticmethod
    def test_on_dataset(buildings, test_images_per_building=1, train_images_per_building=-1, class_count=-1,
                        n_neighbors=N_NEIGHBORS,
                        weights=KNN_WEIGHTS,method='mode', iterations=1, seed=-1):
        """
        :type test_images_per_building: int
        :type train_images_per_building: int
        :type weights: str
        :type n_neighbors: int
        :type buildings: list[Building]
        :rtype: float
        """
        if method not in METHODS:
            raise Exception('Pick valid method from '+str(METHODS))
        method_idx = METHODS.index(method)
        ult_score = np.zeros(len(METHODS))
        for iter in range(1, iterations + 1):
            print_info("Starting testing iteration " + str(iter) + "/" + str(iterations))
            train_images, test_images = MagentoClassifier._test_train_split_buildings(buildings,
                                                                                      train_images_per_building=train_images_per_building,
                                                                                      test_images_per_building=test_images_per_building,
                                                                                      class_count=class_count,
                                                                                      seed=seed)
            mc = MagentoClassifier(n_neighbors=n_neighbors, weights=weights)
            mc.fit(train_images)
            score = mc.score(test_images,method=method)
            print_result("Iteration " + str(iter) + "/" + str(iterations) + " score is " + str(score[method_idx]) +" (all scores: "+str(zip(METHODS,score))+")")
            ult_score += score
            print_result("Ultimate score so far is " + str(ult_score[method_idx] / iter)+" (all scores: "+str(zip(METHODS,ult_score/iter))+")")
        score_iterations = ult_score / iterations

        print_result("Ultimate score is " + str(score_iterations[method_idx])+" (all scores: "+str(zip(METHODS,score_iterations))+")")
        return score_iterations

    @staticmethod
    def _test_train_split_buildings(buildings, test_images_per_building=1, train_images_per_building=-1, class_count=-1,
                                    seed=-1):
        """
        :type test_images_per_building: int
        :type train_images_per_building: int
        :type buildings: list[Building]
        """
        mapped = map(lambda building: building.get_test_train_images(train_count=train_images_per_building,
                                                                     test_count=test_images_per_building, seed=seed),
                     buildings)
        random.shuffle(mapped)
        all_trains, all_tests = [], []
        all_classes = 0
        for train, test in mapped:
            if class_count != -1 and all_classes == class_count:
                break
            if (len(train) < train_images_per_building) or (len(test) < test_images_per_building):
                pass
            else:
                all_trains.extend(train)
                all_tests.extend(test)
                all_classes += 1

        if all_classes != class_count:
            assert class_count == -1, "Database too small"
            print_warn("There are not enough samples for all classes")

        print_info(
                "Loaded " + str(all_classes) + " classes, with " + str(train_images_per_building) + " train and " + str(
                        test_images_per_building) + " test images, with " + str(
                        ("seed=" + str(seed)) if seed != -1 else "default seed"))
        return all_trains, all_tests

    def __init__(self, n_neighbors=N_NEIGHBORS, weights=KNN_WEIGHTS):
        self._classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        self._buildings = None # type: None|list[Building]
        self._classes = None # type: None|list[int]
        self._features_all = None # type: None|list[Feature]


    def fit(self, images):
        """

        :type images: list[Images]
        """
        print_info("Starting fitting process")
        assert all([image.get_building() is not None for image in images])
        self._buildings = list(set( [image.get_building() for image in images]))
        self._buildings.sort(key=Building.get_identifier)
        self._classes = map(Building.get_identifier,self._buildings)

        features_all = []
        for image in images:
            print_result("Processing image " + str(image))
            features_all.append(image.get_all_features())
        features_all = [feature for features in features_all for feature in features]

        # features_all = [feature for image in images for feature in image.get_all_features()]
        self._features_all = features_all

        descriptors_all = np.array([feature.get_descriptor() for feature in features_all])
        assert len(descriptors_all.shape) == 2

        classes_all = np.array(
                [feature.get_pyramid_level().get_image().get_building().get_identifier() for feature in features_all])
        assert len(classes_all.shape) == 1

        assert descriptors_all.shape[0] == classes_all.shape[0]

        self._classifier.fit(descriptors_all, classes_all)

    def predict(self, images,method='mode'):
        """

        :type images: list[Image]
        :rtype: list[int]
        """
        print_info("Starting predict process")
        size = float(len(images))
        data = [(image, self, idx / size,method) for idx, image in enumerate(images)]
        if CPU_COUNT == 1:
            return map(predict, data)
        else:
            pool = Pool(CPU_COUNT)
            results = pool.map(predict, data)
            pool.close()
            pool = None
            gc.collect()
            return results

    def score(self, images, method='mode'):
        """
        :type images: list[Image]
        :rtype: np.ndarray
        """
        print_info("Starting scoring process")
        y_pred = np.array(self.predict(images, method=method))
        y_true = np.array([image.get_building().get_identifier() for image in images])[:,np.newaxis]
        return np.sum((y_true - y_pred) == 0,axis=0) / float(y_pred.shape[0])

    def show_match(self, image_test, descriptors_all):
        """
        :type image_test: Image
        :type matches: np.array
        :type distances: np.array
        """
        distances, matches = self._classifier.kneighbors(descriptors_all, return_distance=True, n_neighbors=1)

        # image_train = self._buildings[0].get_images()[0]
        """
        :type image_train: Image
        """
        # image_train_rgb = image_train.get_rgb()
        image_test_rgb = image_test.get_rgb()

        # offset = image_test_rgb.shape[1]
        # size = offset + image_train_rgb.shape[1]

        # showoff = np.zeros((Image.DEFAULT_HEIGHT, size, 3), np.uint8)

        # showoff[0:image_test_rgb.shape[0], 0:image_test_rgb.shape[1], :] = image_test_rgb
        # showoff[0:image_train_rgb.shape[0], 0 + offset:image_train_rgb.shape[1] + offset, :] = image_train_rgb

        # only_circles = showoff.copy()
        # raw_showoff = showoff.copy()
        # only_circles = image_test_rgb.copy()
        # for feature in image_test.get_all_features():
        #     xy1, w1 = feature.get_global_xy_w()
        #     cv2.circle(only_circles, tuple(xy1), w1, (0, 0, 255), thickness=1)
        # for feature in image_train.get_all_features():
        #     xy2, w2 = feature.get_global_xy_w()
        #     xy2 = xy2 + [offset, 0]  # do not += !!!
        #     cv2.circle(only_circles, tuple(xy2), w2, (0, 0, 255), thickness=1)
        # cv2.imshow("ftrs", only_circles)
        # cv2.waitKey()

        matching_score = 0
        for feature, matchs, distancess in zip(image_test.get_all_features(), matches, distances):
            xy1, w1 = feature.get_global_xy_w()
            for m in matchs:
                other_feature = self._features_all[m]
                image_train_rgb = other_feature.get_pyramid_level().get_image().get_rgb()
                xy2, w2 = other_feature.get_global_xy_w()

                offset = image_test_rgb.shape[1]
                size = offset + image_train_rgb.shape[1]
                xy2 += [offset, 0]

                showoff = np.zeros((Image.DEFAULT_HEIGHT, size, 3), np.uint8)

                showoff[0:image_test_rgb.shape[0], 0:image_test_rgb.shape[1], :] = image_test_rgb
                showoff[0:image_train_rgb.shape[0], 0 + offset:image_train_rgb.shape[1] + offset, :] = image_train_rgb

                cv2.line(showoff, tuple(xy1), tuple(xy2), (0, 0, 255), thickness=1)
                cv2.circle(showoff, tuple(xy1), w1, (0, 0, 255), thickness=1)
                cv2.circle(showoff, tuple(xy2), w2, (0, 0, 255), thickness=1)
                plt.imshow(cv2.cvtColor(showoff, cv2.COLOR_RGB2BGR)), plt.show()
                #
                #     matching_score += (xy1 - xy2) ** 2  # + (w1-w2)**2
                # print "", np.sqrt(np.average(matching_score)), " +- ", np.sqrt(np.std(matching_score))
                #
                # FACTOR = 3
                # for feature, matchs, distancess in zip(image_test.get_all_features(), matches, distances):
                #     showoff2 = raw_showoff.copy()
                #     xy1, w1 = feature.get_global_xy_w()
                #
                #     # if all(distances>FACTOR):
                #     #     continue
                #
                #     cv2.circle(showoff, tuple(xy1), w1, (0, 0, 255), thickness=1)
                #     cv2.circle(showoff2, tuple(xy1), w1, (0, 0, 255), thickness=1)
                #
                #     feature.show("test")
                #     for idx, (match, distance) in enumerate(zip(matchs, distancess)):
                #         # if distance>FACTOR:
                #         #     continue
                #         other_feature = image_train.get_all_features()[match]
                #         xy2_, w2 = other_feature.get_global_xy_w()
                #         xy2 = xy2_ + [offset, 0]  # do not += !!!
                #         other_feature.show("train")
                #         print distance
                #
                #         score___ = (1 - (feature._score * other_feature._score) / (15.0 ** 2))
                #         rating = int(score___ * 128 + 127)
                #         distance_ = int(1 / (distance + 1) * 10)
                #         # print distance, distance_, rating, feature._score, other_feature._score
                #         cv2.line(showoff, tuple(xy1), tuple(xy2), (0, 0, rating), distance_)
                #         cv2.line(showoff2, tuple(xy1), tuple(xy2), (0, 0, rating), distance_)
                #         cv2.circle(showoff, tuple(xy2), w2, (0, 0, 255), thickness=1)
                #         cv2.circle(showoff2, tuple(xy2), w2, (0, 0, 255), thickness=1)
                #
                #         # if (xy2_-xy1).dot(xy2_-xy1) < 1000:
                #         cv2.imshow("", showoff2)
                #         k = cv2.waitKey(0)
                #         if k == 27:  # wait for ESC key to exit
                #             cv2.destroyAllWindows()
                #             exit(0)
                #             # def get_stripe(img, f):
                #             #     return cv2.resize(img[f._y - 5:f._y + 5 + 1, f._x - f._w:f._x + f._w + 1], (0, 0), fx=10, fy=10,
                #             #                       interpolation=cv2.INTER_NEAREST)
                #             #
                #             # first_stripe = get_stripe(image_test.get_processed(), feature)
                #             # second_stripe = get_stripe(image_train.get_processed(), other_feature)
                #             # cv2.imshow("A", first_stripe)
                #             # cv2.imshow("B", second_stripe)
                #
                # cv2.imshow("", showoff)
                # cv2.imwrite("../data/outputs/jej/_" + str(random.random()) + "cool.jpg", showoff,
                #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # cv2.waitKey()


class Feature(object):
    WIDTH = 8  # actual width -> 2*WIDTH
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
        """

        :type pyramid_level: PyramidLevel
        :type feature:Feature
        :rtype: np.array
        """
        CHUNKS = 4

        x = self._x
        y = self._y
        w = Feature.WIDTH
        h = Feature.HEIGHT

        angles, weights_all, _ = self._pyramid_level.get_data()
        scan_lines = angles[y - h: y + h + 1, x - w:x + w]
        weights = weights_all[y - h: y + h + 1, x - w:x + w]

        # left, right = _get_left(scan_lines, x, w), _get_right(scan_lines, x, w)
        # left_w, right_w = _get_left(weights, x, w), _get_right(weights, x, w)


        # sum = left - right
        # sum = np.minimum(sum, 360 - sum)
        # sum_w = left_w + right_w

        # avg_line = (sum) / 2
        # avg_line_w = (sum_w) / 2

        # self._descriptor = avg_line.ravel()
        # return

        splitted_scan_lines = np.split(scan_lines, CHUNKS, axis=1)
        splitted_weights = np.split(scan_lines, CHUNKS, axis=1)

        hs = []
        for chunk_lines, chunk_weights in zip(splitted_scan_lines, splitted_weights):
            hs.append(np.histogram(chunk_lines, bins=8, range=(-180, 180), weights=chunk_weights)[0])

        self._descriptor = np.array(hs).flatten()
        return

        # print avg_line[:, :]
        half = avg_line.shape[1] / 2.0

        h1 = np.histogram(avg_line[:, :half], bins=8, range=(-180, 180), weights=avg_line_w[:, :half], density=True)[0]
        h2 = np.histogram(avg_line[:, half:], bins=8, range=(-180, 180), weights=avg_line_w[:, half:], density=True)[0]

        # print h1
        # print h2

        vals = np.append(h1, h2)
        self._descriptor = vals

        # print vals
        #
        # exit(0)
        # raw = (avg_line < np.average(avg_line)).astype(float)
        # # raw *= c1
        # ravel = np.ravel(raw)
        # # assert ravel.shape[0] == (h * 2 + 1) * w, "" + str(ravel.shape) + " " + str((h * 2 + 1) * w)
        # # np.append(ravel,feature.get_global_xy_w()[0])
        # self._descriptor = ravel
        # # return sp.ndimage.interpolation.zoom(raw, 128 / float(w), order=3, mode='constant')

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
        pass
        # # img = self.get_pyramid_level().get_data().copy()
        # # self.draw_me(img)
        # # cv2.imshow(title if title is not "" else self.__repr__(), img)
        # # cv2.waitKey()
        # matrix = self.get_pyramid_level().get_data()
        # scan_lines = matrix[self._y - Feature.HEIGHT: self._y + Feature.HEIGHT + 1, :]
        #
        # area = scan_lines[:, self._x -Feature.WIDTH:self._x + Feature.WIDTH]
        #
        # left = area[:, :area.shape[1] / 2]
        # right = area[:, area.shape[1] / 2:]
        #
        # right = np.fliplr(right)
        #
        # aaaaaa = (left / 2 + right / 2)
        #
        #     # cv2.imshow("sl",cv2.resize(scan_lines[:,x-w:x+w],dsize=(0,0),fx=10,fy=10,interpolation=cv2.INTER_NEAREST))
        #     # k = cv2.waitKey(0)
        #     # if k == 27:         # wait for ESC key to exit
        #     #     cv2.destroyAllWindows()
        #     #     exit(0)
        #
        #
        # avg = np.average(area)
        #     # print avg
        #     # avg = np.tile(avg, (area.shape[1], 1)).T
        #     # print avg
        #
        #
        #
        # area2 = (np.bitwise_not(area > avg)).astype(np.uint8) * 255
        #
        # resize = lambda img: cv2.resize(img, dsize=(0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        #
        # descriptor = self.get_descriptor()
        # desc = np.reshape(descriptor, (area.shape[0], len(descriptor) / area.shape[0]))
        # desc = (desc - desc.min()) / (desc.min() - desc.max()) * 255
        # desc = desc.astype(np.uint8)
        #
        # area = cv2.cvtColor(area, cv2.COLOR_GRAY2BGR)
        # area2 = cv2.cvtColor(area2, cv2.COLOR_GRAY2BGR)
        # desc = cv2.cvtColor(desc.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        # padding = np.zeros((area.shape[0], 5, 3))
        # padding[:] = (0, 0, 255)
        #
        # large_image = np.hstack((padding, area2, padding, desc))
        #
        # large_image[:area.shape[0], :area.shape[1], :] = area
        #
        # cv2.imshow(title, resize(area))
        # cv2.imshow(title + "-aaaaa", resize(aaaaaa))
        # cv2.imshow(title + "-binary", resize(large_image))
        #     # cv2.imshow(title +"-descriptor", resize(desc))
        #     # k = cv2.waitKey(0)
        #     # if k == 27:         # wait for ESC key to exit
        #     #     cv2.destroyAllWindows()
        #     #     exit(0)

    def __repr__(self):
        return "{xyw=(" + str(self._x) + "," + str(self._y) + "," + str(self._w) + "), score=" + str(
                self._score) + ", locality=" + str(self._locality) + "}"

    def get_global_xy_w(self):
        scale = self.get_pyramid_level().get_scale()
        return np.array([int(round(self._x / scale)), int(round(self._y / scale))]), int(round(Feature.WIDTH / scale))


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
        self._features = None
        self._scale = scale

    def get_parent_image(self):
        return self._image

    def get_data(self):
        if self._data is None:
            self._generate_data()
        return self._data

    def _generate_data(self):
        NORM_LIMIT = 500
        SURPRESSING_FACTOR = 0.5
        NECESSARY_ANGLE_OFFSET = 60
        THRESHOLD = 0.05

        # print "Hello"
        img = self._grayscale
        # self._grayscale = None

        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        sobel = 1j * sobel_x + sobel_y  # so the angles are relative to the y-axis

        sobel_norms = np.sqrt((sobel * sobel.conjugate()).real)

        sobel_norms /= NORM_LIMIT
        sobel_norms[sobel_norms > 1] = 1
        sobel_norms[sobel_norms < 1] *= SURPRESSING_FACTOR

        sobel_angles = np.angle(sobel, deg=True).real

        sobel_angles_factor = np.abs(sobel_angles)
        sobel_angles_factor = np.minimum(sobel_angles_factor, 180 - sobel_angles_factor)
        sobel_angles_factor /= NECESSARY_ANGLE_OFFSET
        sobel_angles_factor[sobel_angles_factor > 1] = 1

        sobel_weights = sobel_angles_factor * sobel_norms

        # print sobel_weights[sobel_weights>1].shape

        allowed_area = (sobel_weights == 1)

        ones = np.ones((Feature.HEIGHT * 2 + 1, Feature.WIDTH))
        zeros = ones * 0
        left_test = np.hstack((ones, zeros))
        right_test = np.hstack((zeros, ones))

        # allowed_area = allowed_area.astype(np.uint8)
        # left = ndimage.convolve(allowed_area,left_test,mode='constant',cval=0)
        # right =ndimage.convolve(allowed_area,right_test,mode='constant',cval=0)
        # threshold = THRESHOLD * (Feature.HEIGHT * 2 + 1) * Feature.WIDTH
        # allowed_area = (left > threshold) & (right>threshold)

        allowed_area = morphology.binary_dilation(allowed_area, structure=left_test) & morphology.binary_dilation(
                allowed_area, structure=right_test)

        self._data = (sobel_angles, sobel_weights, allowed_area)
        # plt.subplot(231), plt.imshow(self._grayscale, cmap='gray')
        # plt.subplot(232), plt.imshow(allowed_area, cmap='gray')
        # plt.subplot(233), plt.imshow(sobel_norms, cmap='gray')
        # plt.subplot(234), plt.imshow(sobel_angles, cmap='gray')
        # plt.subplot(235), plt.imshow(sobel_weights,'gray'),plt.show()
        # plt.subplot(235), plt.imshow(sobel_x, cmap='gray')
        # plt.subplot(236), plt.imshow(sobel_y, cmap='gray'), plt.show()
        # print np.min(sobel_x), np.max(sobel_x), np.min(sobel_y), np.max(sobel_y)

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
            self._data = None
            self._grayscale = None
        return self._features

    def _extract_features(self):
        """
        :rtype: list[Feature]
        """
        DISALLOWED_AREA_CONSTANT = 10e10
        DIFFERENCE_THRESHOLD = 10
        IMAGE_BORDER_IGNORE_RATIO = 0.05

        FEATURE_CACHE_PATH = '../data/cache'

        cache_file_name = str(self.get_image().__repr__() + str(self.get_scale())) + ".cache"

        cache_path = os.path.abspath(join(FEATURE_CACHE_PATH, cache_file_name))

        try:
            raise Exception()
            data = np.load(cache_path + '.npy')
            minimums = data[:, :2]
            min_vals = data[:, 2]
        except:
            w = Feature.WIDTH
            h = Feature.HEIGHT

            angles, norms, allowed_area = self.get_data()

            x_offset, y_offset = int(angles.shape[1] * IMAGE_BORDER_IGNORE_RATIO), int(
                    angles.shape[0] * IMAGE_BORDER_IGNORE_RATIO)

            heatmap = np.ones(angles.shape) * DISALLOWED_AREA_CONSTANT

            # kernel_x = cv2.getGaussianKernel(Feature.WIDTH * 2, 0)[np.newaxis, :]
            # kernel_y = cv2.getGaussianKernel(Feature.HEIGHT * 2 + 1, 0)[:, np.newaxis]
            # kernel = kernel_x * kernel_y
            # kernel = kernel[:, kernel.shape[1] / 2:]

            window_size = w * 2 * (h * 2 + 1)

            # cv2.imshow("nja", img)
            for y in range(y_offset + h, angles.shape[0] - h - y_offset):
                scan_line = angles[y - h:y + h + 1, :]
                weight_line = norms[y - h:y + h + 1, :]
                for x in range(x_offset + w, angles.shape[1] - w - x_offset):
                    if allowed_area[y, x]:
                        trace_scan_line_b4 = _get_left(scan_line, x, w) + _get_right(scan_line, x,
                                                                                     w)  # npr ak imamo [ 65 21 172 -156 -25 -60] onda ddobijemo [  65  21  172] + [ -60 -25 -156] = [  5   -4   16]
                        # [  5   -4   16]
                        trace_scan_line_b4 = np.minimum(trace_scan_line_b4,
                                                        360 - trace_scan_line_b4)  # primjer [179] i [179] su udaljeni za 2 a ne za 358 stupnjeva
                        trace_weight_line = (_get_left(weight_line, x, w) * _get_right(weight_line, x, w))
                        # weight_line_sum = np.sum(trace_weight_line)

                        trace_scan_line_b4 = np.abs(trace_scan_line_b4)  # [  5   -4   16] -> [  5   4   16]

                        trace_scan_line = trace_scan_line_b4.copy()
                        trace_scan_line *= trace_weight_line  # * kernel[:,:,0]
                        dot = np.sum(trace_scan_line)
                        trace_weight_line_sum = np.sum(trace_weight_line)
                        if trace_weight_line_sum == 0:
                            print_warn("trace_weight_line_sum is 0")
                            continue
                        dot /= trace_weight_line_sum
                        dot /= len(trace_scan_line)
                        dot += np.sqrt(1 - trace_weight_line_sum / window_size) * 10

                        # dot = np.sqrt(trace_scan_line.dot(trace_scan_line))  # / weight_line_sum
                        dot /= len(trace_scan_line)
                        heatmap[y, x] = dot

                        # if dot < 0:
                        #     img = cv2.cvtColor(self._grayscale.copy(), cv2.COLOR_GRAY2BGR)
                        #     cv2.rectangle(img, (x - w, y - h), (x + w, y + h + 1), color=(0, 0, 255))
                        #     cv2.imshow("jej", img)
                        #     # cv2.imshow("jej",pyramid_level._grayscale[y-h:y+h,x-w:x+w])
                        #     cv2.waitKey()
                        #     # plt.subplot(122),plt.imshow(scan_line[:,x-w:x+w],'gray'),plt.show()
                        #     print "score", dot, x, y
                        #     print np.round(_get_left(scan_line, x, w).astype(np.int)), "left scan"
                        #     print np.round(_get_left(weight_line, x, w), 2), "left scan weights"
                        #     print np.round(-_get_right(scan_line, x, w).astype(np.int)), "right scan"
                        #     print np.round(_get_right(weight_line, x, w), 2), "right scan weights"
                        #     print np.round(trace_scan_line, 2), "trace after"
                        #     print np.round(trace_weight_line, 2), "factors"

            filtered_heatmap = heatmap[heatmap != DISALLOWED_AREA_CONSTANT]
            # if len(filtered_heatmap) > 0:
            #    print np.max(filtered_heatmap), np.min(filtered_heatmap), np.average(filtered_heatmap), np.std(
            #            filtered_heatmap), heatmap.shape

            # heatmap_to_showoff = heatmap.copy()
            # heatmap_to_showoff[heatmap==DISALLOWED_AREA_CONSTANT] =np.average(heatmap_to_showoff[heatmap_to_showoff!=DISALLOWED_AREA_CONSTANT])
            # plt.imshow(heatmap_to_showoff, cmap='gray'), plt.show()

            minimums = np.array(detect_local_minima(heatmap)).T
            img = cv2.cvtColor(self._grayscale, cv2.COLOR_GRAY2BGR)

            min_vals = heatmap[minimums[:, 0], minimums[:, 1]]
            argsort_indices = np.argsort(min_vals)
            minimums = minimums[argsort_indices]
            min_vals = min_vals[argsort_indices]
            LIMIT = 100
            if minimums.shape[0] > LIMIT:
                minimums = minimums[:LIMIT]
                min_vals = min_vals[:LIMIT]
            minimums = minimums[min_vals < DIFFERENCE_THRESHOLD]
            min_vals = min_vals[min_vals < DIFFERENCE_THRESHOLD]

            for xy, minval in zip(minimums, min_vals):
                cv2.circle(img, (xy[1], xy[0]), 1, (0, 0, 255))
                # img[xy[0], xy[1]] = (0, 0, 255 - 0*(minval - np.min(min_vals)) / (np.max(min_vals) - np.min(min_vals)) * 255)

            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            # plt.subplot(121), plt.imshow(img), plt.subplot(122), plt.imshow(allowed_area, 'gray'), plt.show()
            # print minimums.shape
            np.save(cache_path, np.hstack((minimums, min_vals[:, np.newaxis])))

            # distance_matrix = cdist(minimums, minimums, 'euclidean')
            #
            # locality = np.zeros(min_vals.shape)
            # for i, (min_val) in enumerate(min_vals):
            #     min_distance = Image.DEFAULT_WIDTH + Image.DEFAULT_HEIGHT
            #     for j, (other_min_val) in enumerate(min_vals):
            #         if other_min_val > min_val:
            #             distance = distance_matrix[i, j]
            #             if distance < min_distance:
            #                 min_distance = distance
            #     locality = min_distance
            # minimums,min_vals = zip(*[(xy,val) for (xy,val),loc in zip(zip(minimums,min_vals),locality) if loc >5])

        features_in_level = []
        for (y, x), min_val in zip(minimums, min_vals):
            feature = Feature(self, min_val, x, y)
            features_in_level.append(feature)
        if len(minimums) == 0:
            self._features = []
            return
        self._features = features_in_level

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

    def __init__(self, image_path):
        self._image_path = image_path
        self._image_gray = None
        self._image_rgb = None
        self._image_processed = None
        self._building = None  # type: Building

        self._all_features = None  # type: list[Feature]

    def __repr__(self):
        return (str(self._building.get_name()) if self._building is not None else "unknown_class") + " - " + str(
                basename(self._image_path))

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
        pass

class PyramidImage(Image):
    PYRAMID_RATIO = np.sqrt(2)
    PYRAMID_LIMIT = 10

    def __init__(self, image_path):
        super(PyramidImage,self).__init__(image_path)
        self._pyramid = None

    def get_all_features(self):
        """
        :rtype: list[Feature]
        """
        if self._all_features is None:
            self._all_features = [feature for pyramid_level in self.get_pyramid() for feature in
                                  pyramid_level.get_features()]
            pyr_levels = [feature.get_pyramid_level().get_scale() for feature in self._all_features]
            pyr_levels = np.bincount(np.array((-np.log2(pyr_levels)).astype(int)))
            ii = np.nonzero(pyr_levels)[0]
            pyr_levels = zip(ii,pyr_levels[ii])

            print_info("Features count="+str(len(self._all_features))+" features in pyramids = "+ str(pyr_levels))
            self._image_gray = None
            self._image_rgb = None
            self._image_processed = None
            self._pyramid = None
        return self._all_features

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
        while current.shape[0] > PyramidImage.PYRAMID_LIMIT and current.shape[1] > PyramidImage.PYRAMID_LIMIT:
            self._pyramid.append(PyramidLevel(self, current, scale))
            current = PyramidImage.pyr_down(current)
            scale /= 2

    @staticmethod
    def pyr_down(image):
        """

        :param image: np.ndarray
        :rtype: np.ndarray
        """
        return cv2.pyrDown(image)  # , dstsize=(
        # int(image.shape[0] / PyramideImage.PYRAMID_RATIO), int(image.shape[1] / PyramidImage.PYRAMID_RATIO)))

class PyramidFaceImage(PyramidImage):
    def __init__(self, image_path, cascade_path):
        super(PyramidFaceImage,self).__init__(image_path)
        self.cascade_path = cascade_path

    def _lazy_load(self):
        super(PyramidFaceImage, self)._lazy_load()
        face_cascade = cv2.CascadeClassifier(self.cascade_path)
#        cv2.imshow('drek', self._image_rgb)
        faces = face_cascade.detectMultiScale(self._image_gray, 1.3, 5)
        if len(faces) == 0:
            print_warn('No face founds, defaulting to whole image region')
            return
        else:
            face = faces[0]
            print 'Face found:', face
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            self._image_gray = self._image_gray[y:y+h, x:x+w].copy()
            self._image_rgb = self._image_rgb[y:y+h, x:x+w].copy()

class Building(object):
    def __init__(self, identifier, name, images):
        """

        :type identifier: int
        :type images: list[Image]
        """
        self._images = images
        for image in images:
            image.set_building(self)

        self._identifier = identifier
        self._index = None
        self._name = name

    def __hash__(self):
        return self.get_identifier()

    def __repr__(self):
        return self.get_name()

    def get_name(self):
        """

        :rtype: str
        """
        return self._name

    def get_images(self):
        """

        :rtype: list[Image]
        """
        return self._images

    def get_test_train_images(self, train_count=1, test_count=-1, seed=-1):
        """

        :type train_count: int
        :type test_count: int
        :type seed: int
        :rtype: tuple[list[Image],list[Image]]
        """
        if (test_count == -1 and train_count <= len(self._images)) or (
                        train_count + test_count <= len(self._images)):
            images = list(self._images)
            if seed != -1:
                random.seed(seed)
            random.shuffle(images)
            train_images = images[:train_count]
            images = images[train_count:]
            test_images = images if test_count == -1 else images[:test_count]
            return train_images, test_images
        else:
            print_warn("Building " + self.get_name() + " doesn't have enough samples (" + str(
                    len(self._images)) + " < " + str(train_count + abs(test_count)) + ") - SKIPPING")
            return [], []

    def get_identifier(self):
        """

        :rtype: int
        """
        return self._identifier


class ImageLoader2(object):
    @staticmethod
    def is_image_file(path):
        return any([path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.JPG']])

    SWARM = "swarm"

    def __init__(self, data_path):
        self._data_path = data_path
        swarm_folder = join(self._data_path, ImageLoader2.SWARM)

        self._image_files = {
            basename(x[0]): [join(x[0], xx) for xx in x[2] if ImageLoader2.is_image_file(join(x[0], xx))] for x
            in
            walk(swarm_folder) if x[0] != swarm_folder}

    def get_image_files(self):
        return self._image_files

    def get_buildings(self):
        """
        :rtype: list[Building]
        """
        buildings = []
        for idx, (name, image_paths) in enumerate(self._image_files.iteritems()):
            images = []
            for image_path in image_paths:
                image = PyramidFaceImage(image_path, 'haarcascade_frontalface_default.xml')
                # image = PyramidImage(image_path)
                images.append(image)

            building = Building(idx, name, images)
            buildings.append(building)
        return buildings

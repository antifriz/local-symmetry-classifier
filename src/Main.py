import cv2

from libSym.FeatureExtractor import FeatureExtractor
from libSym.ImageLoader import ImageLoader
from libSym.Feature import Feature

imageLoader = ImageLoader(data_path="../data")

train_images = imageLoader.get_train_image_matrices()
test_images = imageLoader.get_test_image_matrices()
print str(train_images)

feature_extractor = FeatureExtractor(7)
train_features = feature_extractor.extract_features_multi(train_images, verbose=True)

test_features = feature_extractor.extract_features(test_images[0], verbose=True)

from sklearn.neighbors import NearestNeighbors
import numpy as np


class FeatureSpaceAnalyzer(object):
    @staticmethod
    def getXandYSets(feature_set):
        X = map(lambda x: x.hashFeatures, feature_set)
        y = map(lambda x: x.id_building, feature_set)
        return X, y

    def __init__(self, features, k_nearest=2, stiffness=0.8):
        """

        :type features: list(Feature)
        """
        self._features = features

        hash_features = map(lambda feature: feature.hash_features, features)

        self._classifier = NearestNeighbors(n_neighbors=k_nearest).fit(np.array(hash_features))

    def neighborhood(self, feature):
        """

        :type feature: Feature
        """
        return self._classifier.kneighbors(feature.hash_features)


feature_space_analyzer = FeatureSpaceAnalyzer(train_features)

all_neighbours = []
for feature in test_features:
    neighbours = feature_space_analyzer.neighborhood(feature)
    distances = neighbours[0][0]
    indices = neighbours[1][0]
    close_features = map(lambda (i, d): feature_space_analyzer._features[i], zip(indices, distances))
    all_neighbours.append([{'feature': feature, 'neighbors': close_features, 'distances': distances}])

    #if distances[0]*1.2>distances[1]:
    #    continue
    print distances[0]
    test_img = feature.image.resized_image.copy()
    point = (int(feature.x), int(feature.y))
    cv2.circle(test_img, point, int(feature.w), color=(0, 0, 255))
    cv2.line(test_img, point, point, color=(0, 0, 255), thickness=3)

    train_img = close_features[0].image.resized_image.copy()

    close_feature = close_features[0]

    print distances

    point = (int(close_feature.x), int(close_feature.y))
    cv2.circle(train_img, point, int(close_feature.w), color=(0, 0, 255))
    cv2.line(train_img, point, point, color=(0, 0, 255), thickness=3)


    train_img = cv2.copyMakeBorder(train_img, 0, 480 - train_img.shape[0], 0, 640 - train_img.shape[1],
                                   cv2.BORDER_CONSTANT)
    test_img = cv2.copyMakeBorder(test_img, 0, 480 - test_img.shape[0], 0, 640 - test_img.shape[1], cv2.BORDER_CONSTANT)



    cv2.imshow("ftr", np.hstack((test_img, train_img)))
    cv2.waitKey(0)

print all_neighbours
# neighbourhood = map(lambda feature: (feature,feature_space_analyzer.neighborhood(feature)), test_features)
# print neighbourhood

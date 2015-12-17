from os import listdir
from os.path import isfile, join

import cv2

from libSym.Image import Image


class ImageLoader(object):
    @staticmethod
    def is_image_file(path):
        return any([path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']])
        pass

    TRAIN = "train"
    TEST = "test"

    def __init__(self, data_path):
        self._data_path = data_path
        self._image_files = {name: [join(self._data_path, name, f) for f in listdir(join(self._data_path, name)) if
                                    isfile(join(self._data_path, name, f)) if ImageLoader.is_image_file(f)] for name in
                             [ImageLoader.TRAIN, ImageLoader.TEST]}

    def get_image_files(self):
        return self._image_files

    def get_test_image_matrices(self):
        return map(lambda (idx, path): Image(path, idx), enumerate(self._image_files[ImageLoader.TEST]))

    def get_train_image_matrices(self):
        return map(lambda (idx, path): Image(path, idx), enumerate(self._image_files[ImageLoader.TRAIN]))

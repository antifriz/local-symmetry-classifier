from os import listdir
from os.path import isfile, join


class ImageLoader:
    @staticmethod
    def is_image_file(path):
        return any([path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']], )
        pass

    _data_path = "../../data"

    def __init__(self):
        self._imageFiles = [f for f in listdir(self.data_path) if isfile(join(self.data_path, f)) if
                            ImageLoader.is_image_file(f)]

    def get_image_files(self):
        return self._imageFiles

import cv2


def preprocess_image(image, max_width=640, max_height=480):
    factor = 1
    if image.shape[0] > max_height:
        factor = max_height / float(image.shape[0])
    if image.shape[1] > max_width:
        f2 = max_width / float(image.shape[1])
        factor = f2 if f2 < factor else factor
    resized = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC) if factor < 1 else image
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized, gray


class Image(object):
    def __init__(self, path, id_building, max_width=640, max_height=480):
        """

        :param str path:
        :param int id_building:
        :param int max_width:
        :param int max_height:
        """
        self._path = path
        self._image = cv2.imread(path)

        self._resized_image, self._res_grayscale_image = preprocess_image(self.image, max_width, max_height)
        self._id_building = id_building

    @property
    def path(self):
        """

        :rtype: str
        """
        return self._path

    @property
    def image(self):
        """

        :rtype: np
        """
        return self._image

    @property
    def resized_image(self):
        """

        :rtype: np
        """
        return self._resized_image

    @property
    def res_grayscale_image(self):
        """

        :rtype: np
        """
        return self._res_grayscale_image

    @property
    def id_building(self):
        """

        :rtype: int
        """
        return self._id_building

    def __str__(self):
        return str({'path': self.path, 'id_building': self.id_building})

    def __repr__(self):
        return self.__str__()

from libSym.Image import Image


class Feature(object):
    def __init__(self, hash_features, x, y, image, id_building, w):
        """

        :param np hash_features:
        :param int x:
        :param int y:
        :param Image image:
        :param int id_building:
        :param int w:
        """
        self.hash_features = hash_features
        self.x = x
        self.y = y
        self.image = image
        self.id_building = id_building
        self.w = w

    def __str__(self):
        return str(self.id_building) + " " + self.image.path + " (" + str(self.x) + "," + str(self.y) + ")"

    def __repr__(self):
        return self.__str__()

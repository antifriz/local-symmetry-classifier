class Feature:
    def __init__(self, hashFeatures, x, y, image, id_building, w):
        self.hashFeatures = hashFeatures
        self.x = x
        self.y = y
        self.image = image
        self.id_building = id_building
        self.w = w

    def __str__(self):
        return str(self.hashFeatures)
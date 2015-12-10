import cv2

def preprocess_image(image, max_width=640, max_height = 480):
    factor = 1
    if image.shape[0]>max_height:
        factor = max_height/float(image.shape[0])
    if image.shape[1]>max_width:
        f2 = max_width/float(image.shape[1])
        factor = f2 if f2 <factor else factor
    resized = cv2.resize(image,(0,0),fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC) if factor <1 else image
    gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    return resized,gray

class Image:
    def __init__(self, path, max_width, max_height, id_building):
        self.path = path
        self.image = cv2.imread(path)
        self.resized_image, self.res_grayscale_image = preprocess_image(self.image, max_width, max_height)
        self.id_building = id_building

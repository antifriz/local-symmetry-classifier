from libSym import magento
from libSym.magento import *

if __name__ == '__main__':
    magento.DATA_PATH = '../data/buildings'
    image_loader = ImageLoader(data_path=magento.DATA_PATH)
    # magento.DATA_PATH = '../faces'
    # image_loader = ImageLoader(data_path=magento.DATA_PATH,haar_cascade='../faces/haarcascade_frontalface_default.xml')
    buildings = image_loader.get_buildings()

    magento.LOG_LEVEL = 0
    magento.CPU_COUNT = 1
    magento.SHOW_DETECTIONS = False
    magento.USE_CACHE = False

    MagentoClassifier.test_on_dataset(buildings, class_count=4, train_images_per_building=2,
                                      test_images_per_building=-1, iterations=3, seed=30, method='sum')

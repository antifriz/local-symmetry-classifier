from libSym import magento
from libSym.magento import *


if __name__ == '__main__':
    magento.DATA_PATH = '../data'
    image_loader = ImageLoader2(data_path=magento.DATA_PATH)
    buildings = image_loader.get_buildings()

    magento.LOG_LEVEL = 4
    magento.CPU_COUNT = 1
    magento.SHOW_DETECTIONS = True

    MagentoClassifier.test_on_dataset(buildings,class_count=4,train_images_per_building=4,test_images_per_building=-1,iterations=3,seed=1,method='sum')

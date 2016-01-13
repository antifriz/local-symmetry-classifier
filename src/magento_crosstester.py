import itertools

from libSym import magento
from libSym.magento import *


data_path = '../data'
image_loader = ImageLoader2(data_path=data_path)
buildings = image_loader.get_buildings()

magento.LOG_LEVEL = 0
magento.CPU_COUNT = 8
magento.SHOW_DETECTIONS = False

MagentoClassifier.test_on_dataset(buildings,class_count=-1,train_images_per_building=4,test_images_per_building=-1,iterations=10,seed=-1,method='sum')

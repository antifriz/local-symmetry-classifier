from libSym.magento import *


data_path = '../data'
image_loader = ImageLoader2(data_path=data_path)
buildings = image_loader.get_buildings()
MagentoClassifier.test_on_dataset(buildings,class_count=4,train_images_per_building=4,test_images_per_building=-1,iterations=10,seed=-1)

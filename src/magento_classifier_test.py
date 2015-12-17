from libSym.image_loader import ImageLoader
from libSym.magento import MagentoClassifier

data_path = '../data'

image_loader = ImageLoader(data_path=data_path)
image_files = image_loader.get_image_files()

mc = MagentoClassifier()
mc.fit(image_files['train'])

for filename in image_files['test']:
    print filename,'->', image_files['train'][mc.predict(filename)]

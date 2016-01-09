from libSym.image_loader import ImageLoader
from libSym.magento import *


def extract_name(path):
    return path.split('/')[-1].split('.')[0][:-1]


def extract_name_withoutclass(path):
    return path.split('/')[-1].split('.')[0][1:-1]


# extracts the first number from the filename (TODO, zasad samo 1 char fml)
def make_label(filename):
    return int(extract_name(filename)[0])


# Extract the first number from the image name and sets that number as the image class (puts it in labels)
# returns the labes for images
def make_labels(filenames):
    labels = list()
    for filename in filenames:
        labels.append(make_label(filename))

    return labels


data_path = '../data'

image_loader = ImageLoader(data_path=data_path)
image_files = image_loader.get_image_files()

train_images = image_files['train']
train_images = ['../data/train/2nd1.jpg']#,'../data/train/2nd2.jpg','../data/train/2nd3.jpg','../data/train/2nd4.jpg','../data/train/2nd5.jpg']
test_image = '../data/train/2nd2.jpg'
train_labels = make_labels(train_images)
test_label = make_label(test_image)

train_images = map(lambda x: Image(x), train_images)
#[train_image.show() for train_image in train_images]

# for train_image in train_images:
#     train_image.show_features()



buildings_images = {}
for label, train_image in zip(train_labels, train_images):
    if label not in buildings_images:
        buildings_images.update({label: []})
    buildings_images[label].append(train_image)

buildings = []
for identifier, building_images in buildings_images.iteritems():
    buildings.append(Building(identifier, building_images))



mc = MagentoClassifier()
mc.fit(buildings)

mc.predict(Image(test_image))




#
#
#
# i = 0
# cnt = 0
# labels_test = make_labels(image_files['test'])
# for filename, label in zip(image_files['test'], labels_test):
#     razred = mc.predict(filename)
#     is_ok = razred == label
#     if is_ok:
#         i += 1
#     cnt += 1
#     print "Testing %s" % (filename)
#     print ('OK' if is_ok else '--'), label, '->', razred
# print 'Hit:', i, 'Miss:', cnt - i, 'Percentage:', str(i / float(cnt) * 100) + "%", 'Database size:', len(
#         image_files['train'])

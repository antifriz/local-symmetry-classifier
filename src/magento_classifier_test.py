from libSym.image_loader import ImageLoader
from libSym.magento import MagentoClassifier

def extract_name(path):
    return path.split('/')[-1].split('.')[0][:-1]

def extract_name_withoutclass(path):
    return path.split('/')[-1].split('.')[0][1:-1]

#extracts the first number from the filename (TODO, zasad samo 1 char fml)
def make_label(filename):
    return int(extract_name(filename)[0])
    

#Extract the first number from the image name and sets that number as the image class (puts it in labels)
#returns the labes for images
def make_labels(filenames):
    labels = list()
    for filename in filenames:
        labels.append(make_label(filename))
        
    return labels

data_path = '../data'

image_loader = ImageLoader(data_path=data_path)
image_files = image_loader.get_image_files()

mc = MagentoClassifier()
labels = make_labels(image_files['train'])
mc.fit(image_files['train'], labels)


i = 0
cnt = 0
labels_test = make_labels(image_files['test'])
for filename, label in zip(image_files['test'], labels_test):
    razred = mc.predict(filename)
    is_ok = razred == label
    if is_ok:
        i += 1
    cnt += 1
    print "Testing %s" % (filename)
    print ('OK' if is_ok else '--'), label, '->', razred
print 'Hit:',i,'Miss:',cnt-i,'Percentage:',str(i / float(cnt) * 100)+"%", 'Database size:',len(image_files['train'])


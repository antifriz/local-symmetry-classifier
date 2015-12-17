from libSym.image_loader import ImageLoader
from libSym.magento import MagentoClassifier

data_path = '../data'

image_loader = ImageLoader(data_path=data_path)
image_files = image_loader.get_image_files()

mc = MagentoClassifier()
mc.fit(image_files['train'])


def extract_name(path):
    return path.split('/')[-1].split('.')[0][:-1]


i = 0
cnt = 0
for filename in image_files['test']:
    predicted = image_files['train'][mc.predict(filename)]
    is_ok = extract_name(predicted) == extract_name(filename)
    if is_ok:
        i += 1
    cnt += 1
    print ('OK' if is_ok else '--'), filename, '->', predicted
print 'Hit:',i,'Miss:',cnt,'Percentage:',str(i / float(cnt) * 100)+"%", 'Database size:',len(image_files['train'])

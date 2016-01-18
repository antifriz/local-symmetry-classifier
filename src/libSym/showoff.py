from magento import *

image_path = '../../data/buildings/swarm/akal_takht/5AkalTakht1.jpg'

image = PyramidImage(image_path)
image_rgb = image.get_rgb()
print 'Pokrecem'

features = image.get_all_features()
ws = set()
all = []
for feature in image.get_all_features():
    xy, w = feature.get_global_xy_w()
    ws.add(w)
    all.append([xy, w])

all = sorted(all, key=lambda x: x[1])
print sorted(ws)
print all
for wi in sorted(ws):
    copy = image_rgb.copy()
    circle = True
    for item in all:
        xy, w = item
        if w != wi: continue
        cv2.circle(copy, tuple(xy), w if circle else 1, (0, 0, 255), thickness=1)
        circle = False
    cv2.imshow('cec', copy)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
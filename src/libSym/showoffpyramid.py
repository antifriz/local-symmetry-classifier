from magento import *

image_path = '../../data/buildings/swarm/hallgirmskirkja/15Hallgrimskirkja5.jpg'

image = PyramidImage(image_path)
print 'Pokrecem'

for i, pyramid_level in enumerate(image.get_pyramid()):
    level_image = pyramid_level.get_grayscale()
    cv2.imwrite('pyramidlevel' + str(i) + '.jpg', level_image)
    # cv2.imshow('lvl', level_image)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
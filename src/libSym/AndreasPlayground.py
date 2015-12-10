import cv2
import numpy as np
import matplotlib.pyplot as plt

def pretty_show(img):
    plt.figure(figsize=(20,20))
    plt.imshow(img,cmap='gray')
    plt.show()

blank_image = np.zeros((480 * 3, 640 * 3))
# pretty_show(blank_image)

im = cv2.imread('angle2.jpg')
resized, gray = preprocess_image(im)
pretty_show(gray)

blank_image[0:gray.shape[0], 0:gray.shape[1]] = gray[:, :]
pretty_show(blank_image)
positions = []
for x in range(3):
    for y in range(3):
        if x == 1 and y == 1:
            continue
        positions.append((480 * y, 640 * x))
print positions

import copy

images_all_images = []
for i in range(0, 9):
    blank_image = np.zeros((480 * 3, 640 * 3))
    img = images[i].res_grayscale_image
    blank_image[480:(480 + img.shape[0]), 640:(640 + img.shape[1])] = img[:, :]

    other_images = copy.deepcopy(images)
    del other_images[i]

    for j in range(0, 8):
        img = other_images[j].res_grayscale_image
        blank_image[positions[j][0]:(positions[j][0] + img.shape[0]),
        positions[j][1]:(positions[j][1] + img.shape[1])] = img[:, :]

    images_all_images.append(blank_image)

for i in images_all_images:
    pretty_show(i)

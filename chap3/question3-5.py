from cv2 import cv2
from numpy.core.fromnumeric import mean

import numpy as np

num_bg_frames = 10

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mean_bg = np.zeros_like(frame, dtype='float')

for i in range(num_bg_frames):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_bg += frame

mean_bg /= num_bg_frames

std_bg = np.zeros_like(mean_bg, dtype='float')
for i in range(num_bg_frames):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    std_bg += (frame - mean_bg)**2

std_bg /= num_bg_frames - 1
std_bg = np.sqrt(std_bg)

cv2.imshow('mean', mean_bg.astype('uint8'))
cv2.imshow('std', std_bg.astype('uint8'))
cv2.waitKey(0)

kernel = np.ones((5, 5), np.uint8)
thresh = 1.75
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    original = np.array(frame)
    display_frame = np.array(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.abs(frame - mean_bg)
    mask[mask < thresh * std_bg] = 0
    mask[mask >= thresh * std_bg] = 255

    mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, kernel)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[0]
    for i in range(1, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    mask[:, :] = 0
    mask[output == max_label] = 255

    display_frame[mask < thresh * std_bg] = [0, 0, 0]
    cv2.imshow('frame', display_frame)
    cv2.imshow('original', original)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('./chap3/images/q3-5a.png', original)
        cv2.imwrite('./chap3/images/q3-5b.png', display_frame)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

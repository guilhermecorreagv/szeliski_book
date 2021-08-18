# importing required libraries of opencv
import cv2
import numpy as np

# importing library for plotting
from matplotlib import pyplot as plt

# reads an input image
img = cv2.imread(
    '/mnt/data/Desktop/Guilherme/git/szeliski/chap3/images/apple.jpg')

equ = np.array(img)
equ = cv2.cvtColor(equ, cv2.COLOR_BGR2YCR_CB)

y = np.array(equ[:, :, 0])
y = cv2.equalizeHist(y)

low = np.percentile(y, 5)
high = np.percentile(y, 95)
y[y <= low] = 0
y[y >= high] = 255

equ[:, :, 0] = y
equ = cv2.cvtColor(equ, cv2.COLOR_YCR_CB2BGR)

########################################################
# Piece of code to make the image a linear combination #
# of the equalized and original image                  #
########################################################
# alpha = 0
# while True:
#     display = (1 - alpha) * img + alpha * equ
#     cv2.imshow('display', display.astype('uint8'))
#     k = cv2.waitKey(1)
#     if k == ord('+'):
#         alpha += 0.01
#     elif k == ord('-'):
#         alpha -= 0.01
#     elif k == ord('q'):
#         break
#     alpha = min(max(alpha, 0.0), 1.0)
########################################################

cv2.imshow('original', img)
cv2.imshow('equalized', equ.astype('uint8'))
cv2.waitKey()
cv2.destroyAllWindows()

# show the plotting graph of an image
plt.hist(equ[:, :, 0].ravel(), 256, color='blue', alpha=0.5, label="B")
plt.hist(equ[:, :, 1].ravel(), 256, color='green', alpha=0.5, label="G")
plt.hist(equ[:, :, 2].ravel(), 256, color='red', alpha=0.5, label="R")
plt.ylim([0, 15000])
plt.xlim([0, 255])
plt.legend(loc='upper right')
plt.title('Equalized Histogram')
plt.show()
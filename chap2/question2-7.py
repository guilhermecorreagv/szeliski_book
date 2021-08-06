from typing import final
import cv2
import numpy as np

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (0, 0, 0)

# Line thickness of 2 px
thickness = 2


def get_correction(img, gamma):
    new_img = img.copy()
    new_img = new_img * gamma
    new_img = cv2.putText(new_img, str(gamma), org, font, fontScale, color,
                          thickness, cv2.LINE_AA)
    return new_img


img = cv2.imread('./images/exposure2.jpg')
img = img[70:, :, :]  # Removes the vertical info

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = hsv[:, :, 0]
s = hsv[:, :, 1]
v = hsv[:, :, 2]
gammas = [1.0, 1.0, 1.0]
display_img = img.copy()
converted = img.copy().astype('float')
converted = converted**(
    1.8)  # Comment this line if you don't want to do gamma correction

intervals = [[0, 200], [200, 400], [400, 600]]  # The division positions
i = 0
delta = 0.01

while True:
    gamma = gammas[i]
    new_v = get_correction(converted[:, intervals[i][0]:intervals[i][1], :],
                           gamma)
    print(np.amax(new_v))
    print(np.amin(new_v))
    print("gamma", gamma)
    new_v = new_v.astype('uint8')

    display_img[:, intervals[i][0]:intervals[i][1], :] = new_v

    final_display_img = display_img.copy().astype('float')
    final_display_img = final_display_img**2.2
    final_display_img = final_display_img * 255 / np.amax(final_display_img)
    final_display_img = final_display_img.astype('uint8')

    cv2.imshow('original', img)
    cv2.imshow('after', final_display_img)
    k = cv2.waitKey(1)
    if k == ord('+'):
        gammas[i] += delta
    elif k == ord('-'):
        gammas[i] -= delta
    elif k == ord('a'):
        i = max(i - 1, 0)
    elif k == ord('d'):
        i = min(i + 1, 2)
    elif k == ord('n'):
        delta *= 2
    elif k == ord('m'):
        delta /= 2
    elif k == ord('s'):
        cv2.imwrite('./saved_exposure.png', display_img)
    elif k == ord('q'):
        break
cv2.destroyAllWindows()

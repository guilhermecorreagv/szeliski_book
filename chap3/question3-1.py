from cv2 import cv2
from typing import final

import numpy as np


def draw_rectangle(pos, width, height, img):
    init = (int(pos[0] - width / 2), int(pos[1] - height / 2))
    end = (int(pos[0] + width / 2), int(pos[1] + height / 2))
    cv2.rectangle(img, init, end, (127, 127, 127), -1)


def draw_sliders(positions, img, init_x, final_x, ys):

    cv2.line(img, (init_x, ys[0]), (final_x, ys[0]), (255, 0, 0), 3)
    cv2.line(img, (init_x, ys[1]), (final_x, ys[1]), (0, 255, 0), 3)
    cv2.line(img, (init_x, ys[2]), (final_x, ys[2]), (0, 0, 255), 3)

    bpos = (int(init_x * (1 - positions[0]) + positions[0] * final_x), ys[0])
    gpos = (int(init_x * (1 - positions[1]) + positions[1] * final_x), ys[1])
    rpos = (int(init_x * (1 - positions[2]) + positions[2] * final_x), ys[2])

    draw_rectangle(bpos, 10, 10, img)
    draw_rectangle(gpos, 10, 10, img)
    draw_rectangle(rpos, 10, 10, img)

    return img


def adjust_channels(img, sliders_pos):
    img[:, :, 0] = img[:, :, 0] * sliders_pos[0]
    img[:, :, 1] = img[:, :, 1] * sliders_pos[1]
    img[:, :, 2] = img[:, :, 2] * sliders_pos[2]
    return img.astype('uint8')


def interface(event, x, y, flags, param):
    global sliders_pos, img, grabbed, slider_idx, display_img, init_x, final_x, ys

    display_img = np.array(img)
    display_img = adjust_channels(display_img, sliders_pos)
    draw_sliders(sliders_pos, display_img, init_x, final_x, ys)

    if event == cv2.EVENT_LBUTTONDOWN:
        if grabbed:
            grabbed = False
        else:
            positions = np.array(sliders_pos)
            color_pos = init_x * (1 - positions) + positions * final_x
            for idx, color_p in enumerate(color_pos):
                if np.linalg.norm([x - color_p, y - ys[idx]]) < 15:
                    print(x, color_p, y, ys[idx])
                    slider_idx = idx
                    grabbed = True
                    break

    elif event == cv2.EVENT_MOUSEMOVE:
        if grabbed:
            new_pos = (x - init_x) / (final_x - init_x)
            new_pos = max(min(new_pos, 1.0), 0.0)
            sliders_pos[slider_idx] = new_pos


if __name__ == '__main__':
    cv2.namedWindow('canvas')

    img = cv2.imread('./chap2/images/lena.jpeg')
    display_img = np.array(img)
    init_x, final_x = int(img.shape[1] * 0.8), int(img.shape[1] * 0.95)
    ys = [
        int(0.9 * img.shape[0]),
        int(0.8 * img.shape[0]),
        int(0.7 * img.shape[0]),
    ]

    sliders_pos = [1.0, 1.0, 1.0]
    grabbed = False
    slider_idx = 0

    cv2.setMouseCallback('canvas', interface)

    while True:
        cv2.imshow('canvas', display_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('./chap3/images/q3-1.png', display_img)

    cv2.destroyAllWindows()
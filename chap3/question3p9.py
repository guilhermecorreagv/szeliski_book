from cv2 import cv2
import numpy as np


def pad(img, mode='zero', number=1):
    shape_out = np.array(img.shape)
    shape_out[:2] += 2 * number

    out = np.ones(shape_out, dtype='uint8')

    if mode == 'zero':
        out *= 0
    elif mode == 'constant':
        out *= 127  # Or any other value

    elif mode == 'clamp':
        # Corners
        out[:number, :number] = img[0, 0]
        out[-number:, :number] = img[-1, 0]
        out[-number:, -number:] = img[-1, -1]
        out[:number, -number:] = img[0, -1]

        # Left and Right pads
        for i in range(img.shape[0]):
            val_left = img[i, 0]
            val_right = img[i, -1]
            for j in range(number):
                out[i + number, j] = val_left
                out[i + number, j + img.shape[1] + number] = val_right

        # Up and Down pads
        for j in range(img.shape[1]):
            val_up = img[0, j]
            val_down = img[-1, j]
            for i in range(number):
                out[i, j + number] = val_up
                out[i + img.shape[0] + number, j + number] = val_down

    elif mode == 'mirror':
        # Corners
        out[:number, :number] = np.fliplr(np.flipud(img[:number, :number]))
        out[-number:, :number] = np.fliplr(np.flipud(img[-number:, :number]))
        out[-number:, -number:] = np.fliplr(np.flipud(img[-number:, -number:]))
        out[:number, -number:] = np.fliplr(np.flipud(img[:number, -number:]))

        out[:number, number:-number] = np.flipud(img[:number, :])
        out[-number:, number:-number] = np.flipud(img[-number:, :])
        out[number:-number, :number] = np.fliplr(img[:, :number])
        out[number:-number, -number:] = np.fliplr(img[:, -number:])
    else:
        print("Unknown padding mode")
        exit()

    out[number:-number, number:-number] = img  # Put the image in place
    return out


if __name__ == '__main__':
    img = cv2.imread('./chap2/images/lena.jpeg')

    out = pad(img, mode='clamp', number=20)
    cv2.imwrite(
        '/mnt/data/Desktop/Guilherme/git/szeliski/chap3/images/lena_clamp.png',
        out)
    cv2.imshow('original', img)

    cv2.imshow('out', out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
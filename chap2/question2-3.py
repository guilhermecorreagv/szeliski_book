from cv2 import cv2
import numpy as np
from numba import njit, prange
import time

tile_size = 5
image_resolution = np.array([1200, 800])
img = np.zeros(image_resolution[::-1])
optical_axis = np.array(
    [0, 1, 0]
)  # Medium pixel is given in the direction of camera_center + lambda * optical_axis
optical_axis = np.divide(optical_axis, np.linalg.norm(optical_axis))

fov = 60  # In degrees


@njit(parallel=True)
def get_view(img, camera_center, optical_axis):
    original_phi = np.arccos(optical_axis[2])

    original_theta = np.arccos(optical_axis[0] /
                               np.linalg.norm(optical_axis[:2]))

    n = img.shape[0]
    m = img.shape[1]
    for i in prange(n):
        delta_phi = (i * fov / img.shape[0] -
                     fov / 2) * np.pi / 180  # In radians
        phi = original_phi + delta_phi

        for j in range(m):
            delta_theta = (j * fov / img.shape[1] -
                           fov / 2) * np.pi / 180  # In radians
            theta = original_theta + delta_theta

            # Direction of the ray comming from the center
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            # Now we find the intersection with the z = 0 plane
            # Center + lambda * direction = [..., ..., 0]

            l = -camera_center[2] / direction[2]

            pt_intersec = camera_center + l * direction
            v1 = pt_intersec - camera_center

            if np.dot(v1, direction) < 0:
                # img[i, j] = int(255 * np.sin(i * np.pi / img.shape[0]))
                img[i, j] = int(255 *
                                np.exp(-((i - img.shape[0] / 2)**2 +
                                         (j - img.shape[1] / 2)**2) / 100000))
                continue

            x_sign = 1 if (pt_intersec[0] // tile_size) % 2 == 0 else -1
            y_sign = 1 if (pt_intersec[1] // tile_size) % 2 == 0 else -1
            sign = x_sign * y_sign

            img[i, j] = 0 if sign == 1 else 255
    return img


if __name__ == '__main__':
    if np.linalg.norm(optical_axis[:2]) < 10E-5:
        print("Theta is not well defined for this optical axis")
        exit()

    camera_center = np.array([0, 0, 25])
    commands = [ord('w'), ord('a'), ord('s'), ord('d')]
    rot_degree = 5  # In degrees, for the visualizer
    rot_pos = np.array(
        [[np.cos(rot_degree * np.pi / 180), -np.sin(rot_degree * np.pi / 180)],
         [np.sin(rot_degree * np.pi / 180),
          np.cos(rot_degree * np.pi / 180)]])
    rot_neg = rot_pos.copy().T

    img = get_view(img, camera_center,
                   optical_axis)  # One execution just for numba to compile

    k = ""
    while True:
        cv2.imshow('img', img.astype('uint8'))
        if k in commands:
            start = time.time()
            if k == ord('w'):
                camera_center = camera_center + optical_axis
            elif k == ord('s'):
                camera_center = camera_center - optical_axis
            elif k == ord('a'):
                optical_axis[:2] = rot_pos @ optical_axis[:2]
            elif k == ord('d'):
                optical_axis[:2] = rot_neg @ optical_axis[:2]
            img = get_view(img, camera_center, optical_axis)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

# cv2.imwrite("./infinite_checkerboard.png", img)
import cv2
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


@njit(parallel=True)
def generate_lists(hsv_frame, frame):
    xlist = np.zeros(int(hsv_frame.shape[0] * hsv_frame.shape[1]))
    ylist = np.zeros(int(hsv_frame.shape[0] * hsv_frame.shape[1]))
    clist = np.zeros((int(hsv_frame.shape[0] * hsv_frame.shape[1]), 3))
    for i in prange(hsv_frame.shape[0]):
        for j in prange(hsv_frame.shape[1]):
            xlist[i * hsv_frame.shape[1] + j] = hsv_frame[i, j, 0]
            ylist[i * hsv_frame.shape[1] + j] = hsv_frame[i, j, 1]
            clist[i * hsv_frame.shape[1] +
                  j] = [frame[i, j, 2], frame[i, j, 1], frame[i, j, 0]]
    return xlist, ylist, clist


def save_plot_scatter(cap):
    ret, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xlist, ylist, clist = generate_lists(hsv_frame, frame)
    clist = np.array(clist) / 255
    ax.scatter(xlist, ylist, c=clist)
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    cv2.imwrite('./frame.png', frame)
    plt.savefig('./frame_scatterplot.png', dpi=400)


def fill_image(hk, mesh):
    m, n = hk.shape[:2]
    for i in range(mesh.shape[0]):
        for j in range(mesh.shape[1]):
            mesh[i, j] = hk[i % m, j % n]


cap = cv2.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)
save_plot_scatter(cap)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv_frame[:, :, 0]
    s = hsv_frame[:, :, 1]
    v = hsv_frame[:, :, 2]

    mask = (h < 250) & (h > 5) & (s > 75)
    mask = mask.astype('uint8')
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    frame[mask.astype('bool')] = [0, 255, 0]
    cv2.imshow('frame', frame)
    cv2.imshow('hue', h)
    cv2.imshow('saturation', s)
    cv2.imshow('value', v)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

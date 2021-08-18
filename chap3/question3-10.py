import numpy as np
from cv2 import cv2
from question3p9 import pad
import time


def prepare_svd(kernel, order=1):
    '''
    Will take a Kernel of dimension K x K, and use the "order" vectors 
    with highest energy from the SVD as an approximation to the kernel. Since they are separable
    '''
    U, S, VT = np.linalg.svd(kernel, full_matrices=False)
    vectors = []
    reconstruction = np.zeros_like(kernel, dtype='float')
    for i in range(min(order, U.shape[1])):
        row, col = VT[i, :], U[:, i]
        row = np.expand_dims(row, 1).T
        col = np.expand_dims(col, 1)
        vectors.append([S[i] * row, col])
        m = S[i] * (col @ row)
        reconstruction += m
    print("Frobenius norm of the reconstruction error:",
          np.linalg.norm(kernel - reconstruction))
    return vectors


def separable_conv(img, row, col, buffer, out):
    offset = col.shape[0] // 2
    shape = img.shape
    # Now with the column vector
    for i in range(offset, shape[0] - offset):
        for j in range(shape[1]):
            buffer[i - offset,
                   j] = np.sum(img[i - offset:i + offset + 1, j] * col)

    # cv2.imshow('buffer', buffer.astype('uint8'))
    # cv2.waitKey(0)

    # Convolution with the row vector first
    for i in range(shape[0]):
        for j in range(offset, shape[1] - offset):
            out[i, j - offset] = np.sum(buffer[i, j - offset:j + offset + 1] *
                                        row)
    return np.abs(out)


def normal_conv(img, kernel, out):
    offset = kernel.shape[0] // 2
    shape = img.shape
    for i in range(offset, shape[0] - offset):
        for j in range(offset, shape[1] - offset):
            out[i - offset, j - offset] = np.abs(
                np.sum(
                    img[i - offset:i + offset + 1, j - offset:j + offset + 1] *
                    kernel))
    return out


if __name__ == '__main__':
    kernel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
    kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 4

    img = cv2.imread('./chap2/images/lena.jpeg', cv2.IMREAD_GRAYSCALE)

    pad_size = 200
    if pad_size > 0:
        img = pad(img, mode='mirror', number=pad_size)

    start_separable = time.time()

    buffer = np.zeros(img.shape, dtype='float')
    out1 = np.zeros(img.shape, dtype='float')
    arrays = prepare_svd(kernel1)
    for arr in arrays:
        row, col = arr[0], arr[1]
        row, col = np.squeeze(row), np.squeeze(col)
        out1 = separable_conv(img.astype('float'), row, col, buffer, out1)

    out2 = np.zeros(img.shape, dtype='float')
    arrays = prepare_svd(kernel2)
    for arr in arrays:
        row, col = arr[0], arr[1]
        row, col = np.squeeze(row), np.squeeze(col)
        out2 = separable_conv(img.astype('float'), row, col, buffer, out2)

    final_out = np.zeros(img.shape, dtype='float')
    final_out = np.sqrt(out1**2 + out2**2)

    final_separable = time.time()

    # For normal convolution
    start_normal = time.time()
    outx = np.zeros(img.shape)
    outy = np.zeros(img.shape)
    final_out_normal = np.zeros(img.shape)
    buffer = np.zeros(img.shape, dtype='float')

    outx = normal_conv(img, kernel1, outx)
    outy = normal_conv(img, kernel2, outy)

    final_out_normal = np.sqrt(outx**2 + outy**2)
    final_normal = time.time()

    cv2.imshow('final', final_out.astype('uint8'))
    cv2.imshow('final normal', final_out_normal.astype('uint8'))
    # cv2.imshow('equalized', equ)
    # cv2.imshow('out normal', out_normal.astype('uint8'))
    # cv2.imshow('out separable', out_separable.astype('uint8'))
    cv2.waitKey()
    cv2.destroyAllWindows()

    print("Time for separable:", final_separable - start_separable)
    print("Time for normal:", final_normal - start_normal)
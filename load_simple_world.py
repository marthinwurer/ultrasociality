
import imageio
import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np
from netCDF4 import Dataset


def get_aimfs():
    """
    get the absolute value of the image, floating point, sobel filtered
    """
    im = imageio.imread("~/Downloads/datasets/elevation/elev_bump_8k.jpg")
    imf = np.asarray(im, np.float)

    imfs = np.sqrt(scipy.ndimage.sobel(imf, -1) ** 2 + scipy.ndimage.sobel(imf, -2) ** 2)
    aimfs = abs(imfs)
    return aimfs


def get_water():
    """
    get the absolute value of the image, floating point, sobel filtered
    """
    im = imageio.imread("~/Downloads/datasets/elevation/water_8k.png")
    imf = np.asarray(im, np.float)
    return imf


def get_precipitation():
    rootgrp = Dataset("~/Downloads/datasets/precip.mon.total.2.5x2.5.v7.nc", "r")


def main():
    im = imageio.imread("~/Downloads/datasets/elevation/elev_bump_8k.jpg")
    imf = np.asarray(im, np.float)

    imfs = np.sqrt(scipy.ndimage.sobel(imf, -1) ** 2 + scipy.ndimage.sobel(imf, -2) ** 2)
    aimfs = abs(imfs)
    plt.imshow(aimfs)
    plt.show()
    pass


def aggregate_2d(im, dims: tuple, func=np.mean):
    """
    aggregate an image with the values in those dimensions
    Args:
        im:
        dims: tuple of step size for each dimension
        func: the aggregation function to use. defaults to average.
    """
    shape = im.shape
    y_steps = shape[0] // dims[0]
    x_steps = shape[1] // dims[1]

    output = np.zeros((y_steps, x_steps))

    for y_step in range(y_steps):
        for x_step in range(x_steps):
            start_y = y_step * dims[0]
            end_y = start_y + dims[0]
            start_x = x_step * dims[1]
            end_x = start_x + dims[1]
            im_slice = im[start_y:end_y, start_x:end_x]
            val = func(im_slice)
            output[y_step, x_step] = val

    return output


def find_closest_palette_color(val):
    if val < 128:
        return 0
    else:
        return 255

def min_max_water(water):
    w = np.zeros(water.shape)
    upper_mask = water > 127
    w[upper_mask] = 255
    return w


def fs_dithering(im: np.ndarray):
    """code modified from
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
    """
    pixel = im.copy()
    xl = im.shape[1]
    yl = im.shape[0]
    for y in range(yl):
        for x in range(xl):
            oldpixel = pixel[y][x]
            newpixel = find_closest_palette_color(oldpixel)
            pixel[y][x] = newpixel
            quant_error = oldpixel - newpixel
            if x < xl - 1:
                pixel[y][x + 1] = pixel[y][x + 1] + quant_error * 7 / 16
            if x != 0 and y < yl - 1:
                pixel[y + 1][x - 1] = pixel[y + 1][x - 1] + quant_error * 3 / 16
            if x < xl - 1 and y < yl - 1:
                pixel[y + 1][x + 1] = pixel[y + 1][x + 1] + quant_error * 1 / 16
            if y < yl - 1:
                pixel[y + 1][x] = pixel[y + 1][x] + quant_error * 5 / 16
    return pixel


if __name__ == '__main__':
    main()



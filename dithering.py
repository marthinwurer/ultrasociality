import numpy as np


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


def dither_mask(mask):
    cop = 255 * mask
    dithered = fs_dithering(cop)
    out = dithered == 255
    return out

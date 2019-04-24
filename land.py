import argparse

import imageio
import pyfastnoisesimd as fns
from opensimplex import OpenSimplex
import numpy as np
import matplotlib.pyplot as plt
import math as m
from enum import Enum
from scipy.ndimage.filters import gaussian_filter

# height = 512
# width = 1024
# height = 128
# width = 256
sea_level = 0.6
inclination = 1

"""
Better agriculture/biome makes it easier to gain ultrasociality traits? Worse makes it easier to lose them?
Changes the total amount possible? Mild AI towards getting better land?
"""


def normalize_map(map):
    max = np.max(map)
    min = np.min(map)
    map = (map - min) / (max - min)
    return map


class NoiseWrapper():
    def __init__(self, **kwargs):
        self.noise = OpenSimplex(**kwargs)
        self.fast_noise = fns.Noise()
        # self.fast_noise.frequency = 0.05

    def polar_noise(self, theta, phi, rho):
        """Generates noise at the given polar coordinates

        Args:
            theta: inclination (north-south)
            phi: rotation around center (east-west)
            rho: The radius (altitude)

        Returns:
        """

        # calculate cartesian coordinates
        x = rho * m.sin(theta) * m.cos(phi)
        y = rho * m.sin(theta) * m.sin(phi)
        z = rho * m.cos(theta)
        return self.noise.noise3d(x, y, z)

    def polar_fast_noise(self, h, w, o):
        numCoords = h * w
        coords = fns.emptyCoords(numCoords)
        for y in range(h):
            for x in range(w):
                theta = (y + .5) / (w/2) * m.pi
                phi = x / h * m.pi
                xv = o * m.sin(theta) * m.cos(phi)
                yv = o * m.sin(theta) * m.sin(phi)
                zv = o * m.cos(theta)
                index = y*w+x
                coords[0][index] = xv
                coords[1][index] = yv
                coords[2][index] = zv

        result = self.fast_noise.genFromCoords(coords)

        return result.reshape((h, w))

    def noise(self, nx, ny):
        # Rescale from -1.0:+1.0 to 0.0:1.0
        return self.noise.noise2d(nx, ny) / 2.0 + 0.5

    def get_normalized_world(self, h, w, octaves):
        map = np.zeros((h, w))

        for y in range(h):
            for x in range(w):
                for e in range(octaves):
                    theta = y / (w/2) * m.pi
                    phi = x / w * m.pi
                    v = 2**e
                    map[y][x] += self.polar_noise(theta, phi, v) * 1/v
        return normalize_map(map)

    def get_normalized_world_fast(self, h, w, octaves, offset:int=6):
        map = np.zeros((h, w))
        for e in range(offset, octaves + offset):
            v = 2**e
            print(v)
            map += self.polar_fast_noise(h, w, v) / v
            # self.fast_noise = fns.Noise()

        return normalize_map(map)



def hillshade(array:np.ndarray, azimuth:int=315, angle_altitude:int=45):
    """ from
    http://geoexamples.blogspot.com/2014/03/shaded-relief-images-using-gdal-python.html

    use values from http://www.shadedrelief.com/web_relief/
    Args:
        array:
        azimuth:
        angle_altitude:

    Returns:

    """
    x, y = np.gradient(array)
    slope = m.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.
    altituderad = angle_altitude * m.pi / 180.

    shaded = np.sin(altituderad) * np.sin(slope) \
             + np.cos(altituderad) * np.cos(slope) \
               * np.cos(azimuthrad - aspect)
    return (shaded + 1) / 2

def base_temp(h, w):
    """

    """
    map = np.zeros((h, w))

    for y in range(h):
        latitude_percent = (1.0 - (y + .5) / (h / 2))

        if abs(latitude_percent) < .2:
            temp = 27 + 273.15
        else:
            temp = abs(latitude_percent) * -67.5 + 40.5 + 273.15
        # print(latitude_percent * 90, temp - 273.15)

        for x in range(w):
            map[y][x] = temp
    return map

def base_moisture(h, w):
    map = np.zeros((h, w))

    for y in range(h):
        latitude_percent = (1.0 - (y + .5) / (h / 2))
        latitude_rads = latitude_percent * np.pi / 2
        lr = latitude_rads

        c = m.cos(latitude_rads)
        moist = 2 * c * m.cos(6.6 * lr) + 3.5 * c + .25

        for x in range(w):
            map[y][x] = moist
    return map

class Biome(Enum):
    OCEAN = 0
    DESERT = 1
    SAVANNAH = 2
    FOREST = 3
    RAINFOREST = 4
    GRASSLAND = 5
    TAIGA = 6
    TUNDRA = 7
    ICE = 8

color = {
    Biome.OCEAN: (0, 0,255),
    Biome.DESERT: (240,230,140),
    Biome.SAVANNAH: (255,225,0),
    Biome.FOREST: (0,128,0),
    Biome.RAINFOREST: (0,100, 0),
    Biome.GRASSLAND: (173,255,47),
    Biome.TAIGA: (46,139,87),
    Biome.TUNDRA: (102,205,170),
    Biome.ICE: (240,240,255),

}




def whittaker(t, moisture):
    if t > 18:
        if moisture < 50:
            return Biome.DESERT
        elif moisture < 150:
            return Biome.SAVANNAH
        elif moisture < 250:
            return Biome.FOREST
        else:
            return Biome.RAINFOREST
    elif t > 3:
        if moisture < 25:
            return Biome.DESERT
        elif moisture < 150:
            return Biome.GRASSLAND
        elif moisture < 225:
            return Biome.FOREST
        else:
            return Biome.RAINFOREST
    elif t > -5:
        if moisture < 25:
            return Biome.DESERT
        elif moisture < 50:
            return Biome.GRASSLAND
        else:
            return Biome.TAIGA
    elif t > -15:
        return Biome.TUNDRA
    else:
        return Biome.ICE


def gen_heightmap(h, w, s):
    n = NoiseWrapper()
    n.fast_noise.seed = s
    n.fast_noise.perturb.perturbType = fns.PerturbType.GradientFractal
    n.fast_noise.perturb.frequency = .001
    n.fast_noise.perturb.octaves = 2

    heightmap = n.get_normalized_world_fast(h, w, 6)

    rn = NoiseWrapper()
    rn.fast_noise.seed = s + 1 # best practices, amirite?
    roughness_map = rn.get_normalized_world_fast(h, w, 3, offset=5)

    heightmap = (heightmap - sea_level).clip(0) / sea_level
    heightmap **= roughness_map * 2 + 1.5 #1.5

    height_m = heightmap * 9000.
    m_adibatic_change = height_m * -5e-3
    base_temp_map = base_temp(h, w)
    t = NoiseWrapper()
    t.fast_noise.seed = s + 2
    temp_map = (t.get_normalized_world_fast(h, w, 3) - .5) * 10 + base_temp_map + m_adibatic_change

    temp_map_c = temp_map - 273.15

    l = NoiseWrapper()
    l.fast_noise.seed = s+3
    moisture_map = base_moisture(h, w)
    moisture_map *= l.get_normalized_world_fast(h, w, 1)  * 2  # kg/m^2/Day

    moisture_map = moisture_map * 365 / 10.0  # change to cm/year


    biomes = [[Biome.OCEAN for x in range(w)] for y in range(h)]

    for y in range(h):
        for x in range(w):
            val = whittaker(temp_map_c[y][x], moisture_map[y][x])
            if height_m[y][x] > 0.0 or val == Biome.ICE:
                biomes[y][x] = val

    return height_m, temp_map_c, moisture_map, biomes

def gen_image(height_m, biomes):
    image = np.zeros(height_m.shape + (3,))

    srm = normalize_map(hillshade(height_m)) + .2
    srm_image = np.reshape(srm, srm.shape + (1,))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            val = biomes[y][x]
            val = color[val]
            image[y][x] = val

    image = ((image * srm_image)/255.).clip(0, 1)
    return image


def main():
    parser = argparse.ArgumentParser(description="Build a new, random world")
    parser.add_argument('height', type=int, help="The y component")
    parser.add_argument('width', type=int, help="The x component")
    parser.add_argument('seed', type=int, help="The seed for the noise")
    args = parser.parse_args()
    height = args.height
    width = args.width



    height_m, temp_map_c, moisture_map, biomes = gen_heightmap(height, width, args.seed)
    image = gen_image(height_m, biomes)

    imageio.imwrite("map.png", image)

    plt.imshow(gen_image(height_m, biomes))
    plt.show()


def fastnoisetest():
    n = NoiseWrapper()

    image = n.get_normalized_world_fast(128, 256, 10)
    plt.imshow(image)
    plt.show()

def perturb_test():

    n = NoiseWrapper()
    n.fast_noise.seed = 1337
    n.fast_noise.perturb.perturbType = fns.PerturbType.GradientFractal
    n.fast_noise.perturb.frequency = .001
    n.fast_noise.perturb.octaves = 2


    image = (n.get_normalized_world_fast(128, 256, 6) - .5).clip(0,1)
    # image = n.polar_fast_noise(height, width, 1)
    plt.imshow(image, cmap='terrain')
    plt.show()

def test_moist():
    m = NoiseWrapper()
    m.fast_noise.seed = 133337
    moisture_map = base_moisture(128, 256)
    moisture_map += m.get_normalized_world_fast(128, 256, 3)  # cm/year
    plt.imshow(moisture_map)
    plt.show()


if __name__=="__main__":
    # fastnoisetest()
    # base_temp(height, width)
    # perturb_test()
    # test_moist()
    main()

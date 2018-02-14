import imageio
import pyfastnoisesimd as fns
from opensimplex import OpenSimplex
import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.ndimage.filters import gaussian_filter

height = 512
width = 1024
# height = 128
# width = 256
sea_level = 0.65
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
                theta = (y + .5) / (width/2) * m.pi
                phi = x / height * m.pi
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
                    theta = y / (width/2) * m.pi
                    phi = x / height * m.pi
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


def main():
    n = NoiseWrapper()
    n.fast_noise.seed = 1337
    n.fast_noise.perturb.perturbType = fns.PerturbType.GradientFractal
    n.fast_noise.perturb.frequency = .001
    n.fast_noise.perturb.octaves = 2

    # image = n.polar_fast_noise(height, width, 1)
    # plt.imshow(image)
    # plt.show()

    image = np.zeros((height, width, 3))

    heightmap = n.get_normalized_world_fast(height, width, 6)
    rn = NoiseWrapper()
    rn.fast_noise.seed = 133377
    roughness_map = rn.get_normalized_world_fast(height, width, 3, offset=5)
    heightmap = (heightmap - sea_level).clip(0) / sea_level
    heightmap **= roughness_map + 1 #1.5


    height_m = heightmap * 9000.
    m_adibatic_change = height_m * -5e-3
    base_temp_map = base_temp(height, width)
    t = NoiseWrapper()
    t.fast_noise.seed = 13337
    temp_map = (t.get_normalized_world_fast(height, width, 3) - .5) * 10 + base_temp_map + m_adibatic_change

    temp_map_c = temp_map - 273.15

    m = NoiseWrapper()
    m.fast_noise.seed = 133337

    moisture_map = m.get_normalized_world_fast(height, width, 3) * 300. # cm/year








    # changed = True
    # min = 1.0
    # while changed:
    #     changed = False
    #     for y in range(height):
    #         for x in range(width):
    #             if moisture_map[y][x] < min:
    #                 changed = True
    #                 north = moisture_map[y-1][x]
    #                 south = moisture_map[(y+1)%height][x]
    #                 east = moisture_map[y][(x+1)%width]
    #                 west = moisture_map[y][x-1]
    #                 moisture_map[y][x] = max([north, south, east, west]) - 1
    #     moisture_map += 1

    # plt.imshow(moisture_map)
    # plt.show()





    srm = normalize_map(hillshade(heightmap)) +.2
    plt.imshow(srm)
    plt.show()
    srm_image = np.reshape(srm, (height, width, 1))



    r = 0
    g = 1
    b = 2
    for y in range(height):
        for x in range(width):
            rv = 0
            gv = 0
            bv = 0
            if temp_map_c[y][x] < -10:
                rv = .75
                gv = .75
                bv = 1.
            elif heightmap[y][x] > 0.:
                gv = heightmap[y][x] / 2 + .5
            else:
                bv = 1.0


            image[y][x][r] = rv
            image[y][x][g] = gv
            image[y][x][b] = bv


    image = (image * srm_image).clip(0, 1)
    plt.imshow(image)
    plt.show()

    imageio.imwrite("map.png", image)

def fastnoisetest():
    n = NoiseWrapper()

    image = n.get_normalized_world_fast(height, width, 10)
    plt.imshow(image)
    plt.show()

def perturb_test():

    n = NoiseWrapper()
    n.fast_noise.seed = 1337
    n.fast_noise.perturb.perturbType = fns.PerturbType.GradientFractal
    n.fast_noise.perturb.frequency = .001
    n.fast_noise.perturb.octaves = 2


    image = (n.get_normalized_world_fast(height, width, 6) - .5).clip(0,1)
    # image = n.polar_fast_noise(height, width, 1)
    plt.imshow(image, cmap='terrain')
    plt.show()


if __name__=="__main__":
    # fastnoisetest()
    # base_temp(height, width)
    # perturb_test()
    main()

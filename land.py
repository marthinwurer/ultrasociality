import imageio
import pyfastnoisesimd as fns
from opensimplex import OpenSimplex
import numpy as np
import matplotlib.pyplot as plt
import math as m

height = 512
width = 1024
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
            self.fast_noise = fns.Noise()

        return normalize_map(map)



def hillshade(array, azimuth, angle_altitude):
    """ from
    http://geoexamples.blogspot.com/2014/03/shaded-relief-images-using-gdal-python.html
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

def main():
    n = NoiseWrapper()

    # image = n.polar_fast_noise(height, width, 1)
    # plt.imshow(image)
    # plt.show()

    image = np.zeros((height, width, 3))
    r = 0
    g = 1
    b = 2

    heightmap = n.get_normalized_world_fast(height, width, 6)
    # roughness_map = n.get_normalized_world_fast(height, width, 3)
    heightmap **= 1.5

    # base_temp_map = n.get_normalized_world(height, width, 10)

    srm = normalize_map(hillshade(heightmap, 315, 45)) + .2
    # plt.imshow(srm)
    # plt.show()
    srm_image = np.reshape(srm, (height, width, 1))



    for y in range(height):
        for x in range(width):
            rv = 0
            gv = 0
            bv = 0
            if heightmap[y][x] < .5:
                bv = 1.0
            else:
                gv = heightmap[y][x]


            image[y][x][r] = rv
            image[y][x][g] = gv
            image[y][x][b] = bv


    image = image * srm_image
    plt.imshow(image)
    plt.show()

    imageio.imwrite("map.png", image)

def fastnoisetest():
    n = NoiseWrapper()

    image = n.get_normalized_world_fast(height, width, 10)
    plt.imshow(image)
    plt.show()



if __name__=="__main__":
    # fastnoisetest()
    main()

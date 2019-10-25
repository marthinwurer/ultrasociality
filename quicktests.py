import unittest

from dithering import fs_dithering
from load_simple_world import *
from pprint import pprint


class TestStuff(unittest.TestCase):
    def test_water_dithering(self):
        water = min_max_water(get_water())
        aggregated_water = aggregate_2d(water, (22, 22))
        dithered_water = fs_dithering(aggregated_water)
        plt.imshow(dithered_water)
        plt.show()

    def test_precip_load(self):
        rootgrp = Dataset("/home/benjamin/Downloads/datasets/precip.mon.total.2.5x2.5.v7.nc", "r")
        print(rootgrp)
        pprint(rootgrp.dimensions)
        pprint(rootgrp.groups)
        pprint(rootgrp.variables)


if __name__ == '__main__':
    unittest.main()


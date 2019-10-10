import os
import unittest
from pprint import pprint
import gdal
import georasters as gr
import matplotlib.pyplot as plt
from gdalconst import GA_ReadOnly
from geo_scripts.process_height import *


class TestStuff(unittest.TestCase):
    def test_load_file(self):
        DATA = "~/Downloads/datasets/elevation/viewfinder_dem3/15-J.tif"  # from http://www.viewfinderpanoramas.org/dem3.html
        DATA = os.path.expanduser(DATA)

        file = gdal.Open(DATA, GA_ReadOnly)
        print(file)

    def test_get_dims(self):
        DATA = "~/Downloads/datasets/elevation/viewfinder_dem3/15-J.tif"  # from http://www.viewfinderpanoramas.org/dem3.html
        DATA = os.path.expanduser(DATA)
        data = gr.from_file(DATA)
        print(data.geot)
        NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(DATA)
        print(NDV, xsize, ysize, GeoT, DataType)

    def test_get_index(self):
        DATA = "~/Downloads/datasets/elevation/viewfinder_dem3/15-J.tif"  # from http://www.viewfinderpanoramas.org/dem3.html
        DATA = os.path.expanduser(DATA)
        data = gr.from_file(DATA)

        print(map_indexes(data, 13, 13))

    def test_gen_ranges(self):
        range = list(gen_ranges(1, 10, 1))
        pprint(range)
        range = list(gen_ranges(10, 1, 1))
        pprint(range)
        range = list(gen_ranges(-10, -1, 1))
        pprint(range)
        range = list(gen_ranges(-1, -10, 1))
        pprint(range)

    def test_aggregate(self):
        DATA = "~/Downloads/datasets/elevation/viewfinder_dem3/15-J.tif"  # from http://www.viewfinderpanoramas.org/dem3.html
        DATA = os.path.expanduser(DATA)
        x_ranges = list(gen_ranges(0, 60, 1))
        y_ranges = list(gen_ranges(45, 0, 1))
        data = gr.from_file(DATA)
        aggregated = aggregate_grid(data, x_ranges, y_ranges)
        print(aggregated)
        plt.imshow(aggregated)
        plt.show()

    def test_load_chelsea(self):
        DATA = "~/Downloads/datasets/chelsea/CHELSA_prec_01_V1.2_land.tif"  # from http://chelsa-climate.org/downloads/
        DATA = os.path.expanduser(DATA)
        x_ranges = list(gen_ranges(-180, 180, 1))
        y_ranges = list(gen_ranges(-90, 90, 1))
        data = gr.from_file(DATA)
        unmasked = data.raster.filled(0.0)
        # aggregated = aggregate_grid(data, x_ranges, y_ranges)
        # plt.imshow(aggregated)
        # plt.show()

    def test_br(self):
        DATA = "~/Downloads/datasets/elevation/viewfinder_dem3/15-J.tif"  # from http://www.viewfinderpanoramas.org/dem3.html
        DATA = os.path.expanduser(DATA)
        data = gr.from_file(DATA)
        aggregated = br_wrapper(data, 1, 1)
        print(aggregated)
        plt.imshow(aggregated.raster)
        plt.show()

    def test_load_agg(self):
        data = gr.from_file(os.path.expanduser("~/Downloads/datasets/elevation/one_deg.tif"))
        plt.imshow(data.raster)
        plt.show()




if __name__ == '__main__':
    unittest.main()


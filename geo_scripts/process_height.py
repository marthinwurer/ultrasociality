import os
import georasters as gr
import matplotlib.pyplot as plt

"""
Split into 1 degree grid. Get standard deviation, average elevation, and land %. 
"""

def round_to_nearest(val, nearest):
    return round(val * nearest) / nearest

# def index_to_lat_long()

def main():
    # DATA = "../data/relief_san_andres.tif"
    DATA = "../data/15-J.tif"

    data = gr.from_file(DATA)
    (xmin, xsize, x, ymax, y, ysize) = data.geot
    print(data.geot)
    NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(DATA)
    print(NDV, xsize, ysize, GeoT, DataType)
    print(Projection)

    # ok, when looking again it looks like the max and min are the edges + half the difference.
    # I might just try indexing for regions that are a multiple of the dimensions into the raster.

    # find top coords of grid cell in map for lat/long
    dx = 1.0  # in degrees
    dy = 1.0  # in degrees
    nw_corner = (round_to_nearest(xmin, dx), round_to_nearest(ymax, dy))
    print(nw_corner)
    se_corner = (round_to_nearest(nw_corner[0]+dx, dx), round_to_nearest(nw_corner[1]-dy, dy))
    print(se_corner)
    # data.plot()
    # plt.show()

    # get the array indexes for the map
    print(type(data.raster))
    print(data.raster.shape)
    print(data.raster)

    print(GeoT)
    x_indexes = int(dx / GeoT[1])
    y_indexes = int(dy / -GeoT[5])
    print(x_indexes, y_indexes)


    # determine the desired final raster size.

    # wait, i want to figure out how to divide this up to give each chunk its own list of data to take stats on
    # so I need to determine the next chunk. Or just iterate through the whole damn image and append the values to
    # a dict for that chunk
    # yeah let's do that, it's easy.

    # lol never mind just get the map pixels for the corners and iterate over them
    # col, row = gr.map_pixel(x,y,GeoT[1],GeoT[-1], GeoT[0],GeoT[3])
    # col, row = gr.map_pixel()
    print(data.map_pixel_location(13, 13))
    row, col = data.map_pixel_location(13,13)
    print(row, col)




if __name__ == "__main__":
    main()

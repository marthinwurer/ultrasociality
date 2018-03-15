import os
import georasters as gr
import matplotlib.pyplot as plt

"""
Split into 1 degree grid. Get standard deviation, average elevation, and land %. 
"""

def main():
    DATA = "/home/benjamin/Downloads/gis/"

    # Import raster
    raster = os.path.join(DATA, 'E020N40.DEM')
    data = gr.from_file(raster)
    (xmin, xsize, x, ymax, y, ysize) = data.geot
    data.plot()
    plt.show()
    NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(raster)
    print(NDV, xsize, ysize, GeoT, Projection, DataType)


if __name__ == "__main__":
    main()

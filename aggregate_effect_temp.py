"""
Calculates the Effective Temperature (Binford LR (2001)) using the chelsea dataset <http://chelsa-climate.org>.
"""
import matplotlib.pyplot as plt
import numpy as np
import geopandas
import os
import georasters as gr
import glob

from tqdm import tqdm

from geo_scripts.process_height import *


def calc_et(mwt, mct):
    """
    WARNING: modifies mwt and mct in-place for performance
    Args:
        mwt:
        mct:

    Returns:

    """
    lower = mwt - mct
    lower += 80
    mwt *= 18
    mct *= 10
    mwt -= mct
    mct = None
    effective_temperature = mwt / lower
    return effective_temperature


def main():
    # ET = ((18 * MWT) - (10 * MCM))/(MWM - MCM + 8)
    # MWT = Mean temperature of warmest month of year
    # MCT = Mean Temperature of coldest month of year

    # load each chelsea mean temperature file
    # reduce to the min and max of these files
    mct = np.full((20880, 43200), 2000, dtype=np.int16)
    mwt = np.full((20880, 43200), -5000, dtype=np.int16)
    x_ranges = list(gen_ranges(-180, 180, 1))
    y_ranges = list(gen_ranges(90, -90, 1))

    temps = []

    for file in tqdm(glob.glob(os.path.expanduser("~/Downloads/datasets/chelsea/CHELSA_temp10_*.tif"))):
        temp01 = gr.from_file(os.path.expanduser(file))
        #     temps.append(temp01)
        #     temp_ag = aggregate_grid(temp01, x_ranges, y_ranges)
        mwt = np.ma.maximum(mwt, temp01.raster)
        mct = np.ma.minimum(mct, temp01.raster)

    print("saving mwt and mct")
    np.save("./data/mean_warmest_temperature.npy", mwt.data)
    np.save("./data/mean_coldest_temperature.npy", mct.data)
    exit()

    # ET = ((18 * MWT) - (10 * MCM))/(MWM - MCM + 8)
    # do a bunch of jank to keep memory usage down
    temp01 = None
    lower = mwt - mct
    lower += 80
    mwt *= 18
    mct *= 10
    mwt -= mct
    mct = None
    effective_temperature = mwt / lower

    np.save("./data/effective_temperature_large.npy", effective_temperature.data)
    np.save("./data/effective_temperature_large_mask.npy", effective_temperature.mask)


if __name__ == '__main__':
    main()

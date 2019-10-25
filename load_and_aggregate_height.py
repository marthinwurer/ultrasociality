import glob
import os
import georasters as gr
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from geo_scripts.process_height import br_wrapper, clip_gr


def main():
    files = glob.glob(os.path.expanduser("~/Downloads/datasets/elevation/viewfinder_dem3/*.tif"))

    # files = files[:2]
    agg_first = True
    how = np.ma.std
    # how = np.ma.mean

    # fake a no data value
    ndv = -1000

    print("loading files")
    rasters = []
    for filename in tqdm(files):
        raster = gr.from_file(filename)
        if agg_first:
            raster.ndv = ndv
            # standard deviation goes to a double
            if how == np.ma.std:
                raster.datatype = "Float64"
            # print(raster.shape)
            if raster.shape[0] % 2 == 1:
                raster = clip_gr(raster)
            aggregated = br_wrapper(raster, 1, 1, how)
            rasters.append(aggregated)
        else:
            rasters.append(raster)

    # exit()

    print("starting merging")
    merged = gr.merge(rasters)
    print(merged.shape)

    print("starting aggregation")
    if not agg_first:
        one_deg = br_wrapper(merged, 1, 1, how)
        one_deg = clip_gr(one_deg)
    else:
        one_deg = merged

    print("Saving")
    one_deg.to_tiff(os.path.expanduser("~/Downloads/datasets/elevation/one_deg_stddev.tif"))





if __name__ == '__main__':
    main()
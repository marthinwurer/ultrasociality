import glob
import os
import georasters as gr
from tqdm import tqdm

from geo_scripts.process_height import br_wrapper


def main():
    files = glob.glob(os.path.expanduser("~/Downloads/datasets/elevation/viewfinder_dem3/*.tif"))

    print("loading files")
    rasters = []
    for filename in tqdm(files):
        raster = gr.from_file(filename)
        aggregated = br_wrapper(raster, 1, 1)
        rasters.append(aggregated)

    print("starting merging")
    merged = gr.merge(rasters)

    # print("starting aggregation")
    # one_deg = br_wrapper(merged, 1, 1)
    one_deg = merged

    print("Saving")
    one_deg.to_tiff(os.path.expanduser("~/Downloads/datasets/elevation/one_deg.tif"))





if __name__ == '__main__':
    main()
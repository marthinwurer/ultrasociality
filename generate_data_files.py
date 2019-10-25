
import matplotlib.pyplot as plt
import geopandas
import georasters as gr
import os

import numpy as np

from geo_scripts.process_height import gen_ranges, get_slices, aggregate_slices, filter_masked, get_global_raster, \
    rasterize_shapefile


def main():
    precip_data = gr.from_file(os.path.expanduser("~/Downloads/datasets/chelsea/CHELSA_bio10_12.tif"))

    # get the chelsea data mask to be used as the main ocean mask
    x_ranges = list(gen_ranges(-180, 180, 1))
    y_ranges = list(gen_ranges(90, -90, 1))
    slices = get_slices(precip_data, x_ranges, y_ranges)
    precip_ag = aggregate_slices(slices, filter_masked)
    chelsea_mask = precip_ag.mask

    # load shapefiles
    water_shape = geopandas.read_file(
        os.path.expanduser("~/Downloads/datasets/natural_earth/ne_50m_ocean/ne_50m_ocean.shp"))
    df_river = geopandas.read_file(os.path.expanduser(
        "~/Downloads/datasets/natural_earth/ne_50m_rivers_lake_centerlines_scale_rank/ne_50m_rivers_lake_centerlines_scale_rank.shp"))
    df_lake = geopandas.read_file(
        os.path.expanduser("~/Downloads/datasets/natural_earth/ne_50m_lakes/ne_50m_lakes.shp"))
    df_lake_historic = geopandas.read_file(
        os.path.expanduser("~/Downloads/datasets/natural_earth/ne_50m_lakes_historic/ne_50m_lakes_historic.shp"))

    # rasterize the shapefiles
    raster_1 = get_global_raster(1)
    ocean = rasterize_shapefile(water_shape, raster_1)

    lake_r = rasterize_shapefile(df_lake, raster_1)
    lake_h_r = rasterize_shapefile(df_lake_historic, raster_1)

    # clip out the caspian sea
    n = 41
    s = 54
    e = 234
    w = 226
    area_mask = np.zeros(lake_r.shape, dtype=np.bool)
    area_mask[n:s, w:e] = True
    sea_mask = ocean & area_mask

    # merge the rasters
    underwater = chelsea_mask | lake_r | lake_h_r | sea_mask

    # save the underwater mask
    np.save("./data/underwater_mask.npy", underwater)

    # generate the desert map
    desert = precip_ag < 250
    np.save("./data/desert.npy", desert.data)

    # generate river map
    river_rasterized = rasterize_shapefile(df_river, raster_1)
    np.save("./data/river.npy", river_rasterized)

    # generate elevation map
    height = gr.from_file(os.path.expanduser("~/Downloads/datasets/elevation/one_deg_height.tif")).raster
    np.save("./data/height.npy", height.data)
    stddev = gr.from_file(os.path.expanduser("~/Downloads/datasets/elevation/one_deg_stddev.tif")).raster
    np.save("./data/stddev.npy", stddev.data)

    # generate steppes map
    ecos_shape = geopandas.read_file(
        os.path.expanduser("~/Downloads/datasets/official_teow/official/wwf_terr_ecos.shp"))
    steppes = ecos_shape[ecos_shape["BIOME"] == 8]
    raster_1 = get_global_raster(1)
    steppes_rasterized = rasterize_shapefile(steppes, raster_1)
    np.save("./data/steppes.npy", steppes_rasterized)






if __name__ == '__main__':
    main()
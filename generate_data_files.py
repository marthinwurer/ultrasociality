
import matplotlib.pyplot as plt
import geopandas
import georasters as gr
import os

import numpy as np

from geo_scripts.process_height import gen_ranges, get_slices, aggregate_slices, filter_masked, get_global_raster, \
    rasterize_shapefile, aggregate_grid


def main():
    precip_data = gr.from_file(os.path.expanduser("~/Downloads/datasets/chelsea/CHELSA_bio10_12.tif"))

    print("ocean map")
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
    print("desert map")
    desert = precip_ag < 250
    np.save("./data/desert.npy", desert.data)

    # generate river map
    print("river map")
    river_rasterized = rasterize_shapefile(df_river, raster_1)
    np.save("./data/river.npy", river_rasterized)

    # generate elevation map
    print("elevation map")
    height = gr.from_file(os.path.expanduser("~/Downloads/datasets/elevation/one_deg_height.tif")).raster
    np.save("./data/height.npy", height.data)
    stddev = gr.from_file(os.path.expanduser("~/Downloads/datasets/elevation/one_deg_stddev.tif")).raster
    np.save("./data/stddev.npy", stddev.data)

    # generate steppes map
    print("Steppes map")
    ecos_shape = geopandas.read_file(
        os.path.expanduser("~/Downloads/datasets/official_teow/official/wwf_terr_ecos.shp"))
    steppes = ecos_shape[ecos_shape["BIOME"] == 8]
    raster_1 = get_global_raster(1)
    steppes_rasterized = rasterize_shapefile(steppes, raster_1)
    np.save("./data/steppes.npy", steppes_rasterized)

    # Efective temperatures?
    print("effective temperature")
    effective_temperature = np.load("./data/effective_temperature_large.npy")
    effective_mask = np.load("./data/effective_temperature_large_mask.npy")
    effective_temperature = np.ma.array(effective_temperature, mask=effective_mask)
    chelsea_geot = (-180.00013888885002, 0.0083333333, 0.0, 83.99986041515001, 0.0, -0.0083333333)
    raster = gr.GeoRaster(effective_temperature, chelsea_geot)
    et_ag = aggregate_grid(raster, x_ranges, y_ranges)
    np.save("./data/effective_temperature.npy", et_ag.data)
    # terrestrial plant threshold
    tpt = et_ag > 12.75
    np.save("./data/terrestrial_plant_threshold.npy", tpt.data)

    # agriculture thresholds
    print("agriculture")
    agriculture = tpt & (desert == False) | (river_rasterized & tpt)
    np.save("./data/agriculture.npy", agriculture.data)






if __name__ == '__main__':
    main()
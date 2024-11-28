import os

import cloudpickle

import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import dask.array as da
import dask
from dask.distributed import Client

from numcodecs import blosc
import zarr
from numcodecs import Zstd

keyfile = "service-account.json"

# basepath = "/home/akmoch/vrt_warp"
basepath = "/vsigs/geo-assets/soil_ml_source"

filelist = {'dem': 'dem_10m_cog.tif',
            'clay': 'eesti_clay1_10m_cog.tif',
            'drained': 'eesti_drainage_10m_cog.tif',
            'landuse': 'eesti_landuse_10m_cog.tif',
            'ndvi': 'eesti_ndvi_suvi2022_10m_cog.tif',
            'rock': 'eesti_rock1_10m_cog.tif',
            'sand': 'eesti_sand1_10m_cog.tif',
            'silt': 'eesti_silt1_10m_cog.tif',
            'lsf': 'lsfaktor_10m_cog.tif',
            'slope': 'slope_10m_cog.tif',
            'tri': 'tri_10m_cog.tif',
            'twi': 'twi_10m_cog.tif'
            }

models_path = "../spatial_features/models"

sys.path.append(models_path)

models = [
    'base_model.pkl',
    'xy_model.pkl',
    'knn_model.pkl',
    'bd_model.pkl',
    'ok_model.pkl',
    'gwr_model.pkl'
]


def read_wkey(fname, chunk_size):
    raster = rxr.open_rasterio(fname, chunks=(1, chunk_size, chunk_size))
    return raster


def read_pickles(modelname):
    with open(os.path.join(models_path, modelname), "rb") as lp:
        predictor = cloudpickle.load(lp)
        return predictor


def recode_landuse(code):
    if (1 <= code <= 18) or (code in [43, 45, 47]):
        return 'grassland'
    if (19 <= code <= 25) or (code == 49):
        return 'wetland'
    if (26 <= code <= 37) or (code == 46): 
        return 'forest'
    if (38 <= code <= 42):
        return 'arable'
    if (code == 44 or code == 48):
        return 'artificial'
    else:
        return np.nan

def _run_predict_base(slope_data,
                      twi_data,
                      tri_data,
                      lsf_data,
                      ndvi_data,
                      clay_data,
                      sand_data,
                      silt_data,
                      rock_data,
                      drained_data,
                      landuse_data,
                      estimator,
                      chunk_size,
                      origin,
                      block_info=None
                    ):

    temp_ds = xr.Dataset()

    chunk_size0 = slope_data.shape[0]
    chunk_size1 = slope_data.shape[1]

    y_start = origin["y"] + ( block_info[0]['chunk-location'][0] * chunk_size * 10 )
    x_start = origin["x"] + ( block_info[0]['chunk-location'][1] * chunk_size * 10 )

    y_end = y_start + (chunk_size0 * 10)
    x_end = x_start + (chunk_size1 * 10)

    y_coords = np.arange(y_start, y_end, 10 )
    x_coords = np.arange(x_start, x_end, 10 )
    
    coords = {'y': y_coords, 'x': x_coords }
    
    temp_ds["slope"] = xr.DataArray(slope_data, dims=("y", "x"), coords=coords)
    temp_ds["twi"] = xr.DataArray(twi_data, dims=("y", "x"), coords=coords)
    temp_ds["tri"]= xr.DataArray(tri_data, dims=("y", "x"), coords=coords)
    temp_ds["lsf"] = xr.DataArray(lsf_data, dims=("y", "x"), coords=coords)
    temp_ds["ndvi"] = xr.DataArray(ndvi_data, dims=("y", "x"), coords=coords)
    temp_ds["clay"] = xr.DataArray(clay_data, dims=("y", "x"), coords=coords)
    temp_ds["sand"] = xr.DataArray(sand_data, dims=("y", "x"), coords=coords)
    temp_ds["silt"] = xr.DataArray(silt_data, dims=("y", "x"), coords=coords)
    temp_ds["rock"] = xr.DataArray(rock_data, dims=("y", "x"), coords=coords)
    temp_ds["drained"] = xr.DataArray(drained_data, dims=("y", "x"), coords=coords)
    temp_ds["landuse"] = xr.DataArray(landuse_data, dims=("y", "x"), coords=coords)

    ddf = temp_ds.to_dataframe()

    ddf["landuse_type"] = ddf["landuse"].apply(recode_landuse)
    ddf["y_coord"] = np.array([t[0] for t in ddf.index])
    ddf["x_coord"] = np.array([t[1] for t in ddf.index])

    keeps = ddf.isna().apply(np.any, axis=1)
    kept_index = ddf.loc[keeps].index
        
    ddf = ddf.drop(columns=["landuse"]).dropna()

    if len(ddf.index) <= 0:
        arr = np.empty((chunk_size0, chunk_size1), dtype=np.float32)
        arr.fill(np.nan)
        return arr

    ddf_X = pd.get_dummies(ddf)

    required_dumms = ['landuse_type_arable',
       'landuse_type_artificial', 'landuse_type_forest',
       'landuse_type_grassland', 'landuse_type_wetland']
    
    for c in required_dumms:
        if not c in ddf_X.columns:
            ddf_X[c] = 0
    
    ddf_X = gpd.GeoDataFrame(ddf_X, crs=3301, geometry=gpd.points_from_xy(ddf_X.y_coord, ddf_X.x_coord)).drop(columns=["y_coord", "x_coord"])

    ddf_X = ddf_X[['slope', 'twi', 'tri', 'lsf', 'ndvi', 'clay', 'sand', 'silt',
       'rock', 'drained', 'geometry', 'landuse_type_arable',
       'landuse_type_artificial', 'landuse_type_forest',
       'landuse_type_grassland', 'landuse_type_wetland']]

    soc = estimator.predict(ddf_X)
    # ddf_X["soc"] = np.exp(soc)
    ddf_X["soc"] = soc
    
    soc_Y = pd.concat([ddf_X[["soc"]], pd.DataFrame({"soc": np.full(len(kept_index), np.nan)}, index=kept_index)])
   
    # 'chunk-location': (4,)
    ct = block_info[0]['chunk-location'][0]
    print(f"I was here: chunk [{ct}]")
    out = soc_Y.to_xarray()["soc"].data

    return out


def do_stuff(client):
    
    chunk_size = 128
    ds = xr.Dataset()

    for k, f in filelist.items():
        r = read_wkey(os.path.join(basepath, f), chunk_size)
        # ds[k] = r
        ds[k] = r.squeeze(dim="band", drop=True)
        # print(r)

    print(ds)

    only_base_model = models[5]
    origin = {"y": ds.slope.coords.get("y").values[0], "x": ds.slope.coords.get("x").values[0]}
    estimator = read_pickles(only_base_model)
    scatter_model = client.scatter(estimator)

    out = da.map_blocks(_run_predict_base,
                    ds.slope.data.astype('f4'),
                    ds.twi.data.astype('f4'),
                    ds.tri.data.astype('f4'),
                    ds.lsf.data.astype('f4'),
                    ds.ndvi.data.astype('f4'),
                    ds.clay.data.astype('f4'),
                    ds.sand.data.astype('f4'),
                    ds.silt.data.astype('f4'),
                    ds.rock.data.astype('f4'),
                    ds.drained.data.astype('i4'),
                    ds.landuse.data.astype('i8'),
                    scatter_model,
                    chunk_size,
                    origin=origin,
                    meta=np.array( [[0, 0]], dtype=np.float32 )
                   )
    
    arr_out = xr.DataArray(out,
                    name="soc",
                    coords=ds.drained.coords,
                    dims=ds.drained.dims,
                    attrs=ds.drained.attrs)

    arr_out = arr_out.fillna(-9999)
    arr_out = arr_out.rio.write_nodata(-9999)
    
    ds_out = xr.Dataset()
    ds_out["soc"] = arr_out
    
    # ds_out = ds_out.rio.write_crs("epsg:3301")
    # ds_out.rio.to_raster("/gpfs/space/home/kmoch/soilml2/soc_output_base.tif", tiled=True, windowed=True, compress='LZW', dtype="float32", compute=True)
    # compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    compressor = Zstd(level=1)
    the_compute = ds_out.to_zarr("/gpfs/space/home/kmoch/soilml2/soc_output_gwr_v2.zarr", encoding={"soc": {"compressor": compressor}}, compute=False)
    the_compute.compute()

    
if __name__ == "__main__":

    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = keyfile
    else:
        keyfile = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] 

    # Connect to the Dask cluster
    # dask cluster was already initialised through the job script, sync through the shared file
    client = Client(scheduler_file='/gpfs/space/home/kmoch/soilml2/scheduler_gwr.json')

    do_stuff(client)

    client.close()


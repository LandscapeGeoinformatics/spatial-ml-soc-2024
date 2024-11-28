import os

import cloudpickle

# import rasterio
import xarray as xr
import rioxarray as rxr
import numpy as np
import dask
import dask
from dask.distributed import Client, LocalCluster

import sys
sys.path.append('..')
from scripts.settings import *

basepath = PROCESSED_DIR

# soc: Real (0.0)
# landuse_type: String (0.0)
# slope: Real (0.0)
# twi: Real (0.0)
# tri: Real (0.0)
# lsf: Real (0.0)
# ndvi: Real (0.0)
# clay: Real (0.0)
# sand: Real (0.0)
# silt: Real (0.0)
# rock: Real (0.0)
# drained: Integer64

filelist = {#'dem': 'dem_10m_cog.tif',
            'clay': 'eesti_clay1_10m.tif',
            'drained': 'eesti_drainage_10m.tif',
            'landuse': 'eesti_landuse_10m.vrt',
            'ndvi': 'eesti_ndvi_suvi2022_10m.vrt',
            'rock': 'eesti_rock1_10m.tif',
            'sand': 'eesti_sand1_10m.tif',
            'silt': 'eesti_silt1_10m.tif',
            'lsf': 'eesti_lsf_10m.vrt',
            'slope': 'eesti_slope_10m.vrt',
            'tri': 'eesti_tri_10m.vrt',
            'twi': 'eesti_twi_10m.vrt'
            }




def read_wkey(fname):
    raster = rxr.open_rasterio(fname, chunks=(1, 1024, 1024))
    return raster

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


def do_stuff(client):
    
    ds = xr.Dataset()

    for k, f in filelist.items():
        r = read_wkey(os.path.join(basepath, f))
        ds[k] = r
        # print(r)

    print(ds)

if __name__ == "__main__":
    # Start a local Dask cluster with 4 workers and 2 threads per worker
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    print(cluster.dashboard_link)

    # Connect to the Dask cluster
    client = Client(cluster)

    do_stuff(client)

    client.close()
    cluster.close()


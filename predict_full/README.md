# Predict with ML models for all of Estonia

Alexander Kmoch: alexander.kmoch@ut.ee

The setup with Dask and Xarray is slightly different across versions, also chunk_size varies between 512 and 1024. Also, the way the sklearn model pickle is loaded and shared/propagated to the prediction routine is different. Sometimes as in-memory reference, as dask scattered (broadcasting), and as pickle-re-un-pickle to create local copies in each dask worker was tried.

Access to GeoTiffs in cloud bucket is also only shown in later versions (knn and bd).

Rioxarray is used to load the input/covariate rasters into an Xarray Dataset, which works nicely.

The dask map_blocks function is supposed to send blocks to the workers that could be processed independently and in parallel.

The full inference was trialed and run on a Cloud VM and on Tartu HPC.
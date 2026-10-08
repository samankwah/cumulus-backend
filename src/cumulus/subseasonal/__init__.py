"""Sub-seasonal (46-day) daily rainfall from downscaled ECMWF IFS runs.

``sources`` fetches raw per-lead NetCDF4 files (local folder or Azure Blob via SAS),
``ingest`` validates them and writes a compact Ghana-only NetCDF3 artifact,
``geomask`` rasterises the district polygons used for masking and area means,
``metrics`` holds the rainfall and spell arithmetic shared by ingest, service and tests.
"""

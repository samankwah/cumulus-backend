Cumulus sub-seasonal rainfall forecasts: sample run, init 2026-10-09 00 UTC
==========================================================================

Every model in this bundle was downscaled from the same initialisation date, so the outputs
can be compared directly.

Folders
-------
ifs/bilinear/                     IFS global precip, bilinear-interpolated to 0.1 deg (baseline, no ML)
ifs/unet/                         IFS -> UNet (checkpoint unet_africa-050000)
ifs/diffusion_members/            IFS -> diffusion UNet, 10 ensemble members (member_000 ... member_009)
aurora/era5/                      Aurora with the ERA5-calibrated precip head
    bilinear/  unet/  diffusion_members/
aurora/imerg/                     Aurora with the IMERG-calibrated precip head
    bilinear/  unet/  diffusion_members/

Diffusion model:  model_diffusion_unet_precip_input_output_residual_shiftedlog_calibrated_silu_selfcond_precip24h_only
                  checkpoint ckpt-diffusion-722-precip-fs64-lr1e-4-100k-100000 (heun, 20 steps, cfg 3, 10 members)

Files
-----
One NetCDF file per lead day, 46 per folder (46 per member folder for diffusion):
    YYYY-MM-DD-HH-LLLL.nc   (init date, init hour, lead time in hours: 0024 ... 1104)

Each file holds:
    precip_24h (time=1, latitude=371, longitude=771)  float32, mm accumulated over 24 h
    coordinates: time (valid day), lead_time, init_time, latitude, longitude
    grid: 0.1 deg, latitude 26.0 -> -11.0 (descending), longitude -20.0 -> 57.0
    domain covers East Africa, Ghana, Kenya, Senegal and West Africa

Notes
-----
- Diffusion members are independent samples. Their mean gives the ensemble-mean forecast, and
  their spread shows uncertainty (for example, the fraction of members above 1 mm gives a rain probability).
- UNet output can be slightly negative (around -0.3 mm). Clip at 0 before use.
- Lead 0024 is the 24 h ending at init + 1 day. For this run, time = 2026-10-10.
- The files carry no NetCDF attributes. Units are mm.

Quick look (Python):
    import xarray as xr
    ds = xr.open_mfdataset("ifs/unet/*.nc", combine="by_coords")
    ghana = ds.precip_24h.sel(latitude=slice(11.5, 4.5), longitude=slice(-3.5, 1.5)).clip(min=0)

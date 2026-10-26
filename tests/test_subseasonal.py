from __future__ import annotations

from datetime import date
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from cumulus.subseasonal import legends, metrics
from cumulus.subseasonal.ingest import IngestError, ingest_run, load_raw_run
from cumulus.subseasonal.sources import AzureBlobSasSource, LocalFolderSource, parse_lead_file, parse_model_path
from cumulus.settings import get_settings

pytest.importorskip("h5netcdf")
pytest.importorskip("h5py")

INIT = pd.Timestamp("2026-09-25T00:00:00")
# Wider than the Ghana window and stored north-to-south like upstream files.
LATITUDES = np.round(np.arange(13.0, 2.95, -0.1), 4).astype(np.float32)
LONGITUDES = np.round(np.arange(-5.0, 3.05, 0.1), 4).astype(np.float32)


def _daily_value(day: int) -> float:
    # Days 1-3 wet (5 mm), 4-9 dry (0.2 mm), 10+ alternating 3 mm / 0 mm.
    if day <= 3:
        return 5.0
    if day <= 9:
        return 0.2
    return 3.0 if day % 2 == 0 else 0.0


def _write_lead_file(
    run_dir: Path,
    lead_day: int,
    *,
    init: pd.Timestamp = INIT,
    variable: str = "precip_24h",
    value: float | None = None,
) -> Path:
    field = np.full((1, LATITUDES.size, LONGITUDES.size), _daily_value(lead_day) if value is None else value, dtype=np.float32)
    field[0, 0, 0] = -0.01  # UNet artefact, must be clipped
    dataset = xr.Dataset(
        {variable: (("time", "latitude", "longitude"), field)},
        coords={
            "time": [init + pd.Timedelta(days=lead_day)],
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
            "init_time": init,
            "lead_time": ("time", [pd.Timedelta(days=lead_day)]),
        },
    )
    path = run_dir / f"{init:%Y-%m-%d}-00-{lead_day * 24:04d}.nc"
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


def _make_run(root: Path, days: int = 14, *, init: pd.Timestamp = INIT, skip: set[int] | None = None) -> Path:
    run_dir = root / f"{init:%Y-%m-%d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for day in range(1, days + 1):
        if skip and day in skip:
            continue
        _write_lead_file(run_dir, day, init=init)
    return run_dir


@pytest.fixture()
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__ARCHIVE_DIR", str(tmp_path / "archive"))
    # Building per-area summaries adds seconds to every ingest; the partner API tests turn it back on.
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__ARCHIVE_SUMMARIES", "false")
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__RAW_CACHE_DIR", str(tmp_path / "raw-cache"))
    # The tests were written against IFS-UNet as the default model; the shipped default is checked
    # separately (test_shipped_default_model_is_aurora_imerg_unet).
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__SOURCE_ID", "ifs_unet")
    # Observed rain lives in its own temporary folder, so no real or demo observations leak in.
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__OBSERVED__ARTIFACT_DIR", str(tmp_path / "observed"))
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__OBSERVED__RAW_CACHE_DIR", str(tmp_path / "observed-cache"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


# --------------------------------------------------------------------------- metrics


def test_run_lengths_and_spell_days_handle_edges():
    condition = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert metrics.run_lengths(condition).tolist() == [2, 2, 0, 3, 3, 3, 0, 0, 1]
    assert metrics.spell_day_mask(condition, 3).tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    grid = np.stack([condition, ~condition], axis=1)  # (day, 2) works along axis 0
    assert metrics.run_lengths(grid)[:, 1].tolist() == [0, 0, 1, 0, 0, 0, 2, 2, 0]


def test_find_spells_flags_open_edges_and_thresholds():
    series = np.array([0, 0, 0, 0, 0, 2, 2, 2, 0.5, 0.9, 3, 0, 0, 0, 0, 0], dtype=float)
    spells = metrics.find_spells(series, wet_threshold_mm=1.0, dry_spell_min_days=5, wet_spell_min_days=3)
    assert [(s.kind, s.start_day, s.end_day, s.open_start, s.open_end) for s in spells] == [
        ("dry", 1, 5, True, False),
        ("wet", 6, 8, False, False),
        ("dry", 12, 16, False, True),
    ]
    outlook = metrics.outlook_metrics(series, wet_threshold_mm=1.0, dry_spell_min_days=5, wet_spell_min_days=3)
    assert float(outlook["total"]) == pytest.approx(10.4)
    assert int(outlook["rainy_days"]) == 4
    assert int(outlook["dry_spell_days"]) == 10
    assert int(outlook["wet_spell_days"]) == 3


def test_weekly_windows_mark_trailing_partial_week():
    windows = metrics.weekly_windows(46)
    assert len(windows) == 7
    assert (windows[0].start_day, windows[0].end_day, windows[0].partial) == (1, 7, False)
    assert (windows[-1].start_day, windows[-1].end_day, windows[-1].days, windows[-1].partial) == (43, 46, 4, True)
    sums = metrics.window_sums(np.ones((46, 2)), windows)
    assert sums[:, 0].tolist() == [7, 7, 7, 7, 7, 7, 4]


def test_rain_day_is_the_accumulation_start_date():
    assert metrics.rain_day(date(2026, 9, 25), 1) == date(2026, 9, 25)
    assert metrics.rain_day(date(2026, 9, 25), 46) == date(2026, 11, 9)


ONSET_RULE = dict(threshold_mm=20.0, window_days=3, guard_days=30, guard_max_dry_days=10, dry_threshold_mm=1.0)


def _onset_day(series):
    return int(metrics.onset(np.asarray(series, dtype=float), **ONSET_RULE).day)


def test_onset_needs_20_mm_within_3_days_and_no_long_dry_spell_after():
    showers = [2.0, 0.0] * 23  # a rain day every other day: never 20 mm in 3 days
    assert _onset_day(showers) == 0

    one_day = list(showers)
    one_day[4] = 25.0  # a single 25 mm day counts ("within at most 3 days")
    assert _onset_day(one_day) == 3  # the 2 mm shower on day 3 opens a 27 mm burst
    one_day[2] = 0.0
    assert _onset_day(one_day) == 5  # never dated to a dry day

    false_start = [0.0] * 46
    false_start[2:5] = [8.0, 8.0, 8.0]  # 24 mm, then 11 dry days: rejected
    false_start[16:19] = [10.0, 10.0, 10.0]  # second burst, followed by rain every 5 days
    for day in range(22, 46, 5):
        false_start[day] = 4.0
    assert _onset_day(false_start) == 17  # first day of the second burst
    result = metrics.onset(np.asarray(false_start), **ONSET_RULE)
    assert int(result.guard_days) == 30 and int(result.longest_dry_after) <= 10


def test_onset_near_the_end_is_provisional_and_grids_work_per_cell():
    late = [0.0, 2.0] * 23
    late[40:42] = [12.0, 12.0]
    result = metrics.onset(np.asarray(late), **ONSET_RULE)
    assert int(result.day) == 40 and int(result.guard_days) == 7  # days 40-46 only
    grid = np.stack([np.asarray(late), np.full(46, np.nan)], axis=1)
    days = metrics.onset(grid, **ONSET_RULE).day
    assert days[0] == 40 and np.isnan(days[1])


def test_onset_legend_labels_weeks_with_dates():
    legend = legends.onset_legend(date(2026, 9, 25), 46)
    labels = [item.label for item in legend.bins]
    assert labels[0] == "No onset" and labels[1] == "25 Sep–1 Oct" and labels[-1] == "6 Nov–9 Nov"
    assert legends.classify(np.array([0.0, 1.0, 7.0, 8.0, 46.0, np.nan]), legend).tolist() == [0, 1, 1, 2, 7, -1]


def test_legend_classification_is_fixed_and_transparent_below_daily_minimum():
    values = np.array([np.nan, 0.4, 1.0, 4.9, 150.0])
    assert legends.classify(values, legends.RAIN_DAILY).tolist() == [-1, -1, 0, 1, 8]
    assert legends.classify(np.array([0.0, 4.9, 5.0, 999.0]), legends.RAIN_WEEKLY).tolist() == [0, 0, 1, 8]
    bins = legends.legend_payload(legends.RAIN_TOTAL)["bins"]
    assert bins[0]["label"] == "<25" and bins[-1]["max"] is None and bins[-1]["label"] == "400+"


# --------------------------------------------------------------------------- sources


def test_parse_lead_file_and_local_source(tmp_path):
    parsed = parse_lead_file("2026-09-25-00-1104.nc")
    assert parsed is not None and parsed.lead_hours == 1104 and parsed.init_date == date(2026, 9, 25)
    assert parse_lead_file("notes.txt") is None
    _make_run(tmp_path, days=3)
    (tmp_path / "not-a-run").mkdir()
    source = LocalFolderSource(tmp_path)
    assert source.list_runs() == [date(2026, 9, 25)]
    assert [item.lead_hours for item in source.list_lead_files(date(2026, 9, 25))] == [24, 48, 72]


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_azure_source_lists_with_pagination_and_downloads(monkeypatch, tmp_path):
    requested: list[str] = []
    page_one = b"""<?xml version="1.0"?><EnumerationResults><Blobs>
        <BlobPrefix><Name>Unet/2026-09-24/</Name></BlobPrefix></Blobs><NextMarker>m1</NextMarker></EnumerationResults>"""
    page_two = b"""<?xml version="1.0"?><EnumerationResults><Blobs>
        <BlobPrefix><Name>Unet/2026-09-25/</Name></BlobPrefix><BlobPrefix><Name>Unet/junk/</Name></BlobPrefix></Blobs>
        <NextMarker /></EnumerationResults>"""
    files = b"""<?xml version="1.0"?><EnumerationResults><Blobs>
        <Blob><Name>Unet/2026-09-25/2026-09-25-00-0048.nc</Name><Properties><Content-Length>4</Content-Length></Properties></Blob>
        <Blob><Name>Unet/2026-09-25/2026-09-25-00-0024.nc</Name><Properties><Content-Length>4</Content-Length></Properties></Blob>
        <Blob><Name>Unet/2026-09-25/readme.txt</Name><Properties><Content-Length>9</Content-Length></Properties></Blob>
        </Blobs><NextMarker /></EnumerationResults>"""

    def fake_urlopen(request, timeout):
        url = request.full_url
        requested.append(url)
        assert "sig=secret" in url
        if "comp=list" in url:
            if "prefix=Unet%2F2026-09-25%2F" in url:
                return _FakeResponse(files)
            return _FakeResponse(page_two if "marker=m1" in url else page_one)
        return _FakeResponse(b"data")

    monkeypatch.setattr("cumulus.subseasonal.sources.urlopen", fake_urlopen)
    source = AzureBlobSasSource("https://acct.blob.core.windows.net/forecasts?sv=1&sig=secret", prefix="Unet")
    assert "secret" not in source.label
    assert source.list_runs() == [date(2026, 9, 24), date(2026, 9, 25)]
    paths = source.fetch_run(date(2026, 9, 25), tmp_path)
    assert [path.name for path in paths] == ["2026-09-25-00-0024.nc", "2026-09-25-00-0048.nc"]
    assert all(path.read_bytes() == b"data" for path in paths)
    downloads = [url for url in requested if "comp=list" not in url]
    # A second fetch reuses the cached files (same size) instead of downloading again.
    source.fetch_run(date(2026, 9, 25), tmp_path)
    assert len([url for url in requested if "comp=list" not in url]) == len(downloads) == 2


def test_azure_source_rejects_non_sas_urls():
    with pytest.raises(Exception, match="SAS URL"):
        AzureBlobSasSource("https://acct.blob.core.windows.net/forecasts")


# --------------------------------------------------------------------------- ingest


def test_ingest_clips_subsets_and_writes_netcdf3_artifact(settings, tmp_path):
    _make_run(tmp_path / "src", days=14)
    result = ingest_run(settings, LocalFolderSource(tmp_path / "src"))
    assert result.run_id == "ifs_unet_2026092500" and result.lead_days == 14 and result.promoted
    assert result.artifact_path.read_bytes()[:3] == b"CDF"

    with xr.open_dataset(result.artifact_path, engine="scipy") as dataset:
        precip = dataset["precip"].load()
        assert precip.dims == ("lead_day", "latitude", "longitude")
        assert float(dataset["latitude"].min()) == pytest.approx(4.0)
        assert float(dataset["latitude"].max()) == pytest.approx(12.0)
        assert float(dataset["longitude"].min()) == pytest.approx(-4.0)
        assert bool(np.all(np.diff(dataset["latitude"].values) > 0))
        assert float(precip.min()) >= 0.0
        assert float(precip.isel(lead_day=0).max()) == pytest.approx(5.0)
        assert float(precip.isel(lead_day=4).mean()) == pytest.approx(0.2)

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["lead_days"] == list(range(1, 15))
    assert manifest["first_rain_day"] == "2026-09-25" and manifest["last_rain_day"] == "2026-10-08"
    assert manifest["raw_variable"] == "precip_24h"
    active = json.loads((Path(settings.subseasonal.artifact_dir) / "ifs_unet" / "active.json").read_text())
    assert active["run_id"] == result.run_id


def test_parse_model_path_reads_model_checkpoint_and_sampler():
    found = parse_model_path(
        "Unet/model_diffusion_unet_precip_input_output_residual_shiftedlog_calibrated_silu_selfcond_precip24h_only/"
        "ckpt-diffusion-722-precip-fs64-lr1e-4-100k-100000_heun_n20_cfg3_ens10/"
    )
    assert found["model_type"] == "diffusion_unet"
    assert found["checkpoint"] == "ckpt-diffusion-722-precip-fs64-lr1e-4-100k-100000"
    assert found["checkpoint_step"] == 100000
    assert (found["sampler"], found["sampler_steps"], found["guidance_scale"], found["ensemble_size"]) == ("heun", 20, 3, 10)
    # The live layout puts the run date above the model folders; cfg 2 from init 2026-10-21.
    live = parse_model_path(
        "live_forecasts/downscaled_models/ifs/east_africa-ghana-kenya-senegal-west_africa/2026-10-15/"
        "model_diffusion_unet_precip_input_output_residual_shiftedlog_calibrated_silu_selfcond_precip24h_only/"
        "ckpt-diffusion-722-precip-fs64-lr1e-4-100k-100000_heun_n20_cfg2_ens10"
    )
    assert (live["sampler"], live["sampler_steps"], live["guidance_scale"], live["ensemble_size"]) == ("heun", 20, 2, 10)
    assert parse_model_path("Unet/") == {}
    assert parse_model_path(r"C:\data\model_det_unet\Unet")["model_type"] == "det_unet"


def test_ingest_records_model_provenance(settings, tmp_path):
    root = tmp_path / "model_diffusion_unet_x" / "ckpt-d-722-100000_heun_n20_cfg3_ens10"
    _make_run(root, days=7)
    result = ingest_run(settings, LocalFolderSource(root))
    provenance = json.loads(result.manifest_path.read_text())["provenance"]
    assert provenance["model_type"] == "diffusion_unet" and provenance["sampler"] == "heun"
    assert provenance["checkpoint"] == "ckpt-d-722-100000" and provenance["ensemble_size"] == 10
    with xr.open_dataset(result.artifact_path, engine="scipy") as dataset:
        assert dataset.attrs["checkpoint"] == "ckpt-d-722-100000"


def test_ingest_accepts_variable_aliases(settings, tmp_path):
    run_dir = tmp_path / "src" / "2026-09-25"
    run_dir.mkdir(parents=True)
    paths = [_write_lead_file(run_dir, day, variable="tp") for day in range(1, 8)]
    dataset, details = load_raw_run(paths, settings)
    assert details["variable"] == "tp" and dataset.sizes["lead_day"] == 7


@pytest.mark.parametrize(
    ("builder", "message"),
    [
        (lambda root: _make_run(root, days=10, skip={4}), "contiguous"),
        (lambda root: _make_run(root, days=3), "at least 7"),
    ],
)
def test_ingest_rejects_gaps_and_short_runs(settings, tmp_path, builder, message):
    builder(tmp_path / "src")
    with pytest.raises(IngestError, match=message):
        ingest_run(settings, LocalFolderSource(tmp_path / "src"))


def test_ingest_rejects_mixed_init_times_and_unknown_variables(settings, tmp_path):
    run_dir = _make_run(tmp_path / "src", days=7)
    _write_lead_file(run_dir, 8, init=INIT + pd.Timedelta(days=1)).rename(run_dir / "2026-09-25-00-0192.nc")
    with pytest.raises(IngestError, match="mix init times"):
        ingest_run(settings, LocalFolderSource(tmp_path / "src"))

    other = tmp_path / "other" / "2026-09-25"
    other.mkdir(parents=True)
    paths = [_write_lead_file(other, day, variable="t2m") for day in range(1, 8)]
    with pytest.raises(IngestError, match="none of"):
        load_raw_run(paths, settings)


def test_retention_keeps_newest_runs_and_active_pointer(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(settings.subseasonal, "retention_runs", 2)
    for offset in range(3):
        _make_run(tmp_path / "src", days=7, init=INIT + pd.Timedelta(days=offset))
    source = LocalFolderSource(tmp_path / "src")
    for offset in range(3):
        ingest_run(settings, source, (INIT + pd.Timedelta(days=offset)).date())
    root = Path(settings.subseasonal.artifact_dir) / "ifs_unet"
    assert sorted(path.name for path in root.iterdir() if path.is_dir()) == ["ifs_unet_2026092600", "ifs_unet_2026092700"]
    # Re-ingesting an older run must not demote the newer active run.
    ingest_run(settings, source, INIT.date())
    assert json.loads((root / "active.json").read_text())["run_id"] == "ifs_unet_2026092700"


def test_reingesting_the_same_run_changes_no_files(settings, tmp_path):
    _make_run(tmp_path / "src", days=7)
    source = LocalFolderSource(tmp_path / "src")
    first = ingest_run(settings, source)
    root = Path(settings.subseasonal.artifact_dir) / "ifs_unet"
    snapshot = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    again = ingest_run(settings, source)
    assert again.unchanged and not again.promoted and again.run_id == first.run_id and again.lead_days == 7
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == snapshot

    # A new lead file upstream means the run changed and is rebuilt; --force rebuilds regardless.
    _write_lead_file(tmp_path / "src" / "2026-09-25", 8)
    grown = ingest_run(settings, source)
    assert not grown.unchanged and grown.lead_days == 8
    assert not ingest_run(settings, source, force=True).unchanged


def test_require_complete_falls_back_to_the_newest_complete_run(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(settings.subseasonal, "expected_lead_days", 10)
    _make_run(tmp_path / "src", days=10)
    _make_run(tmp_path / "src", days=8, init=INIT + pd.Timedelta(days=1))  # still uploading
    source = LocalFolderSource(tmp_path / "src")

    result = ingest_run(settings, source, require_complete=True)
    assert result.run_id == "ifs_unet_2026092500" and result.lead_days == 10
    with pytest.raises(IngestError, match="incomplete"):
        ingest_run(settings, source, (INIT + pd.Timedelta(days=1)).date(), require_complete=True)
    # Without the guard the partial run is still accepted (manual ingests).
    assert ingest_run(settings, source).run_id == "ifs_unet_2026092600"


def test_require_complete_without_any_complete_run_fails(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(settings.subseasonal, "expected_lead_days", 46)
    _make_run(tmp_path / "src", days=7)
    with pytest.raises(IngestError, match="No complete run"):
        ingest_run(settings, LocalFolderSource(tmp_path / "src"), require_complete=True)


# --------------------------------------------------------------------------- API


@pytest.fixture()
def client(settings, tmp_path):
    _make_run(tmp_path / "src", days=14)
    ingest_run(settings, LocalFolderSource(tmp_path / "src"))
    from cumulus.main import app

    return TestClient(app)


def test_runs_endpoint_summarises_active_run(client):
    response = client.get("/subseasonal/runs")
    assert response.status_code == 200
    payload = response.json()
    run = payload["runs"][0]
    assert payload["active_run_id"] == run["run_id"] == "ifs_unet_2026092500"
    assert run["lead_days"] == 14 and run["missing_lead_days"] == list(range(15, 47))
    assert run["first_day"] == "2026-09-25" and run["days"][0]["value"] == pytest.approx(5.0)
    assert [week["days"] for week in run["weeks"]] == [7, 7]
    assert {layer["layer"] for layer in run["layers"]} == {"rainfall", "rainy_days", "dry_spell_days", "wet_spell_days", "extreme_rain", "onset"}
    assert run["thresholds"]["onset_mm"] == 20 and run["thresholds"]["onset_guard_days"] == 30


def test_layer_and_tile_endpoints(client):
    response = client.get("/subseasonal/layer", params={"layer": "rainfall", "aggregation": "weekly", "index": 1})
    assert response.status_code == 200
    layer = response.json()
    assert layer["start_date"] == "2026-09-25" and layer["end_date"] == "2026-10-01"
    assert layer["stats"]["mean"] == pytest.approx(15.8, abs=0.05)  # days 1-3 at 5 mm + days 4-7 at 0.2 mm
    assert layer["legend"]["unit"] == "mm" and "run_id=ifs_unet_2026092500" in layer["tile_url"]

    tile_url = layer["tile_url"].replace("{z}", "7").replace("{x}", "63").replace("{y}", "61")
    tile = client.get(tile_url)
    assert tile.status_code == 200 and tile.content.startswith(b"\x89PNG")
    assert "immutable" in tile.headers["cache-control"]
    ocean = client.get(layer["tile_url"].replace("{z}", "7").replace("{x}", "10").replace("{y}", "10"))
    assert ocean.status_code == 200 and len(ocean.content) < len(tile.content)


def test_outlook_layers_and_area_values(client):
    dry = client.get("/subseasonal/layer", params={"layer": "dry_spell_days", "stat": "days"}).json()
    # Days 4-9 are a 6-day dry run (>= 5) -> 6 dry-spell days everywhere.
    assert dry["stats"]["min"] == dry["stats"]["max"] == 6
    # By default a whole run answers with the chance of such a spell: certain, for a single run.
    chance = client.get("/subseasonal/layer", params={"layer": "dry_spell_days"}).json()
    assert chance["stats"]["min"] == chance["stats"]["max"] == 100 and chance["legend"]["unit"] == "%"
    values = client.get(
        "/subseasonal/area-values", params={"level": "district", "layer": "rainfall", "aggregation": "daily", "index": 1}
    ).json()["values"]
    assert len(values) >= 259 and values["Tamale"] == pytest.approx(5.0)


def test_indicator_layers_support_daily_and_weekly_periods(client):
    layers = {item["layer"]: item for item in client.get("/subseasonal/runs").json()["runs"][0]["layers"]}
    assert set(layers["dry_spell_days"]["aggregations"]) == {"daily", "weekly", "total"}

    def stats(layer, aggregation, index=None):
        params = {"layer": layer, "aggregation": aggregation, "stat": "days"}
        if index is not None:
            params["index"] = index
        response = client.get("/subseasonal/layer", params=params)
        assert response.status_code == 200
        return response.json()

    # Dry run on days 4-9: week 1 holds days 4-7, week 2 days 8-9; the weeks add up to the total.
    week1, week2 = stats("dry_spell_days", "weekly", 1), stats("dry_spell_days", "weekly", 2)
    assert week1["stats"]["min"] == week1["stats"]["max"] == 4
    assert week2["stats"]["min"] == week2["stats"]["max"] == 2
    assert week1["legend"]["unit"] == "days" and week1["legend"]["categorical"] is False

    # Daily maps flag each cell 100 (in the spell / a rain day) or 0.
    in_spell, before_spell = stats("dry_spell_days", "daily", 5), stats("dry_spell_days", "daily", 1)
    assert in_spell["stats"]["min"] == 100 and before_spell["stats"]["max"] == 0
    assert in_spell["legend"]["categorical"] is True and in_spell["legend"]["bins"][0]["label"] == "In a dry spell"
    assert stats("wet_spell_days", "daily", 2)["stats"]["min"] == 100
    assert stats("rainy_days", "weekly", 1)["stats"]["min"] == 3

    share = client.get(
        "/subseasonal/area-values", params={"level": "region", "layer": "dry_spell_days", "aggregation": "daily", "index": 5}
    ).json()
    assert share["unit"] == "%" and share["values"]["Northern"] == pytest.approx(100)

    # Omitting the period keeps the whole-window view for indicators.
    assert client.get("/subseasonal/layer", params={"layer": "rainy_days"}).json()["aggregation"] == "total"


def test_onset_layer_tiles_and_area_values(client):
    layer = client.get("/subseasonal/layer", params={"layer": "onset", "aggregation": "total"}).json()
    assert layer["aggregation"] == "total" and layer["legend"]["unit"] == "date"
    assert layer["legend"]["bins"][1]["label"] == "25 Sep–1 Oct"
    # The fixture never reaches 20 mm in 3 days (15 mm on days 1-3), so nowhere has an onset.
    assert layer["stats"]["share"] == 0 and layer["stats"]["min"] is None
    tile = client.get(layer["tile_url"].replace("{z}", "7").replace("{x}", "63").replace("{y}", "61"))
    assert tile.status_code == 200 and tile.content.startswith(b"\x89PNG")
    values = client.get("/subseasonal/area-values", params={"level": "region", "layer": "onset", "aggregation": "total"}).json()
    assert values["unit"] == "date" and values["values"]["Northern"] == 0
    assert client.get("/subseasonal/layer", params={"layer": "onset", "aggregation": "weekly"}).status_code == 422

    # By day (the default): nothing has started and nothing is coming, so every area reads "not in forecast".
    status = client.get("/subseasonal/layer", params={"layer": "onset", "index": 5}).json()
    assert status["aggregation"] == "daily" and status["legend"]["unit"] == "days_to_onset"
    assert status["title"].startswith("Onset status") and status["stats"]["share"] == 0 and status["stats"]["upcoming_share"] == 0
    countdown = client.get("/subseasonal/area-values", params={"level": "region", "layer": "onset", "index": 5}).json()
    assert countdown["values"]["Northern"] == legends.NO_ONSET


def test_onset_countdown_reads_started_soon_and_none():
    from cumulus.services.subseasonal_service import onset_countdown

    onset_day = np.array([3.0, 10.0, 0.0, np.nan])
    values = onset_countdown(onset_day, 5)
    assert values[:3].tolist() == [-2.0, 5.0, legends.NO_ONSET] and np.isnan(values[3])
    classes = legends.classify(values, legends.ONSET_COUNTDOWN).tolist()
    labels = [legends.ONSET_COUNTDOWN.bins[i].label for i in classes[:3]]
    assert labels == ["Started", "4–7 days", "Not in forecast"] and classes[3] == -1


def test_point_and_area_series(client):
    point = client.get("/subseasonal/sample", params={"latitude": 5.6, "longitude": -0.2}).json()
    assert point["inside_ghana"] and point["region"] == "Greater Accra"
    assert point["metrics"]["rainy_days"] == 3 + 3  # days 1-3 plus even days 10, 12, 14
    assert [(spell["kind"], spell["start_day"], spell["end_day"]) for spell in point["spells"]] == [("wet", 1, 3), ("dry", 4, 9)]
    assert point["spells"][0]["open_start"] is True
    assert point["days"][4]["spell"] == "dry" and point["days"][0]["spell"] == "wet"
    assert point["onset"] is None

    area = client.get("/subseasonal/area", params={"level": "region", "name": "northern"})
    assert area.status_code == 200 and area.json()["name"] == "Northern" and area.json()["cell_count"] > 100


def test_subseasonal_errors_are_structured(client):
    assert client.get("/subseasonal/layer", params={"layer": "humidity"}).json()["error_code"] == "invalid_subseasonal_layer"
    assert client.get("/subseasonal/layer", params={"aggregation": "daily", "index": 15}).status_code == 422
    assert client.get("/subseasonal/layer", params={"layer": "rainy_days", "aggregation": "monthly"}).status_code == 422
    assert client.get("/subseasonal/layer", params={"run_id": "missing"}).json()["error_code"] == "subseasonal_run_not_found"
    assert client.get("/subseasonal/area", params={"level": "district", "name": "Atlantis"}).status_code == 404
    assert client.get("/subseasonal/sample", params={"latitude": 20, "longitude": 0}).json()["error_code"] == "invalid_coordinates"


def test_runs_endpoint_without_artifacts_returns_503(settings):
    from cumulus.main import app

    response = TestClient(app).get("/subseasonal/runs")
    assert response.status_code == 503 and response.json()["error_code"] == "subseasonal_run_not_available"


# --------------------------------------------------------------------------- ensembles

MEMBERS = 10


def _member_value(day: int, member: int) -> float:
    # Days 1-3 wet except member 0 (0.9 mm); members 7-9 get 80 mm on day 2; days 4-9 dry for all.
    if day == 2 and member >= 7:
        return 80.0
    if day <= 3:
        return 5.0 * (member + 1) / 5.5
    return _daily_value(day)


def _make_ensemble_run(root: Path, days: int = 14, *, dim: str = "number") -> Path:
    run_dir = root / f"{INIT:%Y-%m-%d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for day in range(1, days + 1):
        field = np.empty((1, MEMBERS, LATITUDES.size, LONGITUDES.size), dtype=np.float32)
        for member in range(MEMBERS):
            field[0, member] = _member_value(day, member)
        dataset = xr.Dataset(
            {"precip_24h": (("time", dim, "latitude", "longitude"), field)},
            coords={
                "time": [INIT + pd.Timedelta(days=day)],
                dim: np.arange(MEMBERS),
                "latitude": LATITUDES,
                "longitude": LONGITUDES,
                "init_time": INIT,
                "lead_time": ("time", [pd.Timedelta(days=day)]),
            },
        )
        dataset.to_netcdf(run_dir / f"{INIT:%Y-%m-%d}-00-{day * 24:04d}.nc", engine="h5netcdf")
    return run_dir


@pytest.fixture()
def ensemble_client(settings, tmp_path):
    _make_ensemble_run(tmp_path / "ens", days=14)
    ingest_run(settings, LocalFolderSource(tmp_path / "ens"))
    from cumulus.main import app

    return TestClient(app)


def test_ingest_keeps_ensemble_members(settings, tmp_path):
    _make_ensemble_run(tmp_path / "ens", days=7, dim="member")
    result = ingest_run(settings, LocalFolderSource(tmp_path / "ens"))
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["ensemble"] == "ensemble" and manifest["members"] == MEMBERS
    with xr.open_dataset(result.artifact_path, engine="scipy") as dataset:
        assert dataset["precip"].dims == ("member", "lead_day", "latitude", "longitude")
        assert float(dataset["precip"].isel(member=9, lead_day=1).max()) == pytest.approx(80.0)


def _make_member_folders(root: Path, days: int = 7, members: int = 3) -> Path:
    """The sample-bundle layout: no date folder, one ``member_###`` folder per member, one member per file."""
    for member in range(members):
        folder = root / f"member_{member:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        for day in range(1, days + 1):
            _write_lead_file(folder, day, value=_member_value(day, member))
    return root


def test_local_source_reads_a_flat_run_folder(tmp_path):
    for day in range(1, 4):
        _write_lead_file(tmp_path, day)
    source = LocalFolderSource(tmp_path)
    assert source.list_runs() == [date(2026, 9, 25)]
    assert [item.lead_hours for item in source.list_lead_files(date(2026, 9, 25))] == [24, 48, 72]
    assert [path.parent for path in source.fetch_run(date(2026, 9, 25), tmp_path / "cache")] == [tmp_path] * 3


def test_local_source_lists_member_folders_with_unique_keys(tmp_path):
    _make_member_folders(tmp_path, days=2, members=2)
    source = LocalFolderSource(tmp_path)
    assert source.list_runs() == [date(2026, 9, 25)]
    files = source.list_lead_files(date(2026, 9, 25))
    assert [item.key for item in files] == [
        "member_000/2026-09-25-00-0024.nc",
        "member_000/2026-09-25-00-0048.nc",
        "member_001/2026-09-25-00-0024.nc",
        "member_001/2026-09-25-00-0048.nc",
    ]
    # Member folders inside a dated run folder work the same way.
    _make_member_folders(tmp_path / "dated" / "2026-09-25", days=2, members=2)
    assert len(LocalFolderSource(tmp_path / "dated").list_lead_files(date(2026, 9, 25))) == 4


def test_ingest_stacks_member_folders_into_one_ensemble(settings, tmp_path):
    _make_member_folders(tmp_path / "bundle", days=7, members=3)
    source = LocalFolderSource(tmp_path / "bundle")
    result = ingest_run(settings, source)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["members"] == 3 and manifest["ensemble"] == "ensemble"
    assert manifest["source_files"][0]["file"] == "member_000/2026-09-25-00-0024.nc" and len(manifest["source_files"]) == 21
    with xr.open_dataset(result.artifact_path, engine="scipy") as dataset:
        precip = dataset["precip"]
        assert precip.dims == ("member", "lead_day", "latitude", "longitude")
        # Members keep their own values (day 1: 5 * (m + 1) / 5.5 mm), in member-folder order.
        assert precip.isel(lead_day=0, latitude=10, longitude=10).values.tolist() == pytest.approx([0.9, 1.8, 2.7], abs=0.05)
    # Ingesting the same files again is a no-op, even though file names repeat across members.
    assert ingest_run(settings, source).unchanged


def test_ingest_rejects_member_folders_with_different_leads(settings, tmp_path):
    _make_member_folders(tmp_path / "bundle", days=7, members=2)
    (tmp_path / "bundle" / "member_001" / "2026-09-25-00-0168.nc").unlink()
    with pytest.raises(IngestError, match="same lead days"):
        ingest_run(settings, LocalFolderSource(tmp_path / "bundle"))


def test_registry_provenance_fills_in_what_the_path_does_not_say(settings, tmp_path):
    from cumulus.subseasonal.ingest import for_model

    _make_member_folders(tmp_path / "bundle", days=7, members=2)
    result = ingest_run(for_model(settings, "aurora_imerg_diffusion"), LocalFolderSource(tmp_path / "bundle"))
    provenance = json.loads(result.manifest_path.read_text())["provenance"]
    assert provenance["model_type"] == "diffusion_unet" and provenance["sampler"] == "heun"
    assert provenance["sampler_steps"] == 20 and provenance["ensemble_size"] == 2


def test_shipped_default_model_is_aurora_imerg_unet(monkeypatch):
    monkeypatch.delenv("CUMULUS_SUBSEASONAL__SOURCE_ID", raising=False)
    get_settings.cache_clear()
    try:
        config = get_settings().subseasonal
        assert config.source_id == "aurora_imerg_unet"
        families = {(spec.family, spec.downscaling) for spec in config.models.values()}
        assert families == {(family, method) for family in ("ifs", "aurora_imerg", "aurora_era5") for method in ("unet", "diffusion", "bilinear")}
    finally:
        get_settings.cache_clear()


def test_median_onset_day_ranks_no_onset_last():
    from cumulus.services.subseasonal_service import median_onset_day

    days = np.array([[3.0, 0.0], [5.0, 0.0], [0.0, 4.0], [7.0, 0.0]])
    assert median_onset_day(days).tolist() == [5.0, 0.0]


def test_ensemble_runs_and_layers_report_members_and_probabilities(ensemble_client):
    run = ensemble_client.get("/subseasonal/runs").json()["runs"][0]
    assert run["members"] == MEMBERS and run["ensemble"] == "ensemble" and "Ensemble of 10" in run["guidance"]
    assert run["thresholds"]["extreme_rain_mm"] == 50

    rain = ensemble_client.get("/subseasonal/layer", params={"layer": "rainfall", "aggregation": "daily", "index": 4}).json()
    assert rain["statistic"] == "mean" and rain["members"] == MEMBERS

    rainy = ensemble_client.get("/subseasonal/layer", params={"layer": "rainy_days", "aggregation": "daily", "index": 1}).json()
    assert rainy["statistic"] == "probability" and rainy["legend"]["key"] == "rainy_day_probability"
    assert rainy["stats"]["max"] == pytest.approx(90.0)

    extreme = ensemble_client.get("/subseasonal/layer", params={"layer": "extreme_rain", "aggregation": "daily", "index": 2}).json()
    assert extreme["legend"]["key"] == "extreme_rain_probability" and extreme["stats"]["max"] == pytest.approx(30.0)
    total = ensemble_client.get("/subseasonal/layer", params={"layer": "extreme_rain"}).json()
    assert total["aggregation"] == "total" and total["stats"]["mean"] == pytest.approx(30.0)
    tile = ensemble_client.get("/subseasonal/tiles/7/63/62.png", params={"layer": "extreme_rain", "aggregation": "weekly", "index": 1})
    assert tile.status_code == 200 and tile.headers["content-type"] == "image/png"


def test_ensemble_series_carry_spread_and_spell_probabilities(ensemble_client):
    point = ensemble_client.get("/subseasonal/sample", params={"latitude": 5.6, "longitude": -0.2}).json()
    assert point["members"] == MEMBERS
    day2 = point["days"][1]
    assert day2["extreme_prob"] == 30 and day2["spread"]["max"] == pytest.approx(80.0)
    assert day2["spread"]["min"] < day2["spread"]["p50"] < day2["spread"]["max"]
    assert point["days"][0]["wet_prob"] == 90 and point["days"][4]["spell_prob"]["dry"] == 100
    assert point["weeks"][0]["spread"]["p25"] is not None
    spells = [(spell["kind"], spell["start_day"], spell["end_day"], spell["probability"]) for spell in point["spells"]]
    assert ("wet", 1, 3, 90) in spells and ("dry", 4, 9, 100) in spells
    assert point["metrics"]["extreme_days"] == pytest.approx(0.3)

    area = ensemble_client.get("/subseasonal/area", params={"level": "region", "name": "Northern"}).json()
    assert area["members"] == MEMBERS and area["days"][1]["spread"]["max"] == pytest.approx(80.0, abs=0.2)


def test_ensemble_weeks_carry_day_count_spreads(ensemble_client):
    point = ensemble_client.get("/subseasonal/sample", params={"latitude": 5.6, "longitude": -0.2}).json()
    week = point["weeks"][0]
    counts = week["day_counts"]
    assert set(counts) == {"dry_spell", "wet_spell", "rainy"}
    # Nine members rain on days 1-3 and are dry from day 4 (4 dry-spell days in week 1); the tenth is
    # dry all week (7): the spread shows that disagreement.
    assert counts["dry_spell"]["min"] == pytest.approx(4.0) and counts["dry_spell"]["max"] == pytest.approx(7.0)
    assert counts["dry_spell"]["mean"] == pytest.approx(4.3)
    length = week["end_day"] - week["start_day"] + 1
    for spread in counts.values():
        assert 0 <= spread["min"] <= spread["p25"] <= spread["p50"] <= spread["p75"] <= spread["max"] <= length
    # The mean matches the daily shares: the average member's rain days are the sum of the daily shares.
    rain_share = sum(day["wet_prob"] for day in point["days"][week["start_day"] - 1 : week["end_day"]]) / 100
    assert counts["rainy"]["mean"] == pytest.approx(rain_share, abs=0.1)


def test_single_run_weeks_have_no_day_count_spreads(client):
    single = client.get("/subseasonal/sample", params={"latitude": 5.6, "longitude": -0.2}).json()
    assert single["weeks"][0].get("day_counts") is None


def test_single_run_extreme_rain_is_a_flag(client):
    layer = client.get("/subseasonal/layer", params={"layer": "extreme_rain", "aggregation": "daily", "index": 1}).json()
    assert layer["statistic"] == "single" and layer["legend"]["categorical"] and layer["stats"]["max"] == 0
    point = client.get("/subseasonal/sample", params={"latitude": 5.6, "longitude": -0.2}).json()
    assert point["days"][0]["extreme_prob"] == 0 and point["metrics"]["extreme_days"] == 0


# --------------------------------------------------------------------------- models and ensemble statistics


@pytest.fixture()
def two_model_client(settings, tmp_path):
    from cumulus.subseasonal.ingest import for_model

    _make_run(tmp_path / "det", days=14)
    ingest_run(settings, LocalFolderSource(tmp_path / "det"))
    _make_ensemble_run(tmp_path / "ens", days=14)
    ingest_run(for_model(settings, "ifs_diffusion"), LocalFolderSource(tmp_path / "ens"))
    from cumulus.main import app

    return TestClient(app)


def test_models_endpoint_lists_models_with_runs(two_model_client):
    payload = two_model_client.get("/subseasonal/models").json()
    models = {item["id"]: item for item in payload["models"]}
    assert payload["default_model"] == "ifs_unet"
    assert models["ifs_unet"]["available"] and models["ifs_unet"]["members"] == 1
    assert models["ifs_diffusion"]["available"] and models["ifs_diffusion"]["members"] == MEMBERS
    assert models["aurora_imerg_unet"]["available"] is False and models["aurora_imerg_unet"]["label"] == "Aurora-IMERG · UNet"
    # Each model names its S2S source and downscaler, so the viewer can offer them as two choices.
    assert (models["ifs_diffusion"]["family"], models["ifs_diffusion"]["downscaling"]) == ("ifs", "diffusion")
    assert models["aurora_era5_bilinear"]["family_label"] == "Aurora · ERA5 head"
    assert models["aurora_era5_bilinear"]["downscaling_label"] == "Bilinear"

    diffusion = two_model_client.get("/subseasonal/runs", params={"model": "ifs_diffusion"}).json()
    assert diffusion["runs"][0]["members"] == MEMBERS and diffusion["active_run_id"] == "ifs_diffusion_2026092500"
    assert two_model_client.get("/subseasonal/runs").json()["runs"][0]["members"] == 1
    assert two_model_client.get("/subseasonal/runs", params={"model": "nope"}).status_code == 422
    assert two_model_client.get("/subseasonal/runs", params={"model": "aurora_imerg_unet"}).status_code == 503


def test_synthetic_runs_are_flagged(settings, two_model_client):
    from cumulus.subseasonal.ingest import for_model, source_dir

    manifest_path = source_dir(for_model(settings, "ifs_diffusion")) / "ifs_diffusion_2026092500" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert two_model_client.get("/subseasonal/runs", params={"model": "ifs_diffusion"}).json()["runs"][0]["synthetic"] is False
    manifest["synthetic"] = True
    manifest_path.write_text(json.dumps(manifest))
    assert two_model_client.get("/subseasonal/runs", params={"model": "ifs_diffusion"}).json()["runs"][0]["synthetic"] is True
    models = {item["id"]: item for item in two_model_client.get("/subseasonal/models").json()["models"]}
    assert models["ifs_diffusion"]["synthetic"] is True and models["ifs_unet"]["synthetic"] is False


def test_ensemble_rainfall_mean_median_and_chance(two_model_client):
    base = {"model": "ifs_diffusion", "layer": "rainfall", "aggregation": "daily", "index": 2}
    mean = two_model_client.get("/subseasonal/layer", params=base).json()
    median = two_model_client.get("/subseasonal/layer", params={**base, "stat": "median"}).json()
    chance = two_model_client.get("/subseasonal/layer", params={**base, "stat": "chance", "threshold": 50}).json()
    assert mean["statistic"] == "mean" and median["statistic"] == "median" and chance["statistic"] == "probability"
    # Day 2: members 0-6 get 0.9-6.4 mm, members 7-9 get 80 mm.
    assert median["stats"]["max"] < mean["stats"]["max"]
    assert chance["stats"]["max"] == pytest.approx(30.0) and chance["legend"]["key"] == "rain_chance"
    assert chance["threshold_mm"] == 50 and chance["chance_thresholds"] == [10, 25, 50]
    assert "model=ifs_diffusion" in chance["tile_url"] and "stat=chance" in chance["tile_url"]
    tile_url = chance["tile_url"].replace("{z}", "7").replace("{x}", "63").replace("{y}", "61")
    assert two_model_client.get(tile_url).status_code == 200
    areas = two_model_client.get("/subseasonal/area-values", params={**base, "stat": "chance", "threshold": 50, "level": "region"}).json()
    assert areas["values"]["Northern"] == pytest.approx(30.0)
    total = two_model_client.get(
        "/subseasonal/layer", params={**base, "aggregation": "total", "stat": "chance", "threshold": 100}
    ).json()
    assert total["stats"]["max"] == pytest.approx(30.0)
    bad = two_model_client.get("/subseasonal/layer", params={**base, "stat": "chance", "threshold": 33})
    assert bad.status_code == 422
    assert chance["timeline_unit"] == "%" and len(chance["timeline"]) == 14 and chance["timeline"][1] > 0
    assert chance["timeline"][0] == 0 and mean["timeline"] is None
    assert median["timeline_unit"] == "mm"

    point = two_model_client.get("/subseasonal/sample", params={"model": "ifs_diffusion", "latitude": 5.6, "longitude": -0.2}).json()
    assert len(point["days"][1]["members"]) == MEMBERS and point["days"][1]["members"].count(80.0) == 3
    assert len(point["weeks"][0]["members"]) == MEMBERS and len(point["member_totals"]) == MEMBERS
    assert point["days"][1]["chance"]["50"] == 30 and point["days"][1]["extreme_by_date"] == 30
    assert point["days"][-1]["extreme_by_date"] == 30 and point["weeks"][0]["extreme_prob"] == 30
    assert point["weeks"][0]["chance"]["50"] == 30 and point["total_chance"]["100"] == 30
    # An area's chances are the average of its cells' chances, the same as the map's area values.
    area = two_model_client.get("/subseasonal/area", params={"model": "ifs_diffusion", "level": "region", "name": "Northern"}).json()
    map_values = two_model_client.get(
        "/subseasonal/area-values",
        params={"model": "ifs_diffusion", "layer": "extreme_rain", "aggregation": "daily", "index": 2, "level": "region"},
    ).json()["values"]
    assert area["days"][1]["extreme_prob"] == pytest.approx(map_values["Northern"], abs=0.5)
    single_point = two_model_client.get("/subseasonal/sample", params={"latitude": 5.6, "longitude": -0.2}).json()
    assert single_point["days"][0]["members"] is None and single_point["member_totals"] is None
    # A single run ignores the statistic.
    single = two_model_client.get("/subseasonal/layer", params={"layer": "rainfall", "stat": "median"}).json()
    assert single["statistic"] == "single"


def test_a_real_run_retires_synthetic_demo_runs(settings, tmp_path):
    from cumulus.subseasonal.ingest import for_model, list_run_ids, source_dir

    diffusion = for_model(settings, "ifs_diffusion")
    # A demo run issued later than the real one, flagged synthetic.
    later = pd.Timestamp("2026-10-02T00:00:00")
    _make_run(tmp_path / "demo", days=7, init=later)
    demo = ingest_run(diffusion, LocalFolderSource(tmp_path / "demo"))
    manifest = json.loads(demo.manifest_path.read_text())
    manifest["synthetic"] = True
    demo.manifest_path.write_text(json.dumps(manifest))

    _make_ensemble_run(tmp_path / "real", days=7)
    real = ingest_run(diffusion, LocalFolderSource(tmp_path / "real"))
    assert demo.run_id in real.removed_runs
    assert list_run_ids(diffusion) == [real.run_id]
    assert json.loads((source_dir(diffusion) / "active.json").read_text())["run_id"] == real.run_id


# --------------------------------------------------------------------------- observed rain (IMERG)

# A 0.25 degree grid wider than Ghana, stored as GES DISC does: precipitation(time, lon, lat).
OBS_LATITUDES = np.round(np.arange(2.125, 14.0, 0.25), 4)
OBS_LONGITUDES = np.round(np.arange(-5.875, 4.0, 0.25), 4)


def _write_observed_day(folder: Path, day: date, value: float, *, holes: int = 0, name: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    field = np.full((1, OBS_LONGITUDES.size, OBS_LATITUDES.size), value, dtype=np.float32)
    if holes:
        # Missing cells over Ghana, as IMERG's fill value.
        field[0, 10 : 10 + holes, 12:30] = -9999.9
    dataset = xr.Dataset(
        {"precipitation": (("time", "lon", "lat"), field, {"units": "mm/day"})},
        coords={"time": [pd.Timestamp(day)], "lon": OBS_LONGITUDES, "lat": OBS_LATITUDES},
    )
    path = folder / (name or f"3B-DAY-L.MS.MRG.3IMERG.{day:%Y%m%d}-S000000-E235959.V07B.nc")
    dataset.to_netcdf(path, engine="h5netcdf")
    return path


def _ingest_observed(settings, folder: Path, today: date = date(2026, 10, 15)):
    from cumulus.subseasonal import observed

    return observed.ingest(settings, observed.build_source(settings.subseasonal.observed, "local", folder), today=today)


def test_spell_masks_count_a_spell_under_way_before_the_start():
    dry = np.array([True, True, False, True, True, True, True, True, False])
    in_spell, starts = metrics.spell_masks(dry, 5)
    # Without knowing the days before, days 1-2 are too short, and nothing starts on day 1.
    assert in_spell.tolist() == [False, False, False, True, True, True, True, True, False]
    assert np.flatnonzero(starts).tolist() == [3]
    # Three dry days before the start make days 1-2 part of a five-day spell already under way.
    in_spell, starts = metrics.spell_masks(dry, 5, prefix=np.array([False, True, True, True]))
    assert in_spell[:2].tolist() == [True, True] and not starts[0]
    # A wet day just before: a spell touching day 1 is known to begin there.
    in_spell, starts = metrics.spell_masks(np.array([True] * 6), 5, prefix=np.array([False]))
    assert starts.tolist() == [True, False, False, False, False, False]
    assert metrics.starts_within(np.array([False, False, True, False]), 2).tolist() == [False, True, True, False]


def test_find_spells_with_observed_days_before_the_run():
    series = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rule = {"wet_threshold_mm": 1.0, "dry_spell_min_days": 5, "wet_spell_min_days": 3}
    first = metrics.find_spells(series, **rule, prefix=np.array([0.0, 0.0, 0.0]))[0]
    assert (first.kind, first.start_day, first.end_day, first.open_start) == ("dry", 1, 2, True)
    assert metrics.find_spells(np.zeros(6), **rule, prefix=np.array([4.0]))[0].open_start is False
    assert metrics.find_spells(np.zeros(6), **rule)[0].open_start is True


def test_observed_ingest_reads_imerg_daily_files(settings, tmp_path):
    from datetime import timedelta

    from cumulus.subseasonal import observed

    raw = tmp_path / "imerg"
    for offset, value in enumerate([0.0, 2.0, 12.0]):
        _write_observed_day(raw, date(2026, 9, 25) + timedelta(days=offset), value)
    _write_observed_day(raw, date(2026, 9, 28), 4.0, holes=12)  # too much of Ghana missing
    _write_observed_day(raw, date(2026, 3, 1), 9.0)  # outside the kept window
    result = _ingest_observed(settings, raw)
    assert (result.first_date, result.last_date, result.days) == (date(2026, 9, 25), date(2026, 9, 28), 4)
    manifest = json.loads((Path(settings.subseasonal.observed.artifact_dir) / "manifest.json").read_text())
    assert manifest["coverage"]["2026-09-25"] == 1.0 and manifest["coverage"]["2026-09-28"] < 0.95
    loaded = observed.load(settings.subseasonal.observed)
    # Latitude ascending on a (lat, lon) grid, fill values masked, 0.1 mm packing.
    assert loaded.values.shape[1:] == (loaded.latitudes.size, loaded.longitudes.size)
    assert loaded.latitudes[0] < loaded.latitudes[-1] and float(np.nanmax(loaded.values[2])) == pytest.approx(12.0)
    # Nothing new: a second run changes nothing; a new day is folded in.
    assert _ingest_observed(settings, raw).unchanged
    _write_observed_day(raw, date(2026, 9, 29), 1.5)
    added = _ingest_observed(settings, raw)
    assert added.last_date == date(2026, 9, 29) and added.new_dates == (date(2026, 9, 29),)


def test_observed_files_named_by_date_without_a_time_coordinate(settings, tmp_path):
    from cumulus.subseasonal import observed

    raw = tmp_path / "imerg"
    raw.mkdir()
    rate = np.full((OBS_LATITUDES.size, OBS_LONGITUDES.size), 3.0, dtype=np.float32)
    dataset = xr.Dataset(
        {"precip": (("lat", "lon"), rate, {"units": "mm/hr"})},
        coords={"lat": OBS_LATITUDES, "lon": OBS_LONGITUDES},
    )
    dataset.to_netcdf(raw / "imerg_2026-09-30.nc", engine="h5netcdf")
    result = _ingest_observed(settings, raw)
    assert result.last_date == date(2026, 9, 30)
    # mm/hr is a daily mean rate: 3 mm/hr is 72 mm in the day.
    assert float(np.nanmax(observed.load(settings.subseasonal.observed).values)) == pytest.approx(72.0)


@pytest.fixture()
def observed_client(settings, tmp_path):
    """An ensemble run from 25 Sep with observations for 22-27 Sep: three dry days before the run,
    then 0, 0 and 12 mm on its first three days (whatever the members had forecast)."""
    _make_ensemble_run(tmp_path / "ens", days=14)
    ingest_run(settings, LocalFolderSource(tmp_path / "ens"))
    raw = tmp_path / "imerg"
    for day, value in [(22, 0.0), (23, 0.0), (24, 0.0), (25, 0.0), (26, 0.0), (27, 12.0)]:
        _write_observed_day(raw, date(2026, 9, day), value)
    _ingest_observed(settings, raw)
    from cumulus.main import app

    return TestClient(app)


def test_observed_days_replace_the_forecast_in_every_member(observed_client):
    point = observed_client.get("/subseasonal/sample", params={"latitude": 5.6, "longitude": -0.2}).json()
    days = point["days"]
    assert point["observed_days"] == 3 and point["observed_through"] == "2026-09-27"
    assert [day["observed"] for day in days[:4]] == [True, True, True, False]
    # Observed days are certain: every member holds the observation, so the spread collapses and
    # chances are 0 or 100 (day 2 had 30 % of members at 80 mm in the forecast).
    assert days[1]["value"] == 0.0 and days[1]["spread"]["min"] == days[1]["spread"]["max"] == 0.0
    assert days[1]["extreme_prob"] == 0 and days[2]["wet_prob"] == 100 and days[0]["wet_prob"] == 0
    assert days[1]["chance"]["10"] == 0 and days[2]["chance"]["10"] == 100
    # What had been forecast is kept for the drawer.
    assert days[1]["forecast_value"] > 20 and days[3].get("forecast_value") is None
    # Three dry days before the run plus the two dry observed days make a dry spell under way.
    assert days[0]["spell_prob"]["dry"] == 100 and days[1]["spell_prob"]["dry"] == 100
    assert days[0]["spell_start_prob"]["dry"] == 0
    # From day 4 every member is dry for six days: a spell that begins on day 4.
    assert days[3]["spell_start_prob"]["dry"] == 100 and days[2]["spell_ahead_prob"]["dry"] == 100
    assert point["spells"][0]["kind"] == "dry" and point["spells"][0]["open_start"] is True
    assert point["weeks"][0]["observed_days"] == 3 and point["weeks"][1]["observed_days"] == 0


def test_observed_days_reach_the_runs_list_maps_and_tiles(observed_client):
    run = observed_client.get("/subseasonal/runs").json()["runs"][0]
    assert run["observed_through"] == "2026-09-27" and run["observed_source"].startswith("IMERG")
    assert run["observed_version"] and [day["observed"] for day in run["days"][:4]] == [True, True, True, False]
    elapsed = observed_client.get("/subseasonal/layer", params={"layer": "rainfall", "aggregation": "daily", "index": 2}).json()
    ahead = observed_client.get("/subseasonal/layer", params={"layer": "rainfall", "aggregation": "daily", "index": 5}).json()
    assert elapsed["observed"] is True and ahead["observed"] is False and elapsed["observed_days"] == 3
    assert "obs=" in elapsed["tile_url"]
    # The map agrees: the elapsed day's extreme-rain share is 0 everywhere now.
    extreme = observed_client.get("/subseasonal/layer", params={"layer": "extreme_rain", "aggregation": "daily", "index": 2}).json()
    assert extreme["stats"]["max"] == 0


def test_no_observations_leave_the_forecast_alone(ensemble_client):
    point = ensemble_client.get("/subseasonal/sample", params={"latitude": 5.6, "longitude": -0.2}).json()
    assert point["observed_days"] == 0 and point["observed_through"] is None
    assert not any(day["observed"] for day in point["days"])
    assert point["days"][1]["extreme_prob"] == pytest.approx(30, abs=1)
    # Without observations before the run, a spell touching day 1 has no known start.
    assert point["days"][0]["spell_start_prob"]["dry"] == 0


# --------------------------------------------------------------------------- spell length chosen by the viewer


def test_spell_length_is_chosen_by_the_viewer(client):
    def layer(**params):
        response = client.get("/subseasonal/layer", params={"layer": "dry_spell_days", **params})
        assert response.status_code == 200, response.text
        return response.json()

    # Days 4-9 are a 6-day dry run: a spell for "5+ days", not for "7+ days".
    assert layer(aggregation="daily", index=5, spell_days=5)["stats"]["max"] == 100
    assert layer(aggregation="daily", index=5, spell_days=7)["stats"]["max"] == 0
    assert layer(aggregation="total", stat="days", spell_days=6)["stats"]["max"] == 6
    assert layer(aggregation="total", spell_days=7)["stats"]["max"] == 0
    # A spell touching a week counts for that week's chance, even when most of it lies outside.
    week2 = layer(aggregation="weekly", index=2, spell_days=6)
    assert week2["stats"]["min"] == 100 and week2["spell_days"] == 6 and week2["spell_stat"] == "chance"
    assert "spell_days=6" in week2["tile_url"] and "stat=chance" in week2["tile_url"]
    assert "6+ days" in week2["title"] and "6+ days" in week2["legend"]["note"]
    # Out of range, or an unknown statistic, is refused.
    assert client.get("/subseasonal/layer", params={"layer": "dry_spell_days", "spell_days": 40}).status_code == 422
    assert client.get("/subseasonal/layer", params={"layer": "dry_spell_days", "aggregation": "weekly", "stat": "mean"}).status_code == 422
    # Tiles differ by length.
    def tile(length):
        url = layer(aggregation="daily", index=5, spell_days=length)["tile_url"]
        return client.get(url.replace("{z}", "7").replace("{x}", "63").replace("{y}", "61")).content

    assert tile(5) != tile(7)


def test_series_use_the_chosen_spell_lengths(client):
    base = {"latitude": 5.6, "longitude": -0.2}
    default = client.get("/subseasonal/sample", params=base).json()
    longer = client.get("/subseasonal/sample", params={**base, "dry_spell_days": 7}).json()
    assert [spell["kind"] for spell in default["spells"]].count("dry") >= 1
    # The 6-day dry run is no spell when 7 days are asked for, and the series says which length it used.
    assert all(spell["start_day"] != 4 for spell in longer["spells"] if spell["kind"] == "dry")
    assert longer["thresholds"]["dry_spell_min_days"] == 7 and longer["thresholds"]["default_dry_spell_days"] == 5
    assert default["weeks"][0]["spell_chance"]["dry"] == 100 and longer["weeks"][0]["spell_chance"]["dry"] == 0
    assert default["spell_chance_total"]["dry"] == 100
    assert client.get("/subseasonal/sample", params={**base, "wet_spell_days": 99}).status_code == 422
    run = client.get("/subseasonal/runs").json()["runs"][0]
    assert run["spell_days_range"] == {"dry": [2, 15], "wet": [2, 10]}


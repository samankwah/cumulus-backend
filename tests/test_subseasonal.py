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
from cumulus.subseasonal.sources import AzureBlobSasSource, LocalFolderSource, parse_lead_file
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
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__RAW_CACHE_DIR", str(tmp_path / "raw-cache"))
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
    assert {layer["layer"] for layer in run["layers"]} == {"rainfall", "rainy_days", "dry_spell_days", "wet_spell_days"}


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
    dry = client.get("/subseasonal/layer", params={"layer": "dry_spell_days"}).json()
    # Days 4-9 are a 6-day dry run (>= 5) -> 6 dry-spell days everywhere.
    assert dry["stats"]["min"] == dry["stats"]["max"] == 6
    values = client.get(
        "/subseasonal/area-values", params={"level": "district", "layer": "rainfall", "aggregation": "daily", "index": 1}
    ).json()["values"]
    assert len(values) >= 259 and values["Tamale"] == pytest.approx(5.0)


def test_indicator_layers_support_daily_and_weekly_periods(client):
    layers = {item["layer"]: item for item in client.get("/subseasonal/runs").json()["runs"][0]["layers"]}
    assert set(layers["dry_spell_days"]["aggregations"]) == {"daily", "weekly", "total"}

    def stats(layer, aggregation, index=None):
        params = {"layer": layer, "aggregation": aggregation}
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


def test_point_and_area_series(client):
    point = client.get("/subseasonal/sample", params={"latitude": 5.6, "longitude": -0.2}).json()
    assert point["inside_ghana"] and point["region"] == "Greater Accra"
    assert point["metrics"]["rainy_days"] == 3 + 3  # days 1-3 plus even days 10, 12, 14
    assert [(spell["kind"], spell["start_day"], spell["end_day"]) for spell in point["spells"]] == [("wet", 1, 3), ("dry", 4, 9)]
    assert point["spells"][0]["open_start"] is True
    assert point["days"][4]["spell"] == "dry" and point["days"][0]["spell"] == "wet"

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

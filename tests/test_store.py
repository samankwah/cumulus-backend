"""The Blob store sync (cumulus.subseasonal.store), against an in-memory stand-in for Vercel Blob."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cumulus.settings import get_settings
from cumulus.subseasonal import store


class FakeBlob:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.puts: list[str] = []
        self.deletes: list[str] = []

    def get(self, _settings, pathname: str) -> bytes | None:
        return self.files.get(pathname)

    def put(self, _settings, pathname: str, data: bytes, *, content_type: str, max_age: int) -> None:
        assert max_age >= 60 and content_type
        self.files[pathname] = data
        self.puts.append(pathname)

    def delete(self, _settings, pathnames: list[str]) -> None:
        for name in pathnames:
            self.files.pop(name, None)
            self.deletes.append(name)


@pytest.fixture()
def blob(monkeypatch):
    fake = FakeBlob()
    monkeypatch.setattr(store, "_get", fake.get)
    monkeypatch.setattr(store, "_put", fake.put)
    monkeypatch.setattr(store, "_delete", fake.delete)
    monkeypatch.setenv(store.TOKEN_ENV, "vercel_blob_rw_store123_secret")
    store._reset_for_tests()
    yield fake
    store._reset_for_tests()


def _use(monkeypatch, base: Path, *, url: str | None = "https://store123.public.blob.vercel-storage.com"):
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__ARTIFACT_DIR", str(base / "subseasonal"))
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__ARCHIVE_DIR", str(base / "archive"))
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__OBSERVED__ARTIFACT_DIR", str(base / "observed"))
    if url:
        monkeypatch.setenv("CUMULUS_STORE__URL", url)
    else:
        monkeypatch.delenv("CUMULUS_STORE__URL", raising=False)
    get_settings.cache_clear()
    store._reset_for_tests()
    return get_settings()


def _write_run(base: Path, model: str, run_id: str, payload: bytes = b"netcdf-bytes") -> None:
    run = base / "subseasonal" / model / run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": run_id, "artifact": "rainfall.nc"}))
    (run / "rainfall.nc").write_bytes(payload)
    (base / "subseasonal" / model / "active.json").write_text(json.dumps({"run_id": run_id}))
    archive = base / "archive" / model
    archive.mkdir(parents=True, exist_ok=True)
    (archive / f"{run_id}.summary.json.gz").write_bytes(b"summary-" + run_id.encode())


def test_publish_then_mirror_fetches_big_files_only_when_opened(blob, monkeypatch, tmp_path):
    source = tmp_path / "ingest"
    _write_run(source, "ifs_unet", "ifs_unet_2026111300")
    (source / "subseasonal").joinpath("ghana_district_mask.npz").write_bytes(b"mask")
    result = store.publish(_use(monkeypatch, source))
    assert "subseasonal/ifs_unet/ifs_unet_2026111300/rainfall.nc" in result.uploaded
    assert result.promoted == ["ifs_unet"]
    # Indexes go up last, so a reader never sees a file listed that is not there yet.
    assert blob.puts[-1] == "observed/imerg/index.json" or blob.puts[-1].endswith("index.json")

    mirror = tmp_path / "api"
    settings = _use(monkeypatch, mirror)
    store.refresh(settings)
    run = mirror / "subseasonal" / "ifs_unet" / "ifs_unet_2026111300"
    assert (run / "manifest.json").exists() and (mirror / "subseasonal" / "ghana_district_mask.npz").exists()
    assert json.loads((mirror / "subseasonal" / "ifs_unet" / "active.json").read_text())["run_id"] == "ifs_unet_2026111300"
    assert not (run / "rainfall.nc").exists()
    assert store.ensure_file(settings, run / "rainfall.nc")
    assert (run / "rainfall.nc").read_bytes() == b"netcdf-bytes"
    # Summaries are listed without being downloaded, and fetched when read.
    assert store.listed(settings, mirror / "archive" / "ifs_unet") == ["ifs_unet_2026111300.summary.json.gz"]
    assert not (mirror / "archive" / "ifs_unet" / "ifs_unet_2026111300.summary.json.gz").exists()


def test_ci_round_trip_prunes_runs_but_never_the_archive(blob, monkeypatch, tmp_path):
    first = tmp_path / "first"
    _write_run(first, "aurora_imerg_diffusion", "aurora_imerg_diffusion_2026111600")
    store.publish(_use(monkeypatch, first))

    # A later CI job: pull (manifests only), ingest a new run, retention removes the old one.
    ci = tmp_path / "ci"
    settings = _use(monkeypatch, ci)
    store.pull(settings)
    old = ci / "subseasonal" / "aurora_imerg_diffusion" / "aurora_imerg_diffusion_2026111600"
    assert (old / "manifest.json").exists() and not (old / "rainfall.nc").exists()
    # Nothing changed yet: publishing must not delete the run file it never downloaded.
    unchanged = store.publish(settings)
    assert unchanged.uploaded == [] and unchanged.deleted == []

    for path in old.iterdir():
        path.unlink()
    old.rmdir()
    (ci / "archive" / "aurora_imerg_diffusion").mkdir(parents=True, exist_ok=True)
    _write_run(ci, "aurora_imerg_diffusion", "aurora_imerg_diffusion_2026111700")
    result = store.publish(settings)
    assert set(result.deleted) == {
        "subseasonal/aurora_imerg_diffusion/aurora_imerg_diffusion_2026111600/manifest.json",
        "subseasonal/aurora_imerg_diffusion/aurora_imerg_diffusion_2026111600/rainfall.nc",
    }
    assert result.promoted == ["aurora_imerg_diffusion"]
    # The partner archive keeps every run, including the one only the store still has.
    index = json.loads(blob.files["archive/subseasonal/index.json"])["files"]
    assert set(index) == {
        "aurora_imerg_diffusion/aurora_imerg_diffusion_2026111600.summary.json.gz",
        "aurora_imerg_diffusion/aurora_imerg_diffusion_2026111700.summary.json.gz",
    }

    # The API mirror drops the pruned run on its next refresh.
    mirror = tmp_path / "api"
    api_settings = _use(monkeypatch, mirror)
    store.refresh(api_settings)
    runs = sorted(path.name for path in (mirror / "subseasonal" / "aurora_imerg_diffusion").iterdir() if path.is_dir())
    assert runs == ["aurora_imerg_diffusion_2026111700"]


def test_refresh_checks_the_store_at_most_once_a_minute(blob, monkeypatch, tmp_path):
    source, mirror = tmp_path / "ingest", tmp_path / "api"
    source_settings = _use(monkeypatch, source)
    api_settings = _use(monkeypatch, mirror)
    _write_run(source, "ifs_unet", "ifs_unet_2026111100")
    store.publish(source_settings)
    clock = iter([1000.0, 1030.0, 1061.0])
    monkeypatch.setattr(store.time, "monotonic", lambda: next(clock))
    store.refresh(api_settings)

    _write_run(source, "ifs_unet", "ifs_unet_2026111300")
    store.publish(source_settings)
    active = mirror / "subseasonal" / "ifs_unet" / "active.json"
    store.refresh(api_settings)  # 30 s later: not checked yet
    assert json.loads(active.read_text())["run_id"] == "ifs_unet_2026111100"
    store.refresh(api_settings)  # 61 s later: picked up
    assert json.loads(active.read_text())["run_id"] == "ifs_unet_2026111300"


def test_without_a_store_url_nothing_changes(blob, monkeypatch, tmp_path):
    local = tmp_path / "local"
    _write_run(local, "ifs_unet", "ifs_unet_2026111300")
    settings = _use(monkeypatch, local, url=None)
    store.refresh(settings)
    assert blob.puts == [] and blob.files == {}
    run = local / "subseasonal" / "ifs_unet" / "ifs_unet_2026111300"
    assert store.ensure_file(settings, run / "rainfall.nc")
    assert not store.ensure_file(settings, run / "missing.nc")
    assert store.listed(settings, local / "archive" / "ifs_unet") == []
    with pytest.raises(store.StoreError):
        store.publish(settings)


@pytest.fixture(autouse=True)
def _clear_settings():
    yield
    get_settings.cache_clear()


def test_publish_never_wipes_the_store_from_a_job_that_did_not_pull(blob, monkeypatch, tmp_path):
    source = tmp_path / "ingest"
    for day in range(10, 16):
        _write_run(source, "ifs_unet", f"ifs_unet_202611{day}00")
    store.publish(_use(monkeypatch, source))
    before = dict(blob.files)

    # An empty checkout (e.g. the observed-rain job) publishes nothing for the forecast trees.
    empty = _use(monkeypatch, tmp_path / "empty")
    store.publish(empty)
    assert blob.files == before

    # A checkout holding only one run would delete most of the store: refused.
    lone = tmp_path / "lone"
    _write_run(lone, "ifs_unet", "ifs_unet_2026111500")
    with pytest.raises(store.StoreError, match="refusing"):
        store.publish(_use(monkeypatch, lone), only=["subseasonal"])
    assert blob.files == before

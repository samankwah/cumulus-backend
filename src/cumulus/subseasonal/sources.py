"""Where raw IFS-UNet runs come from.

Upstream publishes one NetCDF file per lead day, grouped by init date::

    <prefix>YYYY-MM-DD/YYYY-MM-DD-HH-LLLL.nc     (HH = init hour, LLLL = lead hours)

Two variations are accepted as well: a *flat* folder holding one run's files directly (the init
date then comes from the file names, as in the downscaled sample bundles), and ensemble members
split into ``member_###/`` subfolders, each with its own full set of lead files::

    <prefix>[YYYY-MM-DD/]member_000/YYYY-MM-DD-HH-LLLL.nc

Two interchangeable sources implement :class:`RunSource`: a local folder (the sample run,
manual drops) and Azure Blob Storage read through a container SAS URL. The Azure adapter talks
to the Blob REST API with ``urllib`` so no SDK is needed in the ingest environment.

Upstream's *live* layout puts the run date above the model folders, one tree per S2S source::

    <root>/<src>/<region>/YYYY-MM-DD/<model_dir>/[<ckpt_dir>/]<variable>/precip_24h/data/
        YYYY-MM-DD-HH-LLLL.nc            (deterministic; the ensemble mean when members exist)
        member_###/YYYY-MM-DD-HH-LLLL.nc (ensemble members)

:class:`LiveAzureSource` and :class:`LiveLocalSource` read it for one model, picked by its
model folder, checkpoint prefix and variable (see :class:`LiveLayout`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import date
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

RUN_DIR_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
LEAD_FILE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{4})\.nc$")
MEMBER_DIR_PATTERN = re.compile(r"^member_?\d+$", re.IGNORECASE)


class SourceError(RuntimeError):
    """Raised when a run cannot be listed or downloaded."""


@dataclass(frozen=True)
class LeadFile:
    name: str
    init_date: date
    init_hour: int
    lead_hours: int
    size: int | None = None
    # The ``member_###`` folder the file sits in, when members are split across folders.
    member: str | None = None

    @property
    def key(self) -> str:
        """The file's path relative to the run folder; unique even when members share file names."""
        return f"{self.member}/{self.name}" if self.member else self.name


class RunSource(Protocol):
    label: str
    # The folder or blob prefix the runs live under; model provenance is parsed from it.
    location: str

    def list_runs(self) -> list[date]: ...

    def list_lead_files(self, init_date: date) -> list[LeadFile]: ...

    def fetch_run(self, init_date: date, destination: Path) -> list[Path]: ...


def parse_lead_file(name: str, size: int | None = None) -> LeadFile | None:
    match = LEAD_FILE_PATTERN.match(Path(name).name)
    if not match:
        return None
    return LeadFile(
        name=Path(name).name,
        init_date=date.fromisoformat(match.group(1)),
        init_hour=int(match.group(2)),
        lead_hours=int(match.group(3)),
        size=size,
    )


MODEL_DIR_PATTERN = re.compile(r"^model_(?P<type>(?:.+?_)?unet)(?:_|$)")
# Upstream's folder for plain bilinear interpolation (no model, no checkpoint level).
BILINEAR_DIR_PATTERN = re.compile(r"^bilinear(?:_|$)")
SAMPLER_NUMBER_PATTERN = re.compile(r"^(?P<key>n|cfg|ens)(?P<value>\d+(?:\.\d+)?)$")


def parse_model_path(location: str) -> dict[str, Any]:
    """Model provenance from a path such as ``.../model_<arch>/ckpt-<name>-<step>_<sampler>_n20_cfg3_ens10/``.

    Upstream encodes the downscaling model in the directory names: ``model_*`` names the
    architecture and training setup, ``ckpt-*`` the checkpoint followed by the sampler settings.
    Anything missing is left out; the raw directory names are always kept.
    """
    segments = [part for part in re.split(r"[\\/]+", location) if part]
    model_dir = next((part for part in segments if part.startswith("model_")), None)
    checkpoint_dir = next((part for part in segments if part.startswith("ckpt")), None)
    found: dict[str, Any] = {}
    bilinear_dir = next((part for part in segments if BILINEAR_DIR_PATTERN.match(part)), None)
    if model_dir is None and bilinear_dir:
        found["model_dir"] = bilinear_dir
        found["model_type"] = "bilinear interpolation"
    if model_dir:
        found["model_dir"] = model_dir
        match = MODEL_DIR_PATTERN.match(model_dir)
        found["model_type"] = match.group("type") if match else model_dir.removeprefix("model_")
    if checkpoint_dir:
        found["checkpoint_dir"] = checkpoint_dir
        checkpoint, *settings = checkpoint_dir.split("_")
        found["checkpoint"] = checkpoint
        step = re.search(r"-(\d+)$", checkpoint)
        if step:
            found["checkpoint_step"] = int(step.group(1))
        sampler: list[str] = []
        for token in settings:
            number = SAMPLER_NUMBER_PATTERN.match(token)
            if number is None:
                # "heun", or "flow_heun" for flow matching.
                sampler.append(token)
                continue
            value = float(number.group("value"))
            key = {"n": "sampler_steps", "cfg": "guidance_scale", "ens": "ensemble_size"}[number.group("key")]
            found[key] = int(value) if value.is_integer() else value
        if sampler:
            found["sampler"] = "_".join(sampler)
    return found


def _sorted_leads(files: list[LeadFile], init_date: date) -> list[LeadFile]:
    return sorted((item for item in files if item.init_date == init_date), key=lambda item: (item.member or "", item.lead_hours))


class _LayoutSource:
    """Run discovery shared by the local and Azure sources, on top of one listing primitive.

    ``_entries(folder)`` returns the files (name, size) and subfolder names directly inside
    ``folder`` (a path relative to the source root, ``""`` for the root itself).
    """

    def _entries(self, folder: str) -> tuple[list[tuple[str, int | None]], list[str]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _lead_files_in(self, folder: str) -> list[LeadFile]:
        """Lead files in ``folder``, or in its ``member_###`` subfolders when it has them.

        Upstream writes the ensemble mean next to the member folders; it is not a member, so the
        files in ``folder`` itself are left out whenever members exist.
        """
        files, folders = self._entries(folder)
        members = sorted(name for name in folders if MEMBER_DIR_PATTERN.match(name))
        found = [] if members else [item for name, size in files if (item := parse_lead_file(name, size)) is not None]
        for member in members:
            member_files, _ = self._entries(_join(folder, member))
            for name, size in member_files:
                item = parse_lead_file(name, size)
                if item is not None:
                    found.append(replace(item, member=member))
        return found

    def _run_folders(self) -> dict[date, str]:
        _, folders = self._entries("")
        return {date.fromisoformat(name): name for name in folders if RUN_DIR_PATTERN.match(name)}

    def list_runs(self) -> list[date]:
        folders = self._run_folders()
        if folders:
            return sorted(folders)
        # A flat folder: one (or more) runs whose files sit directly in the root.
        return sorted({item.init_date for item in self._lead_files_in("")})

    def _run_folder(self, init_date: date) -> str:
        folders = self._run_folders()
        if folders:
            if init_date not in folders:
                raise SourceError(f"No run folder for {init_date.isoformat()} under {self.label}")
            return folders[init_date]
        return ""

    def list_lead_files(self, init_date: date) -> list[LeadFile]:
        return _sorted_leads(self._lead_files_in(self._run_folder(init_date)), init_date)


def _join(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


class LocalFolderSource(_LayoutSource):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.label = f"local:{self.root.name}"
        self.location = self.root.resolve().as_posix()

    def _entries(self, folder: str) -> tuple[list[tuple[str, int | None]], list[str]]:
        directory = self.root / folder if folder else self.root
        if not directory.is_dir():
            if not folder:
                raise SourceError(f"Local run folder does not exist: {self.root}")
            return [], []
        files: list[tuple[str, int | None]] = []
        folders: list[str] = []
        for child in directory.iterdir():
            if child.is_dir():
                folders.append(child.name)
            elif child.is_file():
                files.append((child.name, child.stat().st_size))
        return files, folders

    def fetch_run(self, init_date: date, destination: Path) -> list[Path]:
        # Files are read in place; copying 50+ MB into the cache would only waste disk.
        run_dir = self.root / self._run_folder(init_date)
        return [run_dir / item.key for item in self.list_lead_files(init_date)]


class AzureBlobSasSource(_LayoutSource):
    """Read-only access to a blob container through a SAS URL (needs Read + List permissions)."""

    def __init__(
        self, sas_url: str, prefix: str = "Unet/", *, timeout: float = 60.0, retries: int = 3, workers: int = 8
    ):
        parts = urlsplit(sas_url.strip())
        if parts.scheme != "https" or not parts.netloc or not parts.query:
            raise SourceError(
                "Azure SAS URL must look like https://<account>.blob.core.windows.net/<container>?<sas-token>"
            )
        self._scheme, self._netloc = parts.scheme, parts.netloc
        self._container_path = parts.path.rstrip("/")
        self._sas_params = parse_qsl(parts.query, keep_blank_values=True)
        self.prefix = prefix.lstrip("/")
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"
        self.timeout = timeout
        self.retries = max(1, retries)
        self.workers = max(1, workers)
        # Never put the token in labels or logs.
        self.label = f"azure:{self._netloc}{self._container_path}/{self.prefix}"
        self.location = self.prefix

    def _entries(self, folder: str) -> tuple[list[tuple[str, int | None]], list[str]]:
        base = f"{self.prefix}{folder.strip('/')}/" if folder else self.prefix
        blobs, prefixes = self._list(base, delimiter="/")
        files = [(name[len(base) :], size) for name, size in blobs if "/" not in name[len(base) :]]
        folders = [name[len(base) :].strip("/") for name in prefixes]
        return files, [name for name in folders if name]

    def fetch_run(self, init_date: date, destination: Path) -> list[Path]:
        folder = self._run_folder(init_date)
        run_dir = Path(destination) / init_date.isoformat()
        paths: list[Path] = []
        missing: list[tuple[str, Path]] = []
        for item in self.list_lead_files(init_date):
            target = run_dir / item.key
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.exists() and item.size is not None and target.stat().st_size == item.size):
                missing.append((f"{self.prefix}{_join(folder, item.key)}", target))
            paths.append(target)
        # An ensemble run is ~460 files; fetch several at once.
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(lambda job: self._download(*job), missing))
        return paths

    def list_blobs(self) -> list[tuple[str, int | None]]:
        """Every blob under the prefix, recursively: (path relative to the prefix, size in bytes)."""
        blobs, _ = self._list(self.prefix)
        return [(name[len(self.prefix) :], size) for name, size in blobs if name[len(self.prefix) :]]

    def download(self, relative_name: str, target: Path) -> Path:
        """Download one blob (path relative to the prefix) to ``target``, atomically."""
        target.parent.mkdir(parents=True, exist_ok=True)
        self._download(f"{self.prefix}{relative_name.lstrip('/')}", target)
        return target

    def _url(self, path: str, extra: dict[str, str] | None = None) -> str:
        query = urlencode([*self._sas_params, *(extra or {}).items()])
        return urlunsplit((self._scheme, self._netloc, path, query, ""))

    def _open(self, url: str):
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                return urlopen(Request(url, headers={"x-ms-version": "2021-08-06"}), timeout=self.timeout)
            except HTTPError as exc:
                if exc.code in (401, 403, 404):
                    raise SourceError(
                        f"Azure Blob request failed with HTTP {exc.code}; check the SAS URL, its expiry and Read/List permissions."
                    ) from exc
                last_error = exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
            if attempt + 1 < self.retries:
                time.sleep(min(2**attempt, 8))
        raise SourceError(f"Azure Blob request failed after {self.retries} attempts: {last_error}")

    def _list(self, prefix: str, *, delimiter: str | None = None) -> tuple[list[tuple[str, int | None]], list[str]]:
        blobs: list[tuple[str, int | None]] = []
        prefixes: list[str] = []
        marker = ""
        while True:
            params = {"restype": "container", "comp": "list", "prefix": prefix}
            if delimiter:
                params["delimiter"] = delimiter
            if marker:
                params["marker"] = marker
            with self._open(self._url(self._container_path, params)) as response:
                root = ET.fromstring(response.read())
            for blob in root.iter("Blob"):
                name = blob.findtext("Name") or ""
                length = blob.findtext("Properties/Content-Length")
                blobs.append((name, int(length) if length and length.isdigit() else None))
            prefixes.extend(node.findtext("Name") or "" for node in root.iter("BlobPrefix"))
            marker = root.findtext("NextMarker") or ""
            if not marker:
                return blobs, prefixes

    def _download(self, blob_name: str, target: Path) -> None:
        temporary = target.with_suffix(target.suffix + ".part")
        url = self._url(f"{self._container_path}/{quote(blob_name)}")
        # The body can stall mid-read too (seen on the live container), not just the connect.
        for attempt in range(self.retries):
            try:
                with self._open(url) as response, temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1 << 20)
                break
            except (TimeoutError, ConnectionError, OSError) as exc:
                if attempt + 1 >= self.retries:
                    raise SourceError(f"Download of {blob_name} failed after {attempt + 1} attempts: {exc}") from exc
                time.sleep(min(2**attempt, 8))
        os.replace(temporary, target)


@dataclass(frozen=True)
class LiveLayout:
    """Which model folder of upstream's live layout a source reads (see the module docstring)."""

    model_dir: str
    variable: str
    # Checkpoint folder prefix, e.g. "ckpt-diffusion-717-"; None when the model has no checkpoint
    # level (bilinear). A prefix, so a sampler change (n30 -> n40) still matches.
    ckpt_prefix: str | None = None
    data_subpath: str = "precip_24h/data"


def _checkpoint_rank(name: str) -> tuple[int, str]:
    step = re.search(r"-(\d+)(?:_|$)", name)
    return (int(step.group(1)) if step else -1, name)


class _LiveLayoutMixin:
    """Run discovery for the live layout: date folders at the root, the model's data folder below.

    Every date folder counts as a run; a date without this model's folder simply has no lead
    files, which ``--require-complete`` skips.
    """

    layout: LiveLayout
    label: str
    location: str
    _base_location: str
    _resolved: dict[date, str | None]

    def _init_layout(self, layout: LiveLayout) -> None:
        self.layout = layout
        self._resolved = {}

    def _run_folders(self) -> dict[date, str]:
        _, folders = self._entries("")  # type: ignore[attr-defined]
        return {date.fromisoformat(name): name for name in folders if RUN_DIR_PATTERN.match(name)}

    def list_runs(self) -> list[date]:
        return sorted(self._run_folders())

    def _resolve(self, init_date: date) -> str | None:
        if init_date in self._resolved:
            return self._resolved[init_date]
        layout = self.layout
        model = _join(init_date.isoformat(), layout.model_dir)
        found: str | None = None
        if layout.ckpt_prefix is None:
            found = _join(model, layout.variable, layout.data_subpath)
        else:
            _, checkpoints = self._entries(model)  # type: ignore[attr-defined]
            matches = [name for name in checkpoints if name.startswith(layout.ckpt_prefix)]
            if matches:
                found = _join(model, max(matches, key=_checkpoint_rank), layout.variable, layout.data_subpath)
        self._resolved[init_date] = found
        return found

    def _run_folder(self, init_date: date) -> str:
        folder = self._resolve(init_date)
        if folder is None:
            raise SourceError(f"No {self.layout.model_dir} run for {init_date.isoformat()} under {self.label}")
        # Provenance is parsed from the run actually read (its checkpoint and sampler settings).
        self.location = _join(self._base_location, folder)
        return folder

    def list_lead_files(self, init_date: date) -> list[LeadFile]:
        folder = self._resolve(init_date)
        if folder is None:
            return []
        return _sorted_leads(self._lead_files_in(folder), init_date)  # type: ignore[attr-defined]


class LiveAzureSource(_LiveLayoutMixin, AzureBlobSasSource):
    """One model of upstream's live layout in Azure; ``prefix`` is ``<root>/<src>/<region>/``."""

    def __init__(self, sas_url: str, prefix: str, layout: LiveLayout, **kwargs: Any):
        super().__init__(sas_url, prefix, **kwargs)
        self._init_layout(layout)
        self._base_location = self.prefix
        self.label = f"{self.label}<date>/{layout.model_dir}/{layout.ckpt_prefix or ''}*/{layout.variable}"


class LiveLocalSource(_LiveLayoutMixin, LocalFolderSource):
    """The live layout copied to a local folder (tests, manual drops)."""

    def __init__(self, root: Path, layout: LiveLayout):
        super().__init__(root)
        self._init_layout(layout)
        self._base_location = self.location

"""Where raw IFS-UNet runs come from.

Upstream publishes one NetCDF file per lead day, grouped by init date::

    <prefix>YYYY-MM-DD/YYYY-MM-DD-HH-LLLL.nc     (HH = init hour, LLLL = lead hours)

Two interchangeable sources implement :class:`RunSource`: a local folder (the sample run,
manual drops) and Azure Blob Storage read through a container SAS URL. The Azure adapter talks
to the Blob REST API with ``urllib`` so no SDK is needed in the ingest environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import re
import shutil
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

RUN_DIR_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
LEAD_FILE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{4})\.nc$")


class SourceError(RuntimeError):
    """Raised when a run cannot be listed or downloaded."""


@dataclass(frozen=True)
class LeadFile:
    name: str
    init_date: date
    init_hour: int
    lead_hours: int
    size: int | None = None


class RunSource(Protocol):
    label: str

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


def _sorted_leads(files: list[LeadFile], init_date: date) -> list[LeadFile]:
    return sorted((item for item in files if item.init_date == init_date), key=lambda item: item.lead_hours)


class LocalFolderSource:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.label = f"local:{self.root.name}"

    def list_runs(self) -> list[date]:
        if not self.root.is_dir():
            raise SourceError(f"Local IFS-UNet folder does not exist: {self.root}")
        return sorted(
            date.fromisoformat(child.name)
            for child in self.root.iterdir()
            if child.is_dir() and RUN_DIR_PATTERN.match(child.name)
        )

    def list_lead_files(self, init_date: date) -> list[LeadFile]:
        run_dir = self.root / init_date.isoformat()
        if not run_dir.is_dir():
            raise SourceError(f"No run folder for {init_date.isoformat()} under {self.root}")
        parsed = [parse_lead_file(path.name, path.stat().st_size) for path in run_dir.glob("*.nc")]
        return _sorted_leads([item for item in parsed if item is not None], init_date)

    def fetch_run(self, init_date: date, destination: Path) -> list[Path]:
        # Files are read in place; copying 50+ MB into the cache would only waste disk.
        run_dir = self.root / init_date.isoformat()
        return [run_dir / item.name for item in self.list_lead_files(init_date)]


class AzureBlobSasSource:
    """Read-only access to a blob container through a SAS URL (needs Read + List permissions)."""

    def __init__(self, sas_url: str, prefix: str = "Unet/", *, timeout: float = 60.0, retries: int = 3):
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
        # Never put the token in labels or logs.
        self.label = f"azure:{self._netloc}{self._container_path}/{self.prefix}"

    def list_runs(self) -> list[date]:
        _, prefixes = self._list(self.prefix, delimiter="/")
        runs: list[date] = []
        for name in prefixes:
            folder = name[len(self.prefix) :].strip("/")
            if RUN_DIR_PATTERN.match(folder):
                runs.append(date.fromisoformat(folder))
        return sorted(runs)

    def list_lead_files(self, init_date: date) -> list[LeadFile]:
        blobs, _ = self._list(f"{self.prefix}{init_date.isoformat()}/", delimiter="/")
        parsed = [parse_lead_file(name, size) for name, size in blobs]
        return _sorted_leads([item for item in parsed if item is not None], init_date)

    def fetch_run(self, init_date: date, destination: Path) -> list[Path]:
        run_dir = Path(destination) / init_date.isoformat()
        run_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for item in self.list_lead_files(init_date):
            target = run_dir / item.name
            if not (target.exists() and item.size is not None and target.stat().st_size == item.size):
                self._download(f"{self.prefix}{init_date.isoformat()}/{item.name}", target)
            paths.append(target)
        return paths

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
        with self._open(url) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
        os.replace(temporary, target)

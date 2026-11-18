"""Forecast artifacts kept in Vercel Blob instead of git.

Ingested runs used to be committed to the repository, which grew by ~25 MB a day once every live
model was ingested. Now the ingest publishes its local artifact folders to a public Blob store and
the API keeps a local *mirror* of them, so every reader that opens local files keeps working:

    subseasonal/          <- settings.subseasonal.artifact_dir   (runs, active.json, district mask)
    archive/subseasonal/  <- settings.subseasonal.archive_dir    (partner summaries, kept forever)
    observed/imerg/       <- settings.subseasonal.observed.artifact_dir

Each tree has an ``index.json`` listing its files with size and SHA-1. ``refresh`` (API side,
at most every ``store.refresh_seconds``) brings the small files up to date and drops what the index
no longer lists; big files (``.nc``, archived summaries) are fetched only when first opened, via
``ensure_file``. ``pull`` and ``publish`` are the ingest side: pull before ingesting, publish after.

Nothing here runs unless ``store.url`` is set (CUMULUS_STORE__URL); tests and local development
read the local folders exactly as before. Writing needs BLOB_READ_WRITE_TOKEN.

Talks to the Blob REST API with ``urllib`` (the same calls as the official ``@vercel/blob``
client, API version 13), so no SDK is needed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from cumulus.settings import Settings, get_settings

logger = logging.getLogger(__name__)

INDEX_NAME = "index.json"
BLOB_API_VERSION = "13"
TOKEN_ENV = "BLOB_READ_WRITE_TOKEN"


class StoreError(RuntimeError):
    """Raised when the Blob store cannot be read or written."""


@dataclass(frozen=True)
class Tree:
    prefix: str
    root: Path
    # Files fetched only when first opened (they are big, or there are many of them).
    lazy: Callable[[str], bool]
    # False for the summary archive: its files are never removed from the store.
    prunable: bool = True


def _lazy_artifacts(relative: str) -> bool:
    return relative.endswith(".nc")


def _lazy_everything(_relative: str) -> bool:
    return True


def trees(settings: Settings) -> list[Tree]:
    config = settings.subseasonal
    return [
        Tree("subseasonal", Path(config.artifact_dir), _lazy_artifacts),
        Tree("archive/subseasonal", Path(config.archive_dir), _lazy_everything, prunable=False),
        Tree("observed/imerg", Path(config.observed.artifact_dir), _lazy_artifacts),
    ]


def enabled(settings: Settings) -> bool:
    return bool(settings.store.url)


# --- Blob HTTP -----------------------------------------------------------------------------------


def _token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise StoreError(f"Set {TOKEN_ENV} to write to the Blob store.")
    return token


def _open(request: Request | str, *, timeout: float = 60.0, retries: int = 3):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as exc:
            if exc.code in (400, 401, 403, 404):
                raise
            last = exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 8))
    raise StoreError(f"Blob request failed after {retries} attempts: {last}")


def public_url(settings: Settings, pathname: str) -> str:
    return f"{str(settings.store.url).rstrip('/')}/{quote(pathname)}"


def _get(settings: Settings, pathname: str) -> bytes | None:
    """A public file's bytes, or None when it does not exist."""
    try:
        with _open(public_url(settings, pathname)) as response:
            return response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise StoreError(f"Reading {pathname} from the Blob store failed with HTTP {exc.code}.") from exc


def _api(settings: Settings, path: str, *, method: str, body: bytes | None = None, headers: dict[str, str] | None = None):
    request = Request(
        f"{settings.store.api_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"authorization": f"Bearer {_token()}", "x-api-version": BLOB_API_VERSION, **(headers or {})},
    )
    try:
        with _open(request, timeout=300.0) as response:
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise StoreError(f"Blob {method} {path.split('?')[0]} failed with HTTP {exc.code}: {detail}") from exc
    return json.loads(payload) if payload else {}


def _put(settings: Settings, pathname: str, data: bytes, *, content_type: str, max_age: int) -> None:
    _api(
        settings,
        "/?" + urlencode({"pathname": pathname}),
        method="PUT",
        body=data,
        headers={
            "x-vercel-blob-access": "public",
            "x-add-random-suffix": "0",
            "x-allow-overwrite": "1",
            "x-content-type": content_type,
            "x-cache-control-max-age": str(max_age),
        },
    )


def _delete(settings: Settings, pathnames: list[str]) -> None:
    for start in range(0, len(pathnames), 100):
        urls = [public_url(settings, name) for name in pathnames[start : start + 100]]
        _api(settings, "/delete", method="POST", body=json.dumps({"urls": urls}).encode(), headers={"content-type": "application/json"})


# --- Index ---------------------------------------------------------------------------------------


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_files(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir():
        return {}
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.endswith((".part", ".tmp")) and path.name != INDEX_NAME:
            found[path.relative_to(root).as_posix()] = {"size": path.stat().st_size, "sha1": _sha1(path)}
    return found


def _fetch_index(settings: Settings, tree: Tree) -> dict[str, dict[str, Any]] | None:
    raw = _get(settings, f"{tree.prefix}/{INDEX_NAME}")
    if raw is None:
        return None
    return dict(json.loads(raw).get("files") or {})


def _download(settings: Settings, tree: Tree, relative: str, entry: dict[str, Any]) -> Path:
    target = tree.root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _get(settings, f"{tree.prefix}/{relative}")
    if data is None:
        raise StoreError(f"{tree.prefix}/{relative} is listed in the index but missing from the Blob store.")
    if entry.get("sha1") and hashlib.sha1(data).hexdigest() != entry["sha1"]:
        # A just-overwritten file can still be cached at the edge; never mirror a stale copy.
        raise StoreError(f"{tree.prefix}/{relative} does not match its index entry yet; try again shortly.")
    temporary = target.with_name(target.name + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return target


# --- API side: the mirror ------------------------------------------------------------------------

_lock = threading.Lock()
_indexes: dict[str, dict[str, dict[str, Any]]] = {}
_checked_at: dict[str, float] = {}


def _reset_for_tests() -> None:
    _indexes.clear()
    _checked_at.clear()


def refresh(settings: Settings, *, force: bool = False) -> None:
    """Bring the local mirror up to date with the store (no-op without ``store.url``).

    Errors are logged, not raised: a visitor is better served by the last good mirror.
    """
    if not enabled(settings):
        return
    now = time.monotonic()
    with _lock:
        for tree in trees(settings):
            key = str(tree.root)
            if not force and key in _checked_at and now - _checked_at[key] < settings.store.refresh_seconds:
                continue
            try:
                index = _fetch_index(settings, tree)
                if index is not None:
                    _sync_tree(settings, tree, index)
                    _indexes[key] = index
                _checked_at[key] = now
            except (StoreError, OSError, ValueError) as exc:
                logger.warning("Blob store refresh of %s failed: %s", tree.prefix, exc)


def _sync_tree(settings: Settings, tree: Tree, index: dict[str, dict[str, Any]]) -> None:
    local = {}
    if tree.root.is_dir():
        local = {
            path.relative_to(tree.root).as_posix(): path
            for path in tree.root.rglob("*")
            if path.is_file() and not path.name.endswith(".part")
        }
    wanted: list[tuple[str, dict[str, Any]]] = []
    for relative, entry in index.items():
        path = local.get(relative)
        if tree.lazy(relative):
            # Fetched when opened; a local copy that no longer matches is dropped.
            if path is not None and path.stat().st_size != entry.get("size"):
                path.unlink()
            continue
        if path is None or path.stat().st_size != entry.get("size") or _sha1(path) != entry.get("sha1"):
            wanted.append((relative, entry))
    # A cold start fetches every manifest and pointer (~100 small files): do it in parallel.
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda item: _download(settings, tree, *item), wanted))
    for relative, path in local.items():
        if relative not in index:
            path.unlink()
    _remove_empty_dirs(tree.root)


def _remove_empty_dirs(root: Path) -> None:
    if not root.is_dir():
        return
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def _index_for(settings: Settings, tree: Tree) -> dict[str, dict[str, Any]]:
    """The tree's index, fetched once if needed. Never touches local files.

    Lookups must not sync: in an ingest process, a sync would delete the runs it just wrote
    (they are not in the store yet). Only the API's ``refresh`` and ``pull`` mirror the store.
    """
    key = str(tree.root)
    if key not in _indexes:
        try:
            _indexes[key] = _fetch_index(settings, tree) or {}
        except (StoreError, OSError, ValueError) as exc:
            logger.warning("Reading the Blob store index of %s failed: %s", tree.prefix, exc)
            return {}
    return _indexes[key]


def _tree_for(settings: Settings, path: Path) -> tuple[Tree, str] | None:
    resolved = Path(path).resolve()
    for tree in trees(settings):
        try:
            return tree, resolved.relative_to(tree.root.resolve()).as_posix()
        except ValueError:
            continue
    return None


def ensure_file(settings: Settings, path: Path) -> bool:
    """Make sure ``path`` exists locally, fetching it from the store if the index lists it."""
    path = Path(path)
    if path.exists() or not enabled(settings):
        return path.exists()
    found = _tree_for(settings, path)
    if found is None:
        return False
    tree, relative = found
    entry = _index_for(settings, tree).get(relative)
    if entry is None:
        return False
    try:
        _download(settings, tree, relative, entry)
    except (StoreError, OSError) as exc:
        logger.warning("Fetching %s from the Blob store failed: %s", relative, exc)
        return False
    return True


def listed(settings: Settings, directory: Path) -> list[str]:
    """Names the store lists directly inside ``directory`` (files not yet fetched included)."""
    if not enabled(settings):
        return []
    found = _tree_for(settings, directory)
    if found is None:
        return []
    tree, relative = found
    prefix = f"{relative}/" if relative not in ("", ".") else ""
    names = []
    for name in _index_for(settings, tree):
        if name.startswith(prefix) and "/" not in name[len(prefix) :]:
            names.append(name[len(prefix) :])
    return sorted(names)


# --- Ingest side ---------------------------------------------------------------------------------


def pull(settings: Settings) -> dict[str, int]:
    """Mirror the store before an ingest (big files stay remote until opened). Returns file counts."""
    if not enabled(settings):
        raise StoreError("Set CUMULUS_STORE__URL to pull from the Blob store.")
    _reset_for_tests()
    refresh(settings, force=True)
    return {tree.prefix: len(_indexes.get(str(tree.root), {})) for tree in trees(settings)}


@dataclass
class PublishResult:
    uploaded: list[str]
    deleted: list[str]
    # Models whose active run changed (for announcing).
    promoted: list[str]


def publish(settings: Settings, only: list[str] | None = None, *, allow_mass_delete: bool = False) -> PublishResult:
    """Upload what changed in the local trees, drop what retention removed, and rewrite the indexes.

    A big file that was never fetched locally is kept as long as its folder still exists; a file
    disappears from the store only when its folder (a pruned run) or the file itself was removed.

    ``only`` limits publishing to some trees (by prefix). As guards against wiping the store from a
    job that never pulled it, an empty local tree is skipped, and a publish that would delete more
    than half of a tree's files is refused unless ``allow_mass_delete``.
    """
    if not enabled(settings):
        raise StoreError("Set CUMULUS_STORE__URL to publish to the Blob store.")
    _token()
    result = PublishResult(uploaded=[], deleted=[], promoted=[])
    for tree in trees(settings):
        if only is not None and tree.prefix not in only:
            continue
        remote = _fetch_index(settings, tree) or {}
        local = _local_files(tree.root)
        if not local:
            if remote:
                logger.warning("Not publishing %s: the local folder is empty (was it pulled?).", tree.prefix)
            continue
        index = dict(local)
        for relative, entry in remote.items():
            if relative in local:
                continue
            if not tree.prunable or (tree.lazy(relative) and (tree.root / relative).parent.is_dir()):
                index[relative] = entry
        changed = [relative for relative, entry in local.items() if remote.get(relative, {}).get("sha1") != entry["sha1"]]
        for relative in changed:
            max_age = 60 if relative.endswith(".json") else 300
            content_type = "application/json" if relative.endswith(".json") else "application/octet-stream"
            _put(settings, f"{tree.prefix}/{relative}", (tree.root / relative).read_bytes(), content_type=content_type, max_age=max_age)
            result.uploaded.append(f"{tree.prefix}/{relative}")
            if tree.prefix == "subseasonal" and relative.endswith("/active.json"):
                result.promoted.append(relative.split("/")[0])
        removed = sorted(set(remote) - set(index))
        if len(removed) * 2 > len(remote) and len(removed) > 4 and not allow_mass_delete:
            raise StoreError(
                f"Publishing {tree.prefix} would delete {len(removed)} of its {len(remote)} files; refusing. "
                "Pull the store first, or pass --allow-mass-delete if this is really intended."
            )
        if changed or removed or not remote:
            # The index goes last, so readers never see files that are not uploaded yet.
            body = json.dumps({"version": 1, "files": index}, sort_keys=True).encode()
            _put(settings, f"{tree.prefix}/{INDEX_NAME}", body, content_type="application/json", max_age=60)
        if removed:
            _delete(settings, [f"{tree.prefix}/{relative}" for relative in removed])
            result.deleted.extend(f"{tree.prefix}/{relative}" for relative in removed)
        _indexes[str(tree.root)] = index
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync forecast artifacts with the Vercel Blob store.")
    parser.add_argument("command", choices=["pull", "publish"])
    parser.add_argument("--github-output", type=Path, help="Append changed=/promoted= lines here (CI).")
    parser.add_argument("--only", action="append", help="Publish only this tree (subseasonal, archive/subseasonal, observed/imerg).")
    parser.add_argument("--allow-mass-delete", action="store_true", help="Allow a publish that deletes most of a tree.")
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        if args.command == "pull":
            for prefix, count in pull(settings).items():
                print(f"{prefix}: {count} files listed")
            return 0
        result = publish(settings, args.only, allow_mass_delete=args.allow_mass_delete)
    except StoreError as exc:
        print(f"store {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(f"uploaded {len(result.uploaded)}, deleted {len(result.deleted)}; promoted: {' '.join(result.promoted) or 'none'}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"changed={'true' if result.uploaded or result.deleted else 'false'}\n")
            handle.write(f"promoted={' '.join(result.promoted)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

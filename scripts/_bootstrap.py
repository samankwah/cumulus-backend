from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    src_path = backend_root / "src"
    src_path_str = str(src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

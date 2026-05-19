"""
DuckDB WASM asset management.

On first use, downloads four files from the CDN and caches them under
~/.cache/logan/duckdb-wasm/<version>/.  Subsequent runs copy from the cache
instead of re-downloading.

Why apache-arrow is included
----------------------------
duckdb-browser.mjs (the DuckDB ES module) contains the line:
    import * as u from "apache-arrow";
Apache Arrow is how DuckDB returns query results — it is a hard dependency,
not optional.  The import map in explorer.html redirects that bare specifier
to the locally bundled apache-arrow.esm.js so nothing is fetched from a CDN
at view time.

File inventory
--------------
  apache-arrow.esm.js            ~196 KB   Arrow IPC / columnar runtime
  duckdb-browser.mjs              ~30 KB   DuckDB async orchestration layer
  duckdb-browser-mvp.worker.js   ~820 KB   DuckDB engine running in a Worker
  duckdb-mvp.wasm                 ~39 MB   DuckDB compiled to WebAssembly

Total download: ~10 MB (gzip-compressed from CDN).
Total on disk:  ~40 MB (decompressed, cached once in ~/.cache/logan/).
"""

import gzip
import shutil
import sys
import urllib.request
from pathlib import Path


DUCKDB_VERSION = "1.29.0"
_ARROW_VERSION = "17.0.0"       # satisfies duckdb-wasm's "apache-arrow": "^17.0.0"
_TSLIB_VERSION = "2.6.3"        # transitive dep of apache-arrow +esm
_FLATBUFFERS_VERSION = "24.3.25"  # transitive dep of apache-arrow +esm

_BASE = f"https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@{DUCKDB_VERSION}/dist"
_CDN  = "https://cdn.jsdelivr.net/npm"

# (local filename, source URL, approx compressed MB)
#
# apache-arrow.esm.js (the jsdelivr +esm build) imports tslib and flatbuffers
# using CDN-relative paths like /npm/tslib@2.6.3/+esm.  When served from
# localhost those paths resolve to localhost, causing 404s.  We download those
# two tiny files and redirect them via the import map in explorer.html.
# Both are self-contained (no further external deps).
_ASSETS = [
    ("tslib.esm.js",                 f"{_CDN}/tslib@{_TSLIB_VERSION}/+esm",                 0.003),
    ("flatbuffers.esm.js",           f"{_CDN}/flatbuffers@{_FLATBUFFERS_VERSION}/+esm",      0.003),
    ("apache-arrow.esm.js",          f"{_CDN}/apache-arrow@{_ARROW_VERSION}/+esm",           0.04),
    ("duckdb-browser.mjs",           f"{_BASE}/duckdb-browser.mjs",                          0.01),
    ("duckdb-browser-mvp.worker.js", f"{_BASE}/duckdb-browser-mvp.worker.js",                0.2),
    ("duckdb-mvp.wasm",              f"{_BASE}/duckdb-mvp.wasm",                             7.7),
]


def _cache_dir() -> Path:
    return Path.home() / ".cache" / "logan" / "duckdb-wasm" / DUCKDB_VERSION


def _download_with_progress(url: str, dest: Path, label: str) -> None:
    """
    Stream-download url → dest with a live progress bar.
    Requests gzip compression; decompresses before writing so the cached file
    is always the raw (browser-ready) bytes.
    """
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        encoding = resp.headers.get("Content-Encoding", "")

        chunks = []
        received = 0
        chunk_size = 65_536  # 64 KB

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
            if total and sys.stdout.isatty():
                pct = received * 100 // total
                bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                print(f"\r[logan]   {label} [{bar}] {pct:3d}%", end="", flush=True)

        raw = b"".join(chunks)
        data = gzip.decompress(raw) if encoding == "gzip" else raw

    if sys.stdout.isatty():
        print()  # newline after progress bar

    dest.write_bytes(data)


def ensure_duckdb_assets(verbose: bool = True) -> Path:
    """
    Return the cache directory, downloading any missing files first.

    Raises urllib.error.URLError if a download fails (no internet / CDN down).
    """
    cache = _cache_dir()
    missing = [(f, u, mb) for f, u, mb in _ASSETS if not (cache / f).exists()]

    if not missing:
        return cache

    total_mb = sum(mb for _, _, mb in missing)
    if verbose:
        print(
            f"[logan] Downloading DuckDB WASM v{DUCKDB_VERSION} + "
            f"Apache Arrow v{_ARROW_VERSION} "
            f"(~{total_mb:.0f} MB compressed).\n"
            f"[logan] Cached at: {cache}\n"
            f"[logan] This download happens once; subsequent runs use the cache."
        )

    cache.mkdir(parents=True, exist_ok=True)
    for fname, url, _ in missing:
        dest = cache / fname
        if verbose:
            if not sys.stdout.isatty():
                print(f"[logan]   {fname} …", end=" ", flush=True)
        _download_with_progress(url, dest, fname)
        if verbose:
            kb = dest.stat().st_size // 1024
            if sys.stdout.isatty():
                print(f"[logan]   {fname} — {kb:,} KB")
            else:
                print(f"{kb:,} KB")

    return cache


def copy_duckdb_to_output(libs_dir: str) -> None:
    """
    Copy cached DuckDB WASM assets to <output>/log_diagnosis/libs/duckdb/.

    Downloads missing assets on first call; subsequent calls copy from the
    local cache with no network access.
    """
    cache = ensure_duckdb_assets(verbose=True)
    dest = Path(libs_dir) / "duckdb"
    dest.mkdir(parents=True, exist_ok=True)
    for fname, _, _ in _ASSETS:
        shutil.copy2(cache / fname, dest / fname)

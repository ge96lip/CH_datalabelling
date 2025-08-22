# ehr_prep/io_utils.py
from pathlib import Path
import time, json, pyarrow as pa, pyarrow.parquet as pq
from datetime import datetime
import pyarrow.compute as pc
from dateutil import parser as dparser
import xxhash
from typing import Optional, Tuple, List
import hashlib, json, os, time
from typing import Optional


LAYER1_SCHEMA = pa.schema([
    ("source_tranche", pa.string()),
    ("source_file", pa.string()),
    ("line_start", pa.int64()),
    ("line_end", pa.int64()),
    ("patient_empi", pa.string()),
    ("patient_mrn", pa.string()),
    ("modality", pa.string()),
    ("doc_id", pa.string()),
    ("report_sequence", pa.int64()),
    ("is_free_text", pa.bool_()),
    ("raw_text", pa.large_string()),
    ("report_end_marker", pa.bool_()),
    ("doc_date", pa.date32()),
    ("performed_date", pa.date32()),
    ("ingest_timestamp", pa.timestamp('ms')),
    ("section_hints", pa.list_(pa.string())),
    ("modality_specific", pa.string()),  # small JSON
])

def file_signature(path: str) -> dict:
    """Streaming SHA256 + size + mtime so we can detect changes cheaply."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk: break
            h.update(chunk)
            size += len(chunk)
    mtime = int(os.path.getmtime(path))
    return {"sha256": h.hexdigest(), "size": size, "mtime": mtime}

def ok_dir_for(out_dir: str) -> Path:
    p = Path(out_dir) / "_ok"
    p.mkdir(parents=True, exist_ok=True)
    return p

def failed_dir_for(out_root: str) -> Path:
    p = Path(out_root).parent / "logs" / "_failed"
    p.mkdir(parents=True, exist_ok=True)
    return p

def ok_marker_path(out_dir: str, source_filename: str) -> Path:
    return ok_dir_for(out_dir) / (source_filename + ".ok.json")

def read_ok_marker(ok_path: Path) -> Optional[dict]:
    if not ok_path.exists(): return None
    try:
        return json.loads(ok_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def write_ok_marker(ok_path: Path, payload: dict) -> None:
    tmp = ok_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ok_path)

def timestamp_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def stable_id(tranche, modality, num):
    return f"{tranche}_{modality}_{num:012d}"

def now_ts_ms():
    return int(time.time() * 1000)

def parse_date_safe(s: Optional[str]):
    if not s or not s.strip():
      return None
    try:
        return dparser.parse(s, fuzzy=True).date()
    except Exception:
        return None
def _undictify_strings(tbl: pa.Table, cols: Tuple[str, ...]) -> pa.Table:
        for name in cols:
            if name in tbl.schema.names:
                idx = tbl.schema.get_field_index(name)
                if idx != -1:
                    f = tbl.schema.field(idx)
                    if pa.types.is_dictionary(f.type):
                        tbl = tbl.set_column(idx, name, pc.cast(tbl[name], pa.string()))
        return tbl

class ParquetSink:
    """Buffered writer with Arrow Table flushes."""
    def __init__(self, out_dir: str, row_group_size=128_000):
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.row_group_size = row_group_size
        self._buffer = []
        self._writer = None

    def write_one(self, rec: dict):
        self._buffer.append(rec)
        if len(self._buffer) >= self.row_group_size:
            self.flush()

    def write_many(self, recs: List[dict]):
        self._buffer.extend(recs)
        if len(self._buffer) >= self.row_group_size:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        tbl = pa.Table.from_pylist(self._buffer, schema=LAYER1_SCHEMA)

        # ensure consistent logical types across parts
        tbl = _undictify_strings(
            tbl,
            (
                "modality",
                "source_tranche",
                "source_file",
                "patient_empi",
                "patient_mrn",
                "doc_id",
                "section",  # if present
            ),
        )

        path = self.out_dir / f"part-{xxhash.xxh64_hexdigest(str(now_ts_ms()))}.parquet"

        pq.write_table(
            tbl,
            path,
            compression="zstd",
            version="2.6",
            data_page_size=1 << 16,
            use_dictionary=False,  # keep off for stability
        )
        self._buffer.clear()

    def close(self):
        self.flush()

def pack_meta_small(row: dict, keep_keys: Tuple[str, ...]) -> Optional[str]:
    keep = {k: row[k] for k in keep_keys if k in row and row[k]}
    return None if not keep else json.dumps(keep, ensure_ascii=False)
# ehr_prep/build_timeline_ds.py
from pathlib import Path
import os
import duckdb
import hashlib
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from typing import Optional, List
import datetime as _dt

# -----------------------------
# Config
# -----------------------------
# Layer-1 
L1_GLOB = r"ehr_store/normalized/tranche=*/mod=*/*.parquet"
# Output folder Layer-2
L2_DIR = r"ehr_store/timeline_ds"
# Partitions: patient_bucket in [0..N_BUCKETS-1]
N_BUCKETS = 512

# -----------------------------
# Build Layer-2
# -----------------------------
def _sql_quote_path(p: str) -> str:
    # DuckDB string literal: single quotes; escape any single quotes inside
    return "'" + p.replace("'", "''") + "'"

def build_timeline_ds(l1_glob: str = L1_GLOB,
                      l2_dir: str = L2_DIR,
                      n_buckets: int = N_BUCKETS,
                      overwrite: bool = True, 
                      manifests_dir: Optional[str] = None) -> None:
    out = Path(l2_dir)
    out.mkdir(parents=True, exist_ok=True)
    # open DuckDB connection
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 4) - 1)}")
    l1_glob_sql = _sql_quote_path(Path(l1_glob).as_posix())
    out_dir_sql = Path(l2_dir).as_posix()
    out_dir_sql_q = _sql_quote_path(out_dir_sql)

    # Read all Layer-1 parts. union_by_name (duckdb: parquet_scan + hive_partitioning)
    # L1 schema (from your normalizer):
    #   source_tranche, source_file, line_start, line_end,
    #   patient_empi, patient_mrn, modality, doc_id, is_free_text,
    #   raw_text, report_end_marker, doc_date, performed_date,
    #   section_hints, modality_specific
    #
    con.execute(f"""
        CREATE OR REPLACE VIEW l1 AS
        SELECT
            patient_empi::VARCHAR              AS patient_id,
            modality::VARCHAR                  AS modality,
            doc_id::VARCHAR                    AS doc_id,
            -- parse/normalize dates to DATE
            CAST(doc_date      AS DATE)        AS doc_date,
            CAST(performed_date AS DATE)       AS performed_date,
            -- some L1 may not fill text for row-based modalities
            NULLIF(raw_text, '')               AS text,
            source_tranche::VARCHAR            AS source_tranche,
            modality_specific::VARCHAR         AS meta_json
        FROM parquet_scan({l1_glob_sql}, hive_partitioning=1)
    """)
    print(f"[INFO] Layer-1 dataset loaded from {l1_glob_sql}")
    # Derive note_date, entry_id, bucket
    # ORDER BY ensures chronological order is preserved in output row groups (best effort)
    con.execute(f"""
        CREATE OR REPLACE VIEW l2 AS
        SELECT
        patient_id,
        md5(
            coalesce(patient_id,'') || '|' ||
            coalesce(doc_id,'')     || '|' ||
            coalesce(modality,'')   || '|' ||
            coalesce(cast(doc_date as varchar),'') || '|' ||
            coalesce(cast(performed_date as varchar),'')
        ) AS entry_id,
        modality,
        doc_id,
        coalesce(doc_date, performed_date) AS note_date,
        performed_date,
        NULL::VARCHAR AS section,          -- reserved, but don't use it to filter!
        text AS text,      -- <— keep entire body (IMPRESSION + FINDINGS)
        source_tranche,
        meta_json,
        mod(abs(hash(patient_id)), {n_buckets}) AS patient_bucket
        FROM l1
        ORDER BY patient_id, note_date NULLS LAST, performed_date NULLS LAST, doc_id;
    """)
    print("[INFO] Layer-2 dataset view created")
    # Write partitioned dataset (patient_bucket). Filename pattern keeps files clean.
    # Keep row groups relatively chunky by letting DuckDB batch rows.
    # 3) Write partitioned dataset (inline path; no parameters)
    copy_opts = (
        "FORMAT PARQUET, "
        "PARTITION_BY (patient_bucket), "
        "FILENAME_PATTERN 'part-{uuid}.parquet'"
    )
    if overwrite:
        copy_opts += ", OVERWRITE_OR_IGNORE"

    con.execute(f"""
        COPY (SELECT * FROM l2)
        TO {out_dir_sql_q}
        ({copy_opts})
    """)

    print(f"[OK] Layer-2 dataset written to {out}")

    # Build a small manifest for QA/resume (one row per bucket)
    # 4) Manifests
    if manifests_dir is None:
        # put them next to the dataset root
        manifests = (out.parent / "manifests")
    else:
        manifests = Path(manifests_dir)
    manifests.mkdir(parents=True, exist_ok=True)
    manifest_path = _sql_quote_path((manifests / "v1_timeline_manifest.parquet").as_posix())
    patient_index_path = _sql_quote_path((manifests / "v1_patient_index.parquet").as_posix())

    # Per-bucket stats
    con.execute(f"""
        COPY (
            SELECT
                patient_bucket,
                count(*)                   AS n_rows,
                count(DISTINCT patient_id) AS n_patients,
                min(note_date)             AS min_date,
                max(note_date)             AS max_date
            FROM parquet_scan('{out_dir_sql}/patient_bucket=*/part-*.parquet', hive_partitioning=1)
            GROUP BY 1
            ORDER BY 1
        )
        TO {manifest_path}
        (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
    """)

    # Patient→bucket index
    con.execute(f"""
        COPY (
            SELECT
                patient_id,
                mod(abs(hash(patient_id)), {n_buckets}) AS patient_bucket,
                count(*) AS n_events,
                min(note_date) AS min_date,
                max(note_date) AS max_date
            FROM parquet_scan('{out_dir_sql}/patient_bucket=*/part-*.parquet', hive_partitioning=1)
            GROUP BY 1,2
        )
        TO {patient_index_path}
        (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
    """)

    print(f"[OK] Manifests written to {manifests}")


# -----------------------------
# Loader (uses manifest lookup)
# -----------------------------
def _lookup_bucket(empi: str, idx_path="ehr_store/manifests/v1_patient_index.parquet"):
    tab = pq.read_table(idx_path, filters=[("patient_id", "=", empi)], columns=["patient_bucket"])
    if tab.num_rows == 0:
        return None
    return int(tab.column("patient_bucket")[0].as_py())

def _to_date32_scalar(x):
    if x is None:
        return None
    if isinstance(x, str):
        # expects ISO (YYYY-MM-DD). If you need to accept other formats,
        # normalize before calling this function.
        x = _dt.date.fromisoformat(x)
    elif isinstance(x, _dt.datetime):
        x = x.date()
    elif not isinstance(x, _dt.date):
        raise TypeError(f"Unsupported date type: {type(x)}")
    return pa.scalar(x, pa.date32())

def load_patient_text_from_dataset(
    patient_id: str,
    ds_root: str,
    include_modalities=None,
    exclude_after=None,
):
    dataset = ds.dataset(ds_root, format="parquet")
    filt = (ds.field("patient_id") == patient_id)

    if include_modalities:
        filt = filt & ds.field("modality").isin(include_modalities)

    if exclude_after is not None:
        date_scalar = _to_date32_scalar(exclude_after)
        filt = filt & (ds.field("note_date") <= date_scalar)

    tab = dataset.to_table(columns=["text","note_date","doc_id"], filter=filt)
    # guaranteed order for deterministic concatenation
    tab = tab.sort_by([("note_date", "ascending"), ("doc_id", "ascending")])

    parts = [t for t in tab.column("text").to_pylist() if isinstance(t, str) and t.strip() != ""]
    return "\n\n".join(parts)

if __name__ == "__main__":
    build_timeline_ds()
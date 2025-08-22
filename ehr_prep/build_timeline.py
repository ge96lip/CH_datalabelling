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
import pyarrow.compute as pc
import pyarrow.parquet as pq
# -----------------------------
# Config
# -----------------------------
# Layer-1 
L1_GLOB = r"/Users/carlotta/Desktop/Code_MT/CH_datalabelling/ehr_prep/ehr_store/normalized_v2/tranche=*/mod=*/*.parquet"
#r"ehr_store/normalized/tranche=*/mod=*/*.parquet"
# Output folder Layer-2
L2_DIR = r"ehr_store/timeline_ds_v2"
# Partitions: patient_bucket in [0..N_BUCKETS-1]
N_BUCKETS = 512
def _sql_quote_path(p: str) -> str:
    # DuckDB string literal: single quotes; escape any single quotes inside
    return "'" + p.replace("'", "''") + "'"

def build_timeline_ds(
    l1_glob: str = L1_GLOB,
    l2_dir: str = L2_DIR,
    n_buckets: int = N_BUCKETS,
    overwrite: bool = True,
    manifests_dir: Optional[str] = None,
) -> None:
    """
    Build Layer-2 (timeline) Parquet dataset from Layer-1 normalized Parquet.

    This version is aligned to the L1 schema:
      - Uses columns written in L1: raw_text, line_start, line_end, etc. (no row_text)
      - entry_id is md5(patient_id|doc_id|modality|line_start|line_end) for true uniqueness
      - Forces ZSTD compression on older DuckDB via force_compression
    """

    out = Path(l2_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA preserve_insertion_order=false")   # better pipelining
    con.execute("PRAGMA threads=4")                        # fewer threads => fewer big hash tables
    con.execute("PRAGMA memory_limit='4GB'")               # or '8GB' if given
    # con.execute("PRAGMA temp_directory='/tmp/duckdb_tmp'") # pick a disk with space
    # con.execute("PRAGMA max_temp_directory_size='200GB'")  # adjust to available disk
    # con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 4) - 1)}")

    # Resolve paths to absolute POSIX; quote for SQL
    l1_glob_path = Path(l1_glob).resolve().as_posix()
    l1_glob_sql  = _sql_quote_path(l1_glob_path)
    out_dir_sql  = out.as_posix()
    out_dir_sql_q = _sql_quote_path(out_dir_sql)

    # Your DuckDB version: use 'force_compression' to set parquet codec
    con.execute("SET force_compression='zstd'")

    # ---------- L1 view ----------
    # Columns available from normalizers:
    # source_tranche, source_file, line_start, line_end,
    # patient_empi, patient_mrn, modality, doc_id, report_sequence,
    # is_free_text, raw_text, report_end_marker, doc_date, performed_date,
    # ingest_timestamp, section_hints, modality_specific
    con.execute(f"""
        CREATE OR REPLACE VIEW l1_raw AS
        SELECT
            patient_empi::VARCHAR        AS patient_id,
            modality::VARCHAR            AS modality,
            doc_id::VARCHAR              AS doc_id,
            CAST(doc_date AS DATE)       AS doc_date,
            CAST(performed_date AS DATE) AS performed_date,
            line_start,
            line_end,
            NULLIF(raw_text, '')         AS text,
            source_tranche::VARCHAR      AS source_tranche,
            modality_specific::VARCHAR   AS meta_json,
            hospital,
            inpatient_outpatient       AS inout,
        FROM parquet_scan({l1_glob_sql}, hive_partitioning=1);
    """)
    print(f"[INFO] L1 loaded from {l1_glob_path}")

    # Basic visibility on null patient IDs (helps explain patient_count failures)
    null_pid, total_l1 = con.execute("SELECT SUM(patient_id IS NULL), COUNT(*) FROM l1_raw").fetchone()
    print(f"[INFO] L1 rows: {total_l1:,} | NULL patient_id: {null_pid:,}")

    # ---------- L2 view ----------
    #con.execute(f
    """
        CREATE OR REPLACE VIEW l1_enriched AS
        SELECT
            patient_id,
            modality,
            doc_id,
            doc_date,
            performed_date,
            line_start,
            line_end,
            text,
            source_tranche,
            meta_json,
            COALESCE(json_extract_string(meta_json, '$.MRN_Type'))             AS hospital_l2,
            COALESCE(json_extract_string(meta_json, '$.Inpatient_Outpatient')) AS inout_l2
        FROM l1_raw;
    """#)
    # entry_id must uniquely identify an atomic entry → include row discriminators (line_start, line_end)
    con.execute(f"""
        CREATE OR REPLACE VIEW l2 AS
        SELECT
            patient_id,
            md5(
                COALESCE(patient_id,'') || '|' ||
                COALESCE(doc_id,'')     || '|' ||
                COALESCE(modality,'')   || '|' ||
                COALESCE(CAST(line_start AS VARCHAR),'') || '|' ||
                COALESCE(CAST(line_end   AS VARCHAR),'')
            ) AS entry_id,
            modality,
            doc_id,
            COALESCE(doc_date, performed_date) AS note_date,
            performed_date,
            NULL::VARCHAR AS section,  -- reserved
            text,
            source_tranche,
            meta_json,
            hospital,
            inout,
            MOD(ABS(HASH(patient_id)), {n_buckets}) AS patient_bucket
        FROM l1_raw
        WHERE patient_id IS NOT NULL
    """)
    print("[INFO] L2 view created")

    # ---------- Write L2 (partitioned by patient_bucket, ZSTD) ----------
    copy_opts = (
        "FORMAT PARQUET, "
        "PARTITION_BY (patient_bucket), "
        "FILENAME_PATTERN 'part-{uuid}.parquet', "
        "COMPRESSION 'zstd'"
    )
    if overwrite:
        copy_opts += ", OVERWRITE_OR_IGNORE"
    for b in range(n_buckets):
        con.execute(f"""
            COPY (
                SELECT * FROM l2
                WHERE patient_bucket = {b}
            ) TO {out_dir_sql_q} ({copy_opts})
        """)
    print(f"[OK] Layer-2 written to {out_dir_sql}")

    # ---------- Manifests ----------
    manifests = Path(manifests_dir).resolve() if manifests_dir else (out.parent / "manifests")
    manifests.mkdir(parents=True, exist_ok=True)
    timeline_manifest_path = _sql_quote_path((manifests / "v1_timeline_manifest.parquet").as_posix())
    patient_index_path     = _sql_quote_path((manifests / "v1_patient_index.parquet").as_posix())

    # Per-bucket stats (kept simple; you can add size/row_group stats later)
    con.execute(f"""
        COPY (
            SELECT
                patient_bucket,
                COUNT(*)                   AS n_rows,
                COUNT(DISTINCT patient_id) AS n_patients,
                MIN(note_date)             AS min_date,
                MAX(note_date)             AS max_date, 
            FROM parquet_scan('{out_dir_sql}/patient_bucket=*/part-*.parquet', hive_partitioning=1)
            GROUP BY 1
            ORDER BY 1
        )
        TO {timeline_manifest_path}
        (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
    """)

    # Patient → bucket index
    con.execute(f"""
        COPY (
            SELECT
                patient_id,
                MOD(ABS(HASH(patient_id)), {n_buckets}) AS patient_bucket,
                COUNT(*)       AS n_events,
                MIN(note_date) AS min_date,
                MAX(note_date) AS max_date
            FROM parquet_scan('{out_dir_sql}/patient_bucket=*/part-*.parquet', hive_partitioning=1)
            GROUP BY 1,2
        )
        TO {patient_index_path}
        (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
    """)
    print(f"[OK] Manifests written to {manifests.as_posix()}")


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
    def count_null_hospital(parquet_dir: str) -> int:
        files = Path(parquet_dir).glob("*.parquet")
        total_nulls = 0
        for f in files:
            tbl = pq.read_table(f, columns=["hospital"])  # only load hospital
            total_nulls += pc.sum(pc.is_null(tbl["hospital"])).as_py()
        return total_nulls

    print("NULL hospital count:", count_null_hospital("/Users/carlotta/Desktop/Code_MT/CH_datalabelling/ehr_prep/ehr_store/timeline_ds_v2"))
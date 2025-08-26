from concurrent.futures import ProcessPoolExecutor, as_completed
import os, logging, json, duckdb, sys

from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional
from pathlib import PureWindowsPath, Path 
from logging.handlers import RotatingFileHandler
# Layer-1 
L1_GLOB = r"Z:/JonathanMueller/llmProject_20250627/ehr_store/normalized_v1/tranche=*/mod=*/*.parquet"
#r"ehr_store/normalized/tranche=*/mod=*/*.parquet"
# Output folder Layer-2
L2_DIR = r"Z:/JonathanMueller/llmProject_20250627/ehr_store/timeline_ds_v2"
# Partitions: patient_bucket in [0..N_BUCKETS-1]
N_BUCKETS = 512
def _setup_logging(log_dir: Path, name: str, to_console: bool = True) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers if reconfigured
    if logger.handlers:
        return logger

    log_path = log_dir / f"{name}.log"
    fh = RotatingFileHandler(log_path, maxBytes=50_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(process)d | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    ))
    logger.addHandler(fh)

    if to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)

    return logger

def _marker_path(out_root: Path, tranche: str, bucket: int) -> Path:
    # Separate “control plane” so it never collides with data files
    return out_root / "_done" / f"tranche={tranche}" / f"bucket={bucket}.done.json"

def _shard_already_done(out_root: Path, tranche: str, bucket: int) -> bool:
    return _marker_path(out_root, tranche, bucket).exists()

def _l1_glob_for_tranche(l1_glob: str, tranche: str) -> str: 
    p = PureWindowsPath(l1_glob).as_posix()
    return p.replace('tranche=*', f'tranche={tranche}')

def _discover_tranches_from_glob(l1_glob: str) -> List[str]:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")

    # Convert to forward slashes; DO NOT resolve() a glob
    l1_glob_posix = PureWindowsPath(l1_glob).as_posix()

    con.execute(f"""
        SELECT DISTINCT source_tranche
        FROM parquet_scan({_sql_quote_path(l1_glob_posix)}, hive_partitioning=1)
        WHERE source_tranche IS NOT NULL
        ORDER BY 1
    """)
    return [r[0] for r in con.fetchall()]


def _ensure_duckdb_session(threads: int, temp_dir: Optional[str] = None):
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, threads)}")
    if temp_dir:
        con.execute(f"SET temp_directory='{Path(temp_dir).resolve().as_posix()}'")
    return con

def _sql_quote_path(p: str) -> str:
    # DuckDB string literal: single quotes; escape any single quotes inside
    return "'" + p.replace("'", "''") + "'"

def _write_done_marker(out_root: Path, tranche: str, bucket: int, extra: Optional[Dict] = None):
    mpath = _marker_path(out_root, tranche, bucket)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tranche": tranche,
        "bucket": bucket,
        "written_at": datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        payload.update(extra)
    mpath.write_text(json.dumps(payload))

def _has_any_rows_for_shard(out_root: Path, tranche: str, bucket: int) -> bool:
    con = duckdb.connect()
    try:
        out_posix = out_root.resolve().as_posix()
        # Only read the specific partition we just wrote, not all buckets
        q = f"""
          SELECT 1
          FROM parquet_scan('{out_posix}/patient_bucket={bucket}/part-*.parquet', hive_partitioning=1)
          WHERE source_tranche = ?
          LIMIT 1
        """
        return con.execute(q, (tranche,)).fetchone() is not None
    except Exception:
        return False
    finally:
        con.close()
    
def _create_views(con: duckdb.DuckDBPyConnection, l1_glob: str, n_buckets: int, tranche: Optional[str] = None, mem_gb: int = 2):
    l1_glob_posix = _l1_glob_for_tranche(l1_glob, tranche) if tranche else PureWindowsPath(l1_glob).as_posix()
    con.execute("SET enable_progress_bar=false")
    con.execute(f"PRAGMA memory_limit='{mem_gb}GB'")        # lower per-conn cap (see B)
    con.execute("PRAGMA threads=1")
    con.execute("SET preserve_insertion_order=false")

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
            inpatient_outpatient         AS inout
        FROM parquet_scan({_sql_quote_path(l1_glob_posix)}, hive_partitioning=1);
    """)

    # Full (with text)
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
            NULL::VARCHAR AS section,
            text,                               -- heavy!
            source_tranche,
            meta_json,
            hospital,
            inout,
            MOD(ABS(HASH(patient_id)), {n_buckets}) AS patient_bucket
        FROM l1_raw
        WHERE patient_id IS NOT NULL;
    """)

    # Keys-only (no text) for cheap probes
    con.execute(f"""
        CREATE OR REPLACE VIEW l2_keys AS
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
            source_tranche,
            MOD(ABS(HASH(patient_id)), {n_buckets}) AS patient_bucket
        FROM l1_raw
        WHERE patient_id IS NOT NULL;
    """)

    con.execute("""
        CREATE OR REPLACE VIEW l2_distinct AS
        SELECT *
        FROM l2
        QUALIFY ROW_NUMBER() OVER (PARTITION BY entry_id ORDER BY source_tranche) = 1;
    """)

def _write_failed_marker(out_root: Path, tranche: str, bucket: int, err: str): 
    mpath = _marker_path(out_root, tranche, bucket)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'tranche': tranche, 
        'bucket': bucket, 
        'failed_at': datetime.utcnow().isoformat() + 'Z', 
        'error': err[:8000],
    }
    mpath.with_suffix('.failed.json').write_text(json.dumps(payload))

def _write_heartbeat(out_root: Path, tranche: str, bucket: int): 
    mpath = _marker_path(out_root, tranche, bucket)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    (mpath.parent / f'bucket={bucket}.heartbeat').write_text(datetime.utcnow().isoformat()+'Z')

def _write_shard(l1_glob: str, out_dir: str, tranche: str, bucket: int, n_buckets: int, threads: int, temp_dir: Optional[str] = None):
    """
    One shard = one (tranche, bucket). Independent DuckDB session.
    """
    # print(f"[START] shard tranche={tranche} bucket={bucket}")

    logger = _setup_logging(Path(out_dir) / "_logs", name=f"worker-{os.getpid()}", to_console=False)
    if temp_dir and not Path(temp_dir).exists(): 
        raise RuntimeError(f"tempd_dir does not exists: {temp_dir}")
    con = _ensure_duckdb_session(threads=threads, temp_dir=temp_dir)
    try: 
        _write_heartbeat(Path(out_dir), tranche, bucket)
        _create_views(con, l1_glob=l1_glob, n_buckets=n_buckets, tranche=tranche)

        # quick existence check before COPY (cheap)
        exists = con.execute(
            """
            SELECT 1 FROM l2_keys
            WHERE source_tranche = ? AND patient_bucket = ?
            LIMIT 1
            """,
            (tranche, bucket),
        ).fetchone() is not None

        if not exists:
            _write_done_marker(Path(out_dir), tranche, bucket, {"n_rows": 0})
            print(f"[SKIP-EMPTY] shard tranche={tranche} bucket={bucket}")
            logger.info(f"[SKIP-EMPTY] tranche=%s bucket=%d", tranche, bucket)
            return

        copy_sql = f"""
            COPY (
                SELECT *
                FROM l2_distinct
                WHERE source_tranche = ?
                AND patient_bucket = ?
                ORDER BY patient_id, note_date NULLS LAST, performed_date NULLS LAST,
                        entry_id
            ) TO '{PureWindowsPath(out_dir).as_posix()}'
            (FORMAT PARQUET,
            PARTITION_BY (patient_bucket),
            FILENAME_PATTERN 'part-{{uuid}}.parquet',
            COMPRESSION 'zstd',
            OVERWRITE_OR_IGNORE TRUE)
        """
        con.execute(copy_sql, (tranche, bucket))

        # validate readable + count files now
        ok = _has_any_rows_for_shard(Path(out_dir), tranche, bucket)
        files_now = list((Path(out_dir) / f"patient_bucket={bucket}").glob("part-*.parquet"))
        _write_done_marker(
            Path(out_dir), tranche, bucket,
            {"n_rows": 1 if ok else 0, "n_files_in_bucket_after": len(files_now)}
        )
        logger.info("[DONE] tranche=%s bucket=%d files_in_bucket=%d ok=%s",
                    tranche, bucket, len(files_now), ok)
        print(f"[DONE] shard tranche={tranche} bucket={bucket} files_in_bucket={len(files_now)}")
    except Exception as e: 
        _write_failed_marker(Path(out_dir), tranche, bucket, repr(e))
        logger.exception("Shard failed: tranche=%s bucket=%d", tranche, bucket)
        raise 
    finally: 
        try: 
            con.close()
        except Exception: 
            pass

def build_timeline_ds_parallel(
    l1_glob: str = L1_GLOB,
    l2_dir: str = L2_DIR,
    n_buckets: int = N_BUCKETS,
    max_workers: Optional[int] = None,
    temp_dir: Optional[str] = None,     # e.g. '/tmp/duckdb_tmp' on a fast local SSD
    overwrite: bool = True,             # if False, we’ll **skip** existing shards
    per_conn_threads: int = 1,           # threads used inside each DuckDB worker
    tranche_whitelist: Optional[List[str]] = None,
    bucket_range: Optional[range] = None,
):
    out = Path(l2_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f'[INFO] maxworker: {max_workers}')
    logger = _setup_logging(out / "_logs", name="run", to_console=True)

    tranches = _discover_tranches_from_glob(l1_glob)
    print(f'[INFO] discovered tranches: {tranches}')
    if tranche_whitelist:
        tranches = [t for t in tranches if t in tranche_whitelist]

    if not tranches:
        logger.warning("No tranches to process.")
        print("[WARN] No tranches discovered from L1 glob.")
        return
    
    # Plan all shards
    plan = []
    for t in tranches:
        for b in range(n_buckets):
            if not overwrite and _shard_already_done(out, t, b):
                print(f"[SKIP-DONE] tranche={t} bucket={b}")
                logger.info("[SKIP-DONE] tranche=%s bucket=%d", t, b)
                continue
            plan.append((t, b))
    
    by_bucket = defaultdict(deque)
    for t, b in plan: 
        by_bucket[b].append(t)
    
    workers = max_workers or max(1, (os.cpu_count() or 4) - 1)

    print(f"[INFO] Planned shards: {len(plan)}  (buckets={n_buckets}, tranches={len(tranches)})")
    print(f"[INFO] Will process {len(plan)} shards across {workers} workers "
          f"(per-conn threads={per_conn_threads}).")

    logger.info("Selected tranches=%s", tranches)
    logger.info("Planned shards=%d (buckets=%d, tranches=%d) workers=%d per_conn_threads=%d",
                len(plan), n_buckets, len(tranches), workers, per_conn_threads)

    MAX_INFLIGHT = max(8, workers*2)
    active_futs = {}
    active_buckets = set()

    def submit_next_for_bucket(ex, bucket): 
        q = by_bucket.get(bucket)
        if not q:
            return False
        tranche = q.popleft()
        fut = ex.submit(
            _write_shard,
            l1_glob, out.as_posix(), tranche, bucket, n_buckets,
            per_conn_threads, temp_dir
        )
        active_futs[fut] = bucket
        active_buckets.add(bucket)
        return True
    with ProcessPoolExecutor(max_workers=workers) as ex:
        # seed: up to MIN(MAX_INFLIGHT, workers, #buckets) distinct buckets
        for bucket in list(by_bucket.keys()):
            if len(active_futs) >= min(MAX_INFLIGHT, workers, len(by_bucket)):
                break
            submit_next_for_bucket(ex, bucket)

        done = 0
        total = len(plan)
        while active_futs:
            # Wait for any one to finish
            for f in as_completed(list(active_futs.keys()), timeout=None):
                bucket = active_futs.pop(f)
                active_buckets.discard(bucket)
                # Propagate errors here; your worker already logs/markers on failure
                f.result()
                done += 1
                if done % 20 == 0:
                    logger.info("[PROGRESS] %d/%d shards complete", done, total)
                    print(f"[PROGRESS] {done}/{total} shards complete")

                # Submit next tranche for this *same* bucket (serial within bucket)
                submitted = submit_next_for_bucket(ex, bucket)

                # If we still have capacity (< MAX_INFLIGHT) submit more from *other* buckets
                if not submitted:
                    # try to find another bucket not currently active
                    for b2 in list(by_bucket.keys()):
                        if b2 in active_buckets:
                            continue
                        if submit_next_for_bucket(ex, b2):
                            break

                # Break out of the inner for-loop to re-evaluate active_futs
                break

    logger.info("Finished writing Layer-2 to %s", out.as_posix())
    print(f"[OK] Finished writing Layer-2 to {out.as_posix()}")


def build_manifests(l2_dir: str, n_buckets: int, manifests_dir: Optional[str] = None):
    out = Path(l2_dir).resolve()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")

    manifests = Path(manifests_dir).resolve() if manifests_dir else (out.parent / "manifests")
    manifests.mkdir(parents=True, exist_ok=True)
    timeline_manifest_path = _sql_quote_path((manifests / "v1_timeline_manifest.parquet").as_posix())
    patient_index_path     = _sql_quote_path((manifests / "v1_patient_index.parquet").as_posix())

    con.execute(f"""
        COPY (
            SELECT
                patient_bucket,
                COUNT(*)                   AS n_rows,
                COUNT(DISTINCT patient_id) AS n_patients,
                MIN(note_date)             AS min_date,
                MAX(note_date)             AS max_date
            FROM parquet_scan('{out.as_posix()}/patient_bucket=*/part-*.parquet', hive_partitioning=1)
            GROUP BY 1
            ORDER BY 1
        )
        TO {timeline_manifest_path}
        (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
    """)

    con.execute(f"""
        COPY (
            SELECT
                patient_id,
                MOD(ABS(HASH(patient_id)), {n_buckets}) AS patient_bucket,
                COUNT(*)                    AS n_events,
                MIN(note_date)              AS min_date,
                MAX(note_date)              AS max_date
            FROM parquet_scan('{out.as_posix()}/patient_bucket=*/part-*.parquet', hive_partitioning=1)
            GROUP BY 1,2
        )
        TO {patient_index_path}
        (FORMAT PARQUET, OVERWRITE_OR_IGNORE TRUE)
    """)
    print(f"[OK] Manifests written to {manifests.as_posix()}")

def count_null_hospital_duck(l2_dir: str) -> int:
    con = duckdb.connect()
    q = f"""
      SELECT SUM(hospital IS NULL)::BIGINT
      FROM parquet_scan('{Path(l2_dir).resolve().as_posix()}/patient_bucket=*/part-*.parquet', hive_partitioning=1)
    """
    return con.execute(q).fetchone()[0]

def compute_patient_bucket(patient_id: str, n_buckets: int = N_BUCKETS) -> int:
    con = duckdb.connect()
    try:
        return con.execute("SELECT MOD(ABS(HASH(?)), ?)", (patient_id, n_buckets)).fetchone()[0]
    finally:
        con.close()

if __name__ == "__main__":
    # 1) Build (resumable + parallel)

    build_timeline_ds_parallel(
        l1_glob=L1_GLOB,
        l2_dir=L2_DIR,
        n_buckets=N_BUCKETS,
        max_workers= 4, # max(1, (os.cpu_count() or 4) - 1),  # process-level parallelism
        per_conn_threads=1,                              # threads *inside* each DuckDB conn
        temp_dir=r"H:\temp_ehr_store",                      # put this on a fast local SSD
        overwrite=False,                                  # skip shards that already exist
        tranche_whitelist=["T01", "T02"],  # use your actual tranche labels
        bucket_range=range(0, N_BUCKETS),
    )

    # 2) Then compute manifests over whatever is present
    build_manifests(L2_DIR, N_BUCKETS)
    # hospital-NULL counter 
    print("NULL hospital count:", count_null_hospital_duck(L2_DIR))
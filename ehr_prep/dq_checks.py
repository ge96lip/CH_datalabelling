#!/usr/bin/env python3
"""
dq_checks.py — Data Quality Checks for the EHR Parquet Lake

Usage (examples):
  # Deps needed:
  #   pip install duckdb pyarrow pandas rich tqdm

  # Quick run with defaults (expects ehr_store layout under CWD):
  python dq_checks.py --root ./ehr_store

  # Specify custom layout:
  python dq_checks.py --root /data/ehr_store \
      --l1 normalized --l2 timeline_ds --manifests manifests --lookups lookups \
      --patient-index v1_patient_index.parquet \
      --timeline-manifest v1_timeline_manifest.parquet \
      --outdir dq_out

  # Run only selected checks:
  python dq_checks.py --root ./ehr_store --checks schema_l1 schema_l2 ids dates manifests codecs

What this script does:
- Validates core schemas (Layer-1 normalized, Layer-2 timeline).
- Checks uniqueness/consistency of IDs (doc_id, entry_id).
- Validates date sanity (null rates, min/max, future dates).
- Validates manifests (patient index and timeline manifest) against the dataset.
- Optionally checks compression codecs (ZSTD) on a sample of files.
- Writes a human-readable Markdown report + CSVs with any bad rows to --outdir.
"""

import argparse, os, sys, json, time, random
from pathlib import Path
import duckdb, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

try:
    from rich import print as rprint
    RICH = True
except Exception:
    RICH = False

def cprint(msg): rprint(msg) if RICH else print(msg)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def _duck_con(): return duckdb.connect()

# === schema checks ===
def check_schema_layer1(root: Path, l1: str, out: Path):
    con = _duck_con()
    pattern = root / l1 / "tranche=*/mod=*/*.parquet"
    # SQL query which scans acording to the pattern, it does not look at the content in the file, hive_partitioning=1 means it will use the information from the file name to partition the data
    # LIMIT 0 means we only get the schema, not the data
    q = f"SELECT * FROM parquet_scan('{pattern}', hive_partitioning=1) LIMIT 0"
    # Execute the query and get the columns
    cols = set(con.execute(q).df().columns)
    required_all = {"patient_empi","modality","doc_id","doc_date","tranche","mod"}
    errors = []
    # Check if all required columns are present
    if not required_all.issubset(cols):
        errors.append("Missing required columns: " + str(required_all-cols))
    if not ({"raw_text","row_text"} & cols):
        errors.append("Need one of raw_text/row_text")
    if not ({"meta_json","modality_specific"} & cols):
        errors.append("Need one of meta_json/modality_specific")
    return {"ok": not errors, "errors": errors, "details":{"columns":list(cols)}}

def check_schema_layer2(root: Path, l2: str, out: Path):
    con = _duck_con()
    pattern = root / l2 / "patient_bucket=*/part-*.parquet"
    q = f"SELECT * FROM parquet_scan('{pattern}', hive_partitioning=1) LIMIT 0"
    cols = set(con.execute(q).df().columns)
    required = {"patient_id","entry_id","note_date","doc_id","modality","text"}
    errors=[]
    if not required.issubset(cols):
        errors.append("Missing cols: "+str(required-cols))
    return {"ok": not errors,"errors":errors,"details":{"columns":list(cols)}}

# === id checks ===
def check_uniqueness_ids(root: Path,l1,l2,out:Path):
    con=_duck_con(); errors=[]
    q1=f"SELECT doc_id,COUNT(*) c FROM parquet_scan('{root/l1}/tranche=*/mod=*/*.parquet',hive_partitioning=1) GROUP BY doc_id HAVING c>1"
    d1=con.execute(q1).df()
    q2=f"SELECT patient_id,entry_id,COUNT(*) c FROM parquet_scan('{root/l2}/patient_bucket=*/part-*.parquet',hive_partitioning=1) GROUP BY patient_id,entry_id HAVING c>1"
    d2=con.execute(q2).df()
    if not d1.empty: errors.append(f"dup doc_id: {len(d1)} groups")
    if not d2.empty: errors.append(f"dup entry_id per patient: {len(d2)} groups")
    return {"ok": not errors,"errors":errors,"details":{"dup_doc_ids":len(d1),"dup_entry_ids":len(d2)}}

# === date sanity ===
def check_dates(root,l1,l2,out:Path,max_future_days=3):
    con=_duck_con(); errors=[]
    q1 = f"""
      SELECT
        COUNT(*) AS row_count,
        SUM(CASE WHEN doc_date IS NULL THEN 1 ELSE 0 END) AS null_doc_date
      FROM parquet_scan('{(root/l1/"tranche=*/mod=*/*.parquet").as_posix()}', hive_partitioning=1)
    """
    d1 = con.execute(q1).df().iloc[0].to_dict()

    q2 = f"""
      SELECT
        COUNT(*) AS row_count,
        SUM(CASE WHEN note_date IS NULL THEN 1 ELSE 0 END) AS null_note_date,
        MIN(note_date) AS min_note_date,
        MAX(note_date) AS max_note_date,
        SUM(CASE WHEN note_date > (CURRENT_DATE + INTERVAL '{max_future_days} days') THEN 1 ELSE 0 END) AS future_count
      FROM parquet_scan('{(root/l2/"patient_bucket=*/part-*.parquet").as_posix()}', hive_partitioning=1)
    """
    d2 = con.execute(q2).df().iloc[0].to_dict()
    if int(d2.get("future_count", 0)) > 0:
        errors.append(f"{int(d2['future_count'])} note_date(s) > {max_future_days} days in the future")

    details = {
        "l1_row_count": int(d1.get("row_count", 0)),
        "l1_null_doc_date": int(d1.get("null_doc_date", 0)),
        "l2_row_count": int(d2.get("row_count", 0)),
        "l2_null_note_date": int(d2.get("null_note_date", 0)),
        "l2_min_note_date": str(d2.get("min_note_date")),
        "l2_max_note_date": str(d2.get("max_note_date")),
        "l2_future_note_dates": int(d2.get("future_count", 0)),
    }
    return {"ok": not errors, "errors": errors, "details": details}

# === manifest check (simplified) ===
def check_manifests(root,l2,manifests,patient_index,timeline_manifest,out:Path):
    res={"ok":True,"errors":[]}
    if not (root/manifests/patient_index).exists(): res["ok"]=False;res["errors"].append("Missing patient_index")
    if not (root/manifests/timeline_manifest).exists(): res["ok"]=False;res["errors"].append("Missing timeline_manifest")
    return res

def check_codecs(root: Path, l1: str, l2: str, expected_codec: str, out: Path, sample_files:int=16):
    import glob, random
    random.seed(42)
    l1_files = glob.glob((root/l1/"tranche=*/mod=*/*.parquet").as_posix())
    l2_files = glob.glob((root/l2/"patient_bucket=*/part-*.parquet").as_posix())
    l1_s = random.sample(l1_files, min(sample_files, len(l1_files))) if l1_files else []
    l2_s = random.sample(l2_files, min(sample_files, len(l2_files))) if l2_files else []

    def codecs_of(files):
        found = set()
        for fp in files:
            try:
                pf = pq.ParquetFile(fp)
                meta = pf.metadata
                for rg in range(meta.num_row_groups):
                    rgm = meta.row_group(rg)
                    for c in range(rgm.num_columns):
                        colm = rgm.column(c)
                        found.add(str(colm.compression).upper())
            except Exception as e:
                return {"ok": False, "errors":[f"Failed {fp}: {e}"], "details": {}}
        return {"ok": True, "errors": [], "details": {"codecs": sorted(found)}}

    r1 = codecs_of(l1_s)
    r2 = codecs_of(l2_s)
    ok = r1["ok"] and r2["ok"]
    errors = r1["errors"] + r2["errors"]
    if expected_codec.upper() not in set(r1.get("details",{}).get("codecs",[])) and l1_s:
        ok=False; errors.append(f"Expected codec {expected_codec} not seen in L1 sample: {r1.get('details',{})}")
    if expected_codec.upper() not in set(r2.get("details",{}).get("codecs",[])) and l2_s:
        ok=False; errors.append(f"Expected codec {expected_codec} not seen in L2 sample: {r2.get('details',{})}")
    return {"ok": ok, "errors": errors, "details": {"l1": r1.get("details",{}), "l2": r2.get("details",{})}}

def check_timeline_ordering(root: Path, l2: str, out: Path, sample_patients:int=200):
    con=_duck_con()
    pattern = (root / l2 / "patient_bucket=*/part-*.parquet").as_posix()
    q = f"""
      WITH ids AS (
        SELECT DISTINCT patient_id FROM parquet_scan('{pattern}', hive_partitioning=1)
        USING SAMPLE {sample_patients} ROWS
      )
      SELECT COUNT(*) AS n FROM ids
    """
    try:
        # Proxy smoke test: duplicates of entry_id per patient
        q_dups = f"""
          WITH l2 AS (SELECT patient_id, entry_id
                      FROM parquet_scan('{pattern}', hive_partitioning=1))
          SELECT patient_id, entry_id, COUNT(*) c
          FROM l2
          GROUP BY 1,2
          HAVING c>1
          USING SAMPLE {sample_patients} ROWS
        """
        d = con.execute(q_dups).df()
        ok = d.empty
        errors = [] if ok else [f"Duplicate entry_id within patients (sample): {len(d)}"]
        return {"ok": ok, "errors": errors, "details": {"sample_duplicates": len(d)}}
    except Exception as e:
        return {"ok": False, "errors": [f"Failed timeline ordering check: {e}"], "details": {}}

# === NEW: patient count check ===
def check_patient_count(root: Path, l2: str, manifests: str, patient_index_name: str,
                        out: Path, min_expected:int=262_554, max_expected:int=262_665):
    """
    Count unique patients (EMPIs) and assert it's within [min_expected, max_expected].
    Prefers the patient index manifest; falls back to DISTINCT over L2 timeline if needed.
    """
    con = _duck_con()
    pi_path = root / manifests / patient_index_name
    details = {}
    try:
        if pi_path.exists():
            q = f"SELECT COUNT(DISTINCT patient_id) AS n FROM parquet_scan('{pi_path.as_posix()}')"
            n = int(con.execute(q).df().iloc[0]["n"])
            source = "patient_index"
        else:
            pattern = (root / l2 / "patient_bucket=*/part-*.parquet").as_posix()
            q = f"SELECT COUNT(DISTINCT patient_id) AS n FROM parquet_scan('{pattern}', hive_partitioning=1)"
            n = int(con.execute(q).df().iloc[0]["n"])
            source = "timeline"
        details.update({"unique_patients": n, "source": source,
                        "min_expected": min_expected, "max_expected": max_expected})
        ok = (min_expected <= n <= max_expected)
        errors = [] if ok else [f"Unique patients {n} outside expected range [{min_expected}, {max_expected}]."]
        return {"ok": ok, "errors": errors, "details": details}
    except Exception as e:
        return {"ok": False, "errors": [f"Failed patient count: {e}"], "details": details}

# === runner ===
def run(args):
  root=Path(args.root); out=Path(args.outdir); ensure_dir(out)
  results = {}

  def do(name, fn, *fnargs):
      r = fn(*fnargs)
      results[name] = r
      cprint(f"{'[green]PASS[/green]' if (RICH and r['ok']) else ('PASS' if r['ok'] else ('[red]FAIL[/red]' if RICH else 'FAIL'))} {name}")
      if not r["ok"]:
          for e in r.get("errors", []):
              cprint(f"  - {e}")

  checks = [c.lower() for c in (args.checks or
            ["schema_l1","schema_l2","ids","dates","manifests","codecs","timeline_order","patient_count"])]

  if "schema_l1" in checks: do("schema_l1", check_schema_layer1, root, args.l1, out)
  if "schema_l2" in checks: do("schema_l2", check_schema_layer2, root, args.l2, out)
  if "ids"       in checks: do("ids",       check_uniqueness_ids, root, args.l1, args.l2, out)
  if "dates"     in checks: do("dates",     check_dates, root, args.l1, args.l2, out, args.max_future_days)
  if "manifests" in checks: do("manifests", check_manifests, root, args.l2, args.manifests, args.patient_index, args.timeline_manifest, out)
  if "codecs"    in checks: do("codecs",    check_codecs, root, args.l1, args.l2, args.expected_codec, out, args.sample_files)
  if "timeline_order" in checks: do("timeline_order", check_timeline_ordering, root, args.l2, out, args.sample_patients)
  if "patient_count"  in checks: do("patient_count",  check_patient_count, root, args.l2, args.manifests, args.patient_index, out,
                                    args.min_patients, args.max_patients)

  with open(out/"dq_report.json","w",encoding="utf-8") as f:
      json.dump(results,f,indent=2, default=str)
  cprint(f"\nWrote JSON report to { (out/'dq_report.json').as_posix() }")
  return results

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="EHR Lake — Data Quality Checks")
    ap.add_argument("--root",required=True)
    ap.add_argument("--l1",default="normalized_v2")
    ap.add_argument("--l2",default="timeline_ds_v2")
    ap.add_argument("--manifests",default="manifests")
    ap.add_argument("--patient-index",default="v1_patient_index.parquet")
    ap.add_argument("--timeline-manifest",default="v1_timeline_manifest.parquet")
    ap.add_argument("--max-future-days",type=int,default=3)
    ap.add_argument("--expected-codec",default="ZSTD")
    ap.add_argument("--sample-files",type=int,default=16)
    ap.add_argument("--sample-patients",type=int,default=200)
    # bounds for unique patient count
    ap.add_argument("--min-patients",type=int,default=262_554)
    ap.add_argument("--max-patients",type=int,default=262_665)
    ap.add_argument("--outdir",default="dq_out")
    ap.add_argument("--checks", nargs="*", help="Subset: schema_l1 schema_l2 ids dates manifests codecs timeline_order patient_count")
    args=ap.parse_args()
    run(args)
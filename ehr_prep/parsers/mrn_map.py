# ehr_prep/parsers/mrn_map.py
import csv, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from typing import Optional
from typing import List

def _norm_mrn(s: Optional[str]):
    if not s: return None
    s = s.upper()
    for pref in ("EPIC-","EPIC-EPSI-","TSI-","EPIC-MGH-","EPIC-BWH-"):
        if s.startswith(pref): s = s[len(pref):]
    s = ''.join(ch for ch in s if ch.isalnum())
    if s.isdigit(): s = s.lstrip("0") or "0"
    return s

def build_mrn_lookup(mrn_files: List[str], out_parquet: str):
    rows = []
    for p in mrn_files:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            rdr = csv.DictReader(f, delimiter="|")
            for r in rdr:
                empi = r.get("Enterprise_Master_Patient_Index") or r.get("EMPI") or r.get("Enterprise_Master_Patient_Index".upper())
                if not empi: continue
                for k, v in r.items():
                    if not v: continue
                    if k.endswith("_MRN") or k in ("EPIC_PMRN","MGH_MRN","BWH_MRN","FH_MRN","SRH_MRN","NWH_MRN","NSMC_MRN","MCL_MRN","MEE_MRN","DFC_MRN","WDH_MRN"):
                        rows.append({"empi": empi, "mrn_raw": v, "mrn_norm": _norm_mrn(v), "source_file": Path(p).name})
    tbl = pa.Table.from_pylist(rows, schema=pa.schema([
        ("empi", pa.string()), ("mrn_raw", pa.string()), ("mrn_norm", pa.string()), ("source_file", pa.string())
    ]))
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out_parquet, compression="zstd", version="2.6")
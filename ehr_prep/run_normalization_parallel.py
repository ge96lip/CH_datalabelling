# run_normalization_parallel.py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from tqdm import tqdm
import re 
# local imports
from parsers.mrn_map import build_mrn_lookup
from normalize import normalize_tranche

BASE = Path(r"Z:\JonathanMueller\llmProject_20250627\data")
OUT  = Path(r"Z:\JonathanMueller\llmProject_20250627\ehr_store\normalized")  # or relative
TRANCHE_RE = re.compile(r"llmTranche(\d+)_raw", flags=re.IGNORECASE)

def discover_tranches(base: Path):
    """Return [(path, Txx)] for tranches 1..11, skipping T01 explicitly."""
    raw_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    jobs = []
    for p in raw_dirs:
        m = TRANCHE_RE.search(p.name)
        if not m:
            continue
        num = int(m.group(1))
        tranche_code = f"T{num:02d}"
        #if num == 1:                        # explicit skip of tranche 1
         #   print(f"[SKIP] {p.name} -> {tranche_code} (requested skip)")
          #  continue
        if not (1 <= num <= 11):            # guard against stray dirs
            print(f"[WARN] unexpected tranche {num} in {p.name} — skipping")
            continue
        jobs.append((p, tranche_code))
    # log what we found
    print("[DISCOVERED] ", ", ".join(code for _, code in jobs) or "(none)")
    return jobs

"""def tranche_already_done(out_root: Path, tranche_code: str) -> bool:
    tdir = out_root / f"tranche={tranche_code}"
    if not tdir.exists():
        return False
    # consider it done if there is at least one parquet file under it
    return any(tdir.rglob("*.parquet"))"""

def run_one(tranche_path: Path, tranche_code: str) -> str:
    normalize_tranche(str(tranche_path), str(OUT), tranche_code)
    return tranche_code

if __name__ == "__main__":
    t0 = perf_counter()

    # 0) (Optional) build MRN lookup once
    tranche_dirs = discover_tranches(BASE)
    tranche_dirs = [
        (p, code) for (p, code) in discover_tranches(BASE)
        #if not tranche_already_done(OUT, code)
    ]
    mrn_files = []
    for d, _ in tranche_dirs:
        mrn_files.extend(d.glob("*_Mrn.txt"))
    print(f"Found {len(mrn_files)} MRN files across {len(tranche_dirs)} tranches")
    # build_mrn_lookup(mrn_files, str(OUT.parent / "lookups" / "mrn_empi_map.parquet"))

    # 1) parallelize by tranche
    # Pick a sane worker count: do not oversubscribe the disk or network share
    import os
    max_workers = min(len(tranche_dirs), max(1, os.cpu_count() // 2))  # start conservative

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, p, code) for (p, code) in tranche_dirs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tranches"):
            code = fut.result()  # will raise if any tranche failed
            tqdm.write(f"[DONE] {code}")

    print(f"[ALL DONE] {len(tranche_dirs)} tranches in {perf_counter()-t0:,.1f}s")
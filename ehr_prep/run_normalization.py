from pathlib import Path
from parsers.mrn_map import build_mrn_lookup
from normalize import normalize_tranche

# 1 — Build MRN→EMPI lookup once
# /Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/raw_MRN_files
mrn_dir = Path("/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/raw_MRN_files")

# Pick all *_Mrn.txt files
mrn_files = sorted(mrn_dir.glob("*_Mrn.txt"))
print(f"Found {len(mrn_files)} MRN files:")
for f in mrn_files:
    print(" -", f)

# Build the lookup table
#build_mrn_lookup(mrn_files, "ehr_store/lookups/mrn_empi_map.parquet")

# 2 — Normalize one tranche at a time
normalize_tranche("./raw_files/tranche_01", "ehr_store/normalized", "T01")
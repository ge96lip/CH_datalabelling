from pathlib import Path
from parsers.mrn_map import build_mrn_lookup
from normalize import normalize_tranche

# 1 — Build MRN→EMPI lookup once
# /Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/raw_MRN_files
# mrn_dir = Path("/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/raw_MRN_files")

base = Path(r"Z:\JonathanMueller\llmProject_20250627\data")

# only tranche folders; no deep recursion
tranche_dirs = sorted(base.glob("llmTranche*_raw_*"))   # 11 matches
mrn_files = []
for d in tranche_dirs:
    mrn_files.extend(d.glob("*_Mrn.txt"))               # top-level files only

print(f"Found {len(mrn_files)} MRN files")

# Build the lookup table
build_mrn_lookup(mrn_files, "ehr_store/lookups/mrn_empi_map.parquet")

# 2 — Normalize one tranche at a time
#normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche1_raw_20250627", "ehr_store/normalized", "T01")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche2_raw_20250627", "ehr_store/normalized", "T02")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche3_raw_20250627", "ehr_store/normalized", "T03")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche4_raw_20250627", "ehr_store/normalized", "T04")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche5_raw_20250627", "ehr_store/normalized", "T05")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche6_raw_20250627", "ehr_store/normalized", "T06")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche7_raw_20250627", "ehr_store/normalized", "T07")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche8_raw_20250627", "ehr_store/normalized", "T08")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche9_raw_20250627", "ehr_store/normalized", "T09")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche10_raw_20250627", "ehr_store/normalized", "T010")
normalize_tranche("Z:\JonathanMueller\llmProject_20250627\data\llmTranche11_raw_20250627", "ehr_store/normalized", "T011")

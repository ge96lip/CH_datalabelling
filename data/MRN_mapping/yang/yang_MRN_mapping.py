import os
import csv
from collections import defaultdict

import re

# === CONFIG ===
PATIENTS_FOLDER = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/patient_txts"  # folder containing ~205 mrn.txt files
MAPPING_FOLDER = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/raw_MRN_files"       # folder containing the 11 mapping files
MRN_LIST_FILE = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/mrn_yang.txt"
MATCHED_FILE = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/mrn_to_empi.csv"
WRONG_MATCH_FILE = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/wrong_match_mrn_to_empi.csv"
NOT_MATCHED_FILE = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/not_matched_mrns.txt"

# Columns in mapping files
MGH_COLUMN = "MGH_MRN"
OTHER_COLUMNS = ["BWH_MRN", "FH_MRN", "SRH_MRN", "NWH_MRN",
                 "NSMC_MRN", "MCL_MRN", "MEE_MRN", "DFC_MRN", "WDH_MRN"]

# === STEP 1: Extract MRNs from patients folder ===
DIGITS_AT_START = re.compile(r'^(\d+)')
def normalize_mrn(val: str):
    """Keep only digits, strip leading zeros. Return None if empty/invalid."""
    if val is None:
        return None
    s = re.sub(r'\D', '', str(val))
    s = s.lstrip('0')
    return s if s else None

def extract_mrn_from_filename(fname: str):
    """
    Extract the leading digit run from a filename like '001234.txt' or '1234_notes.txt'.
    Returns normalized MRN (no leading zeros) or None if none found.
    """
    stem, ext = os.path.splitext(fname)
    m = DIGITS_AT_START.match(stem)
    if not m:
        return None
    return normalize_mrn(m.group(1))

# === STEP 1: Build MRN set from filenames ===
mrns = set()
for fname in os.listdir(PATIENTS_FOLDER):
    if not fname.lower().endswith(".txt"):
        continue
    mrn_norm = extract_mrn_from_filename(fname)
    if mrn_norm:
        mrns.add(mrn_norm)

# Write mrn_yang.txt
with open(MRN_LIST_FILE, "w", encoding="utf-8") as out:
    for m in sorted(mrns, key=lambda x: int(x)):
        out.write(f"{m}\n")


print(f"[INFO] Extracted {len(mrns)} unique MRNs; wrote to {MRN_LIST_FILE}")

# === STEP 2: Build MRN to EMPI mapping from mapping files ===
mrn_to_empi_matches = []
wrong_matches = []
not_matched = set(mrns)

# Track MRN to multiple EMPI situations
mrn_empi_map = defaultdict(set)

for map_file in os.listdir(MAPPING_FOLDER):
    if not map_file.lower().endswith(".txt"):
        continue
    matched_count = 0
    map_path = os.path.join(MAPPING_FOLDER, map_file)

    with open(map_path, "r") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            empi = row["Enterprise_Master_Patient_Index"].strip()
            for mrn_col in [MGH_COLUMN] + OTHER_COLUMNS:
                val = row[mrn_col].strip().lstrip("0")
                if val and val in not_matched:
                    if mrn_col == MGH_COLUMN:
                        mrn_to_empi_matches.append((val, empi, mrn_col, map_file))
                    else:
                        wrong_matches.append((val, empi, mrn_col, map_file))
                    matched_count += 1
                    mrn_empi_map[val].add(empi)
                    if val in not_matched:
                        not_matched.remove(val)

    print(f"[INFO] {map_file} matched {matched_count}; remaining {len(not_matched)}")

# === STEP 3: Write outputs ===
with open(MATCHED_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mrn", "empi", "matched_column", "source_file"])
    writer.writerows(mrn_to_empi_matches)

with open(WRONG_MATCH_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mrn", "empi", "matched_column", "source_file"])
    writer.writerows(wrong_matches)

with open(NOT_MATCHED_FILE, "w") as f:
    for m in sorted(not_matched):
        f.write(f"{m}\n")

# === STEP 4: Warnings ===
# MRNs with multiple EMPIs
multi_empi = {mrn: empis for mrn, empis in mrn_empi_map.items() if len(empis) > 1}
if multi_empi:
    print(f"[WARN] {len(multi_empi)} MRNs map to multiple EMPIs. Showing first few:")
    for mrn, empis in list(multi_empi.items())[:10]:
        print(f"  MRN: {mrn} → EMPIs: {', '.join(empis)}")

# MRNs not matched
if not_matched:
    print(f"[WARN] {len(not_matched)} MRNs were not mapped to an EMPI. Showing first few:")
    for m in list(not_matched)[:10]:
        print(f"  {m}")

# Wrong matches (non-MGH)
if wrong_matches:
    print(f"[WARN] {len(wrong_matches)} MRNs were mapped to EMPIs but outside of MGH. Showing first few:")
    for m in wrong_matches[:10]:
        print(f"  MRN: {m[0]}, EMPI: {m[1]}, Column: {m[2]}, Source: {m[3]}")
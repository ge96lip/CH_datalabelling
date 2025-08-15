import os
import re
import csv
from collections import defaultdict
from pathlib import Path

# === CONFIG ===
INPUT_CSV = "STS_database.csv"       
MAPPING_FOLDER = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/raw_MRN_files"          # folder with the 11 pipe-delimited mapping .txt files
MRN_LIST_FILE = "STS_mrn.txt"
MATCHED_FILE = "STS_mrn_to_empi.csv"
WRONG_MATCH_FILE = "STS_wrong_match_mrn_to_empi.csv"
NOT_MATCHED_FILE = "STS_not_matched_mrns.txt"
# NEW: paths for Yang comparison
YANG_LIST_FILE = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/yang/mrn_yang.txt"
YANG_NOT_IN_STS_FILE = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/yang/mrns_in_yang_not_in_STS.txt"

# CSV column names in INPUT_CSV
CSV_COL_HOSPITAL = "Hospital Name"
CSV_COL_MRN = "MedicalRecord"

# Mapping-file columns (pipe-delimited)
ALL_COLS = [
    "IncomingId","IncomingSite","Status","Enterprise_Master_Patient_Index",
    "EPIC_PMRN","MGH_MRN","BWH_MRN","FH_MRN","SRH_MRN","NWH_MRN","NSMC_MRN",
    "MCL_MRN","MEE_MRN","DFC_MRN","WDH_MRN"
]
MRN_COLS = [c for c in ALL_COLS if c.endswith("_MRN")]

# === Hospital: expected MRN column
# Add/adjust synonyms as needed for your data.
HOSPITAL_SYNONYMS = {
    "MGH_MRN": [
        "massachusetts general hospital", "mgh"
    ],
    "BWH_MRN": [
        "brigham and women's hospital", "bwh"
    ],
    "FH_MRN": [
        "faulkner hospital", "brigham and women's faulkner", "bwf", "bwh faulkner"
    ],
    "SRH_MRN": [
        "salem hospital", "salem regional", "salem (nsmc)"
    ],
    "NSMC_MRN": [
        "north shore medical center", "nsmc"
    ],
    "NWH_MRN": [
        "newton-wellesley hospital", "nwh"
    ],
    "WDH_MRN": [
        "wentworth-douglass hospital", "wdh"
    ],
    "MEE_MRN": [
        "massachusetts eye and ear", "mee", "mass eye and ear"
    ],
    "DFC_MRN": [
        "dana-farber", "dana farber", "dfci", "dana-farber"
    ],
    "MCL_MRN": [
        # Fill if you use MCL — placeholder synonyms here:
        "mcl", "mgb community physicians"
    ],
}

def normalize_mrn(val: str):
    """Keep only digits, strip leading zeros. Return None if empty/invalid."""
    if val is None:
        return None
    s = re.sub(r'\D', '', str(val))
    s = s.lstrip('0')
    return s if s else None

def normalize_text(s: str):
    return (s or "").strip().lower()

def expected_col_for_hospital(hospital_name: str):
    """Return the expected *_MRN column based on hospital name, or None if unknown."""
    h = normalize_text(hospital_name)
    for col, synonyms in HOSPITAL_SYNONYMS.items():
        for syn in synonyms:
            if syn in h:
                return col
    return None  # unknown hospital; we'll warn and accept any mapping as 'unknown-expected'

# === STEP 1: Read MRNs and hospital names from the CSV ===
mrns = set()
mrn_hospital = {}         # mrn -> hospital (first seen)
unknown_hospital_mrns = set()

with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    missing_cols = {c for c in [CSV_COL_HOSPITAL, CSV_COL_MRN] if c not in reader.fieldnames}
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    for row in reader:
        raw_mrn = row.get(CSV_COL_MRN)
        mrn = normalize_mrn(raw_mrn)
        if not mrn:
            continue

        hospital = row.get(CSV_COL_HOSPITAL, "").strip()
        mrns.add(mrn)
        mrn_hospital.setdefault(mrn, hospital)
        if expected_col_for_hospital(hospital) is None:
            unknown_hospital_mrns.add(mrn)

# Write mrn_STS.txt (from CSV)
with open(MRN_LIST_FILE, "w", encoding="utf-8") as out:
    for m in sorted(mrns, key=lambda x: int(x)):
        out.write(f"{m}\n")

print(f"[INFO] Extracted {len(mrns)} MRNs from '{INPUT_CSV}', wrote to {MRN_LIST_FILE}")
if unknown_hospital_mrns:
    print(f"[WARN] {len(unknown_hospital_mrns)} MRNs have unknown hospital names; "
          f"matching will not enforce a specific *_MRN column (showing first few):")
    for m in list(sorted(unknown_hospital_mrns, key=int))[:10]:
        print(f"  MRN {m} -> '{mrn_hospital.get(m, '')}'")

# === STEP 2: Scan mapping files, collect all matches ===
all_matches = defaultdict(lambda: defaultdict(list))  # mrn -> col -> list[(empi, source_file)]
mrn_to_empis = defaultdict(set)
seen_any = set()

mapping_files = [p for p in Path(MAPPING_FOLDER).iterdir() if p.suffix.lower() == ".txt"]
for p in sorted(mapping_files, key=lambda x: x.name):
    newly_seen_this_file = set()
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            empi = (row.get("Enterprise_Master_Patient_Index") or "").strip()
            if not empi:
                continue
            for col in MRN_COLS:
                raw = row.get(col)
                v = normalize_mrn(raw)
                if v and v in mrns:
                    all_matches[v][col].append((empi, p.name))
                    mrn_to_empis[v].add(empi)
                    newly_seen_this_file.add(v)
                    seen_any.add(v)

    remaining = len(mrns - seen_any)
    print(f"[INFO] {p.name} matched {len(newly_seen_this_file)}; remaining {remaining}")

# === STEP 3: Decide outputs per MRN using hospital-based expected column ===
matched_rows = []        # correct column for that hospital
wrong_matched_rows = []  # matched but on a different hospital column + explanation
matched_any_mrn = set()

for mrn in mrns:
    hospital = mrn_hospital.get(mrn, "")
    expected_col = expected_col_for_hospital(hospital)

    col_hits = all_matches[mrn]  # dict col -> list of (empi, src)

    if expected_col:
        # Correct matches
        if expected_col in col_hits:
            matched_any_mrn.add(mrn)
            for empi, src in col_hits[expected_col]:
                matched_rows.append([mrn, empi, expected_col, src])

        # Wrong matches: different MRN column than expected
        other_cols = [c for c in col_hits.keys() if c != expected_col]
        for col in other_cols:
            matched_any_mrn.add(mrn)
            for empi, src in col_hits[col]:
                wrong_matched_rows.append([mrn, empi, col, src, expected_col])  # add expected col

    else:
        # Unknown hospital: accept any mapping as matched
        if col_hits:
            matched_any_mrn.add(mrn)
            for col, hits in col_hits.items():
                for empi, src in hits:
                    matched_rows.append([mrn, empi, f"{col} (no expected hospital)", src])

not_matched_mrns = sorted(mrns - matched_any_mrn, key=lambda x: int(x))

# === STEP 4: Write outputs ===
os.makedirs(os.path.dirname(MATCHED_FILE) or ".", exist_ok=True)

with open(MATCHED_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["mrn", "empi", "matched_column", "source_file"])
    writer.writerows(matched_rows)

with open(WRONG_MATCH_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["mrn", "empi", "matched_column", "source_file", "exp_hos"])
    writer.writerows(wrong_matched_rows)  # includes expected col now

with open(NOT_MATCHED_FILE, "w", encoding="utf-8") as f:
    for m in not_matched_mrns:
        f.write(f"{m}\n")

# === STEP 5: Warnings ===
multi_empi = {mrn: empis for mrn, empis in mrn_to_empis.items() if len(empis) > 1}
if multi_empi:
    print("[WARN] Some MRNs map to multiple EMPIs. Showing first few:")
    for mrn, empis in list(multi_empi.items())[:10]:
        print(f"  MRN: {mrn} mapped to EMPIs: {', '.join(sorted(empis))}")

if not_matched_mrns:
    print(f"[WARN] {len(not_matched_mrns)} MRNs were not mapped to an EMPI. Showing first few:")
    for m in not_matched_mrns[:10]:
        print(f"  {m}")

# MRNs that matched only via a non-expected site (i.e., appear in wrong_match file but not in matched file)
only_wrong = set(r[0] for r in wrong_matched_rows) - set(r[0] for r in matched_rows)
if only_wrong:
    print(f"[WARN] {len(only_wrong)} MRNs were mapped to EMPIs but using MRNs outside the expected hospital. Showing first few:")
    for m in list(sorted(only_wrong, key=int))[:10]:
        # rows for this mrn
        rows_m = [r for r in wrong_matched_rows if r[0] == m][:2]

        # format examples; handle both old (4 fields) and new (5 fields) shapes
        formatted_examples = []
        for r in rows_m:
            # r = [mrn, empi, col, src]  OR  [mrn, empi, col, src, exp_hos]
            mrn_r, empi, col, src = r[0], r[1], r[2], r[3]
            exp_hos = r[4] if len(r) > 4 else expected_col_for_hospital(mrn_hospital.get(mrn_r, ""))
            formatted_examples.append(f"{empi} via {col} ({src}); expected {exp_hos}")

        exp_col = expected_col_for_hospital(mrn_hospital.get(m, ""))
        print(f"  MRN: {m} (expected={exp_col}, hospital='{mrn_hospital.get(m, '')}') -> " +
              ", ".join(formatted_examples))
        
# === STEP 6: Find MRNs which are in Yang but not in STS ===
def load_mrns_from_txt(path: str) -> set[str]:
    """Load MRNs from a one-per-line text file, normalized."""
    mrns = set()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            m = normalize_mrn(line.strip())
            if m:
                mrns.add(m)
    return mrns

def load_mrns_from_sts_csv(path: str, mrn_col: str) -> set[str]:
    """Load MRNs from STS CSV using the given MRN column, normalized."""
    mrns = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if mrn_col not in reader.fieldnames:
            raise ValueError(f"CSV is missing required MRN column '{mrn_col}'. Found: {reader.fieldnames}")
        for row in reader:
            m = normalize_mrn(row.get(mrn_col))
            if m:
                mrns.add(m)
    return mrns

def report_yang_not_in_sts(yang_file: str, sts_csv: str, sts_mrn_col: str, out_file: str) -> int:
    """Compute MRNs present in Yang list but not in STS; write them and print a summary."""
    yang_mrns = load_mrns_from_txt(yang_file)
    sts_mrns = load_mrns_from_sts_csv(sts_csv, sts_mrn_col)
    diff = sorted(yang_mrns - sts_mrns, key=lambda x: int(x))
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for m in diff:
            f.write(f"{m}\n")
    print(f"[INFO] Yang vs STS: {len(diff)} MRNs are in Yang ({yang_file}) but NOT in STS ({sts_csv}).")
    if diff:
        print(f"       First few: {diff[:10]}")
    print(f"[INFO] Wrote list to {out_file}")
    return len(diff)

try:
    report_yang_not_in_sts(YANG_LIST_FILE, INPUT_CSV, CSV_COL_MRN, YANG_NOT_IN_STS_FILE)
except Exception as e:
    print(f"[ERR] Yang vs STS comparison failed: {e}")
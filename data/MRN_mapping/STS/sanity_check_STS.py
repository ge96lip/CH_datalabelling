import os
import re
import csv
import collections
from pathlib import Path

# === CONFIG ===
PATIENTS_FOLDER = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/patient_txts"  # folder containing ~205 mrn.txt files
MAPPING_FOLDER = "/Users/carlotta/Desktop/Code_MT/CH_datalabelling/data/MRN_mapping/raw_MRN_files"       # folder containing the 11 mapping files
MRN_LIST_FILE = "STS_mrn.txt"
MATCHED_FILE = "STS_mrn_to_empi.csv"
WRONG_MATCH_FILE = "STS_wrong_match_mrn_to_empi.csv"
NOT_MATCHED_FILE = "STS_not_matched_mrns.txt"
SANITY_FOUND_FILE = "STS_sanity_recovered_mrn_to_empi.csv"
SANITY_STILL_NOT_FOUND_FILE = "STS_sanity_still_not_matched_mrns.txt"
# All expected headers (ignore extras if present)
ALL_COLS = [
    "IncomingId","IncomingSite","Status","Enterprise_Master_Patient_Index",
    "EPIC_PMRN","MGH_MRN","BWH_MRN","FH_MRN","SRH_MRN","NWH_MRN","NSMC_MRN",
    "MCL_MRN","MEE_MRN","DFC_MRN","WDH_MRN"
]
MRN_COLS = [c for c in ALL_COLS if c.endswith("_MRN")]  # every *_MRN column

def normalize_mrn(val: str):
    if val is None:
        return None
    s = re.sub(r'\D', '', str(val))  # keep digits only
    s = s.lstrip('0')
    return s if s else None

def load_not_matched_set(path: str):
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            v = normalize_mrn(line.strip())
            if v:
                s.add(v)
    return s

def audit_profile(all_patient_mrns, lookup):
    # Length histograms
    from collections import Counter
    lens_pat = Counter(len(m) for m in all_patient_mrns)
    lens_map = Counter(len(m) for m in lookup.keys())
    print("[INFO] MRN length histogram (patients):", dict(sorted(lens_pat.items())))
    print("[INFO] MRN length histogram (mapping) :", dict(sorted(lens_map.items())))

    # Proportion of purely numeric BEFORE normalization (spot alphabetical prefixes)
    # (Here we only have normalized, so probe pattern of *raw* later if needed.)

    # Top-10 unmatched samples
    all_map = set(lookup.keys())
    unmatched = sorted(all_patient_mrns - all_map, key=int)
    print(f"[INFO] Unmatched count = {len(unmatched)}; examples:", unmatched[:10])

    # Fuzzy tail probe: match by last 6/7 digits (to catch prefixed MRNs)
    tails6 = {m[-6:]: m for m in all_map if len(m) >= 6}
    tails7 = {m[-7:]: m for m in all_map if len(m) >= 7}

    fuzzy_hits = []
    for m in unmatched[:5000]:  # cap work
        if len(m) >= 6 and m[-6:] in tails6:
            fuzzy_hits.append((m, tails6[m[-6:]], "last6"))
        elif len(m) >= 7 and m[-7:] in tails7:
            fuzzy_hits.append((m, tails7[m[-7:]], "last7"))
    if fuzzy_hits:
        print(f"[WARN] {len(fuzzy_hits)} unmatched MRNs have tail matches in mapping (prefix/padding issue). Showing first few:")
        for a, b, how in fuzzy_hits[:10]:
            print(f"  patient {a} ~ mapping {b} ({how})")

def open_mapping_rows(path):
    # 1) try utf-8-sig, then utf-16, then latin-1
    encodings = ["utf-8-sig", "utf-16", "utf-16le", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                sample = f.read(4096)
                # 2) delimiter sniff
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters="|\t,")
                    delim = dialect.delimiter
                except Exception:
                    delim = "|"  # fallback

                f.seek(0)
                # 3) find the real header line (contains EMPI + at least one *_MRN)
                header_line = None
                pending = []
                for line in f:
                    pending.append(line)
                    low = line.lower()
                    if "enterprise_master_patient_index" in low and ("_mrn" in low or "epic_pmrn" in low):
                        header_line = line
                        break
                if header_line is None:
                    # No proper header found
                    return []

                # Rebuild file iterator from header_line + rest
                remainder = "".join([header_line] + list(f))
                # 4) DictReader with normalized headers (strip spaces, collapse runs)
                reader = csv.DictReader(remainder.splitlines(), delimiter=delim, quotechar='"', escapechar='\\')
                # Normalize fieldnames map
                norm = lambda s: re.sub(r'\s+', '_', (s or "").strip())
                field_map = {name: norm(name) for name in reader.fieldnames or []}

                # Alias common drifted names
                alias = {
                    "ENTERPRISE_MASTER_PATIENT_INDEX": "Enterprise_Master_Patient_Index",
                    "ENTERPRISE_MASTER_PATIENT_ID": "Enterprise_Master_Patient_Index",
                    "MGH_MRN": "MGH_MRN",
                    "MGH_MRN_": "MGH_MRN",
                    "MGH_MRN__": "MGH_MRN",
                    "MGH_MRN__1": "MGH_MRN",
                    "MGH_MRN__2": "MGH_MRN",
                    "MGH_MRN_ ": "MGH_MRN",
                    "MGH_MRN_SPACE": "MGH_MRN",  # example placeholder
                    "BWH_MRN": "BWH_MRN",
                    "EPIC_PMRN": "EPIC_PMRN",
                }
                # inverse: normalized -> canonical
                inv_alias = {k.upper(): v for k, v in alias.items()}

                # Yield normalized rows with canonical keys where possible
                for row in reader:
                    nr = {}
                    for k, v in row.items():
                        nk = field_map.get(k, k)
                        canon = inv_alias.get(nk.upper(), nk)
                        nr[canon] = v
                    yield nr
            return
        except Exception as e:
            last_err = e
            continue
    # If we get here, all decodes failed
    print(f"[WARN] Could not decode '{path}': {last_err}")
    return []

from collections import defaultdict, Counter

def build_global_lookup(mapping_folder: str):
    lookup = collections.defaultdict(lambda: collections.defaultdict(set))
    seen_mrns = set()
    seen_empis = set()
    total_rows_with_empi = 0  # sum across all files
    empi_occurrences = Counter()  # counts total appearances of each EMPI
    empi_files = defaultdict(set)  # maps EMPI -> set of files where it appeared

    files = [p for p in Path(mapping_folder).iterdir() if p.suffix.lower() == ".txt"]
    for p in sorted(files):
        row_count = 0
        with_data_mrn = 0
        with_data_empi = 0

        for row in open_mapping_rows(str(p)):
            row_count += 1
            empi = (row.get("Enterprise_Master_Patient_Index") or "").strip()
            if empi:
                with_data_empi += 1
                seen_empis.add(empi)
                empi_occurrences[empi] += 1
                empi_files[empi].add(p.name)

            found_any_mrn = False
            for col in MRN_COLS + ["EPIC_PMRN"]:
                raw = row.get(col)
                mrn = normalize_mrn(raw)
                if mrn:
                    found_any_mrn = True
                    if empi:
                        lookup[mrn][empi].add((col, p.name))
                    seen_mrns.add(mrn)

            if found_any_mrn:
                with_data_mrn += 1

        total_rows_with_empi += with_data_empi
        print(f"[INFO] Audit {p.name}: rows={row_count}, rows_with_any_MRN={with_data_mrn}, rows_with_EMPI={with_data_empi}")

    # Final check
    if len(seen_empis) == total_rows_with_empi:
        print(f"[CHECK] Unique EMPIs ({len(seen_empis)}) match total rows_with_EMPI ({total_rows_with_empi})")
    else:
        print(f"[CHECK] Mismatch: {len(seen_empis)} unique EMPIs vs {total_rows_with_empi} rows_with_EMPI")

        # Find EMPIs that appear more than once
        duplicates = {empi: (count, empi_files[empi]) for empi, count in empi_occurrences.items() if count > 1}
        print("\n[DEBUG] Duplicate EMPIs (count, files):")
        for empi, (count, files_set) in duplicates.items():
            print(f"  {empi}: {count} times in files {', '.join(sorted(files_set))}")

    return lookup, seen_mrns, seen_empis

def main():
    combined_file = "combined_MRN_EMPI_mapping.csv"
    save_combined_mapping(MAPPING_FOLDER, combined_file)
    # 1) Load the previously "not matched" MRNs
    prev_unmatched = load_not_matched_set(NOT_MATCHED_FILE)
    print(f"[INFO] Loaded {len(prev_unmatched)} MRNs from {NOT_MATCHED_FILE}")

    # 2) Merge/scan all 11 mapping files into one global lookup
    lookup, seen_any, seen_empis = build_global_lookup(MAPPING_FOLDER)
    print(f"[INFO] Scanned {len(seen_any):,} unique MRNs across {MAPPING_FOLDER}")
    print(f"[INFO] Found {len(seen_empis):,} unique EMPIs across all mapping files")
    # 3) Re-check the previously "not matched" MRNs against ALL columns in ALL files
    recovered_rows = []
    still_not_found = []

    for mrn in sorted(prev_unmatched, key=int):
        if mrn in lookup:
            # Collate: MRN, EMPI, matched_column(s), source_file(s)
            # explode per (empi, column, source) for clarity
            for empi, hits in lookup[mrn].items():
                for col, src in sorted(hits):
                    recovered_rows.append([mrn, empi, col, src])
        else:
            still_not_found.append(mrn)

    # 4) Write outputs
    with open(SANITY_FOUND_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mrn", "empi", "matched_column", "source_file"])
        w.writerows(recovered_rows)

    with open(SANITY_STILL_NOT_FOUND_FILE, "w", encoding="utf-8") as f:
        for mrn in still_not_found:
            f.write(f"{mrn}\n")

    # 5) Print summary
    print(f"[INFO] Recovered {len(recovered_rows)} (MRN,EMPI,column,source) rows for "
          f"{len({r[0] for r in recovered_rows})} previously-unmatched MRNs, wrote to {SANITY_FOUND_FILE}")
    print(f"[INFO] Still not found: {len(still_not_found)} MRNs, wrote to {SANITY_STILL_NOT_FOUND_FILE}")

    # 6) Helpful warnings
    #   - multiple EMPIs for a single MRN among the recovered set
    multi_empi = {}
    for mrn in {r[0] for r in recovered_rows}:
        empis = set(lookup[mrn].keys())
        if len(empis) > 1:
            multi_empi[mrn] = sorted(empis)
    if multi_empi:
        print(f"[WARN] {len(multi_empi)} recovered MRNs map to multiple EMPIs. Showing first few:")
        for mrn, empis in list(multi_empi.items())[:10]:
            print(f"  MRN {mrn} → EMPIs: {', '.join(empis)}")
def save_combined_mapping(mapping_folder: str, output_file: str):
    """Combine all mapping files into a single CSV."""
    combined_rows = []
    header_seen = set()

    files = [p for p in Path(mapping_folder).iterdir() if p.suffix.lower() == ".txt"]
    for p in sorted(files):
        for row in open_mapping_rows(str(p)):
            # Add source file column for traceability
            row["source_file"] = p.name
            combined_rows.append(row)
            header_seen.update(row.keys())

    if not combined_rows:
        print(f"[WARN] No rows found in {mapping_folder}")
        return

    # Ensure consistent column order
    header = list(header_seen)
    # Optionally put Enterprise_Master_Patient_Index first
    if "Enterprise_Master_Patient_Index" in header:
        header.remove("Enterprise_Master_Patient_Index")
        header = ["Enterprise_Master_Patient_Index"] + header

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"[INFO] Combined {len(combined_rows)} rows from {len(files)} files into {output_file}")

if __name__ == "__main__":
    main()
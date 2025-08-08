import pandas as pd
import os

# === Load ===
df = pd.read_csv("/Users/carlotta/Desktop/Code_MT/CH_datalabelling/Recurrence Dataset_altered.csv")

# === Drop columns you don't want ===
columns_to_drop = [
    "Did they develop a second primary lung cancer?",
    "Second Primary Date",
    "Has there been recurrence?",
    "Where is the recurrence site?",
    "Date of Recurrence",
    "Date of CT that Noted Recurrence",
    "Accession # of CT Noting Recurrence",
    "RADIOLOGY",
    "PATHOLOGY",
    "Follow-up Information",
]
df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# === Build tag map (same logic as before) ===
tag_map = {}

# ID | for A–C
id_cols = df.columns[0:3]
tag_map.update({col: f"ID | {col}" for col in id_cols})

# DEM | for D–H
dem_cols = df.columns[3:8]
tag_map.update({col: f"DEM | {col}" for col in dem_cols})

# PATH | for J–X
path_cols = df.columns[8:23]
tag_map.update({col: f"PATH | {col}" for col in path_cols})

# RAD | for Z and AA–AH (columns 23–32)
rad_cols = df.columns[23:32]
tag_map.update({col: f"RAD | {col}" for col in rad_cols})

# Follow-Up Information | for AQ–BA (col 32–42)
followup_cols = df.columns[32:43]
tag_map.update({col: f"Follow-Up Information | {col}" for col in followup_cols})

# Apply tags
df.rename(columns=tag_map, inplace=True)

# === Resolve key columns after renaming ===
def find_col(endswith_str):
    matches = [c for c in df.columns if c.endswith(endswith_str)]
    if not matches:
        raise ValueError(f"Could not find column ending with '{endswith_str}'. Check CSV headers.")
    # Prefer those that also start with "ID |" if multiple
    id_pref = [c for c in matches if c.startswith("ID |")]
    return id_pref[0] if id_pref else matches[0]

record_id_col = find_col("Record ID")
mrn_col       = find_col("MRN")

# === Output dir ===
output_dir = "./data/patient_txts"
os.makedirs(output_dir, exist_ok=True)

# === Helper for writing one row ===
def write_row(fh, row):
    for column, value in row.items():
        value_str = "" if pd.isna(value) else str(value)
        fh.write(f"{column} | {value_str}\n")

def write_patients():
    # === Group by patient (Record ID + MRN) and write once per patient ===
    for (rid, mrn), g in df.groupby([record_id_col, mrn_col], sort=False):
        if pd.isna(mrn):
            mrn_str = ""
        else:
            # Convert to string and remove trailing .0 if present
            mrn_str = str(mrn).strip()
            if mrn_str.endswith(".0"):
                mrn_str = mrn_str[:-2]

        filename = os.path.join(output_dir, f"{mrn_str}.txt")  # naming by MRN

        with open(filename, "w", encoding="utf-8") as f:
            for i, (_, row) in enumerate(g.iterrows(), start=1):
                if i > 1:
                    f.write(
                        "\n" + "-" * 72 +
                        f"\n# Additional entry #{i} for MRN {mrn_str}\n" +
                        "-" * 72 + "\n"
                    )
                write_row(f, row)

def data_analysis():
    # Limit search to the first 209 rows
    df_limited = df.iloc[:209]

    # Get ID columns (your helper still works)
    record_id_col = find_col("Record ID")
    mrn_col       = find_col("MRN")

    # Count entries per patient
    counts = (
        df_limited.groupby([record_id_col, mrn_col], dropna=False)
                .size()
                .reset_index(name="n_entries")
    )

    # Keep only patients with >1 entries
    multi = counts[counts["n_entries"] > 1]

    # Add the row number (first occurrence in the limited data)
    first_rows = (
        df_limited.reset_index()  # index here is the original CSV row number
                .groupby([record_id_col, mrn_col], as_index=False)['index']
                .min()
                .rename(columns={'index': 'first_row_number'})
    )

    # Merge counts and first occurrence
    multi = multi.merge(first_rows, on=[record_id_col, mrn_col])

    # Sort by first occurrence
    multi = multi.sort_values('first_row_number')

    print(multi.to_string(index=False))

def labels(): 
    # --- Normalize labels robustly ---
    label_col = "Rec_Label"
    mrn_col   = find_col("MRN")

    df["_label_norm"] = (
        pd.to_numeric(df[label_col].astype(str).str.strip(), errors="coerce")
        .astype("Int64")
    )

    valid_vals = {0, 1, 3}
    valid_mask = df["_label_norm"].isin(list(valid_vals))

    # --- How many unique patients overall? ---
    all_mrns = (
        df[mrn_col].dropna().astype(str).str.strip().drop_duplicates()
    )
    print("Total unique MRNs (dataset):", len(all_mrns))

    # --- Unique MRNs per valid class (0/1/3) ---
    by_class = {
        v: (
            df.loc[df["_label_norm"] == v, mrn_col]
            .dropna().astype(str).str.strip().drop_duplicates()
        )
        for v in valid_vals
    }
    print({v: len(s) for v, s in by_class.items()})  # sanity counts

    # --- Missing MRNs: exist in data but never have a valid label 0/1/3 ---
    labeled_any = (
        df.loc[valid_mask, mrn_col]
        .dropna().astype(str).str.strip().drop_duplicates()
    )
    missing_mrns = sorted(set(all_mrns) - set(labeled_any))
    print("Missing MRNs (no valid 0/1/3):", missing_mrns)

    # --- Show what raw Rec_Label values those missing MRNs actually have ---
    if missing_mrns:
        raw_for_missing = (
            df[df[mrn_col].astype(str).str.strip().isin(missing_mrns)]
            .assign(raw=df[label_col].astype(str).str.strip())
            .groupby(df[mrn_col].astype(str).str.strip())["raw"]
            .unique()
            .rename("raw_values")
            .reset_index()
        )
        print(raw_for_missing.to_string(index=False))

    # --- Optional: detect conflicting MRNs (more than one valid class across rows) ---
    valid_per_mrn = (
        df.loc[valid_mask, [mrn_col, "_label_norm"]]
        .dropna()
        .assign(MRN=df[mrn_col].astype(str).str.strip())
        .groupby("MRN")["_label_norm"]
        .nunique()
    )
    conflict_mrns = valid_per_mrn[valid_per_mrn > 1].index.tolist()
    print("Conflict MRNs (multiple valid labels):", conflict_mrns)

    # --- Optional: write a small CSV report for auditing ---
    audit = (
        df.assign(
            MRN=df[mrn_col].astype(str).str.strip(),
            Rec_Label_raw=df[label_col].astype(str).str.strip(),
            Rec_Label_norm=df["_label_norm"]
        )[[ "MRN", "Rec_Label_raw", "Rec_Label_norm" ]]
    )
    audit_path = "./data/labels/labels_audit.csv"
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    audit.to_csv(audit_path, index=False)
    print("Audit saved to", audit_path)

if __name__ == "__main__":
    #data_analysis()
    #labels()
    write_patients()
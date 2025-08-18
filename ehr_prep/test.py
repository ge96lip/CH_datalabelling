from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc

def check_mrn_lookup(path: str):
    tbl = pq.read_table(path)
    print(f"Loaded lookup table: {tbl.num_rows:,} rows, {tbl.num_columns} columns")

    # Columns
    print("Columns:", tbl.column_names)

    # 1 — Count unique EMPIs and MRNs
    unique_empi = pc.count_distinct(tbl["empi"])
    unique_mrn_norm = pc.count_distinct(tbl["mrn_norm"])
    print(f"Unique EMPIs: {unique_empi.as_py():,}")
    print(f"Unique MRNs (normalized): {unique_mrn_norm.as_py():,}")

    # 2 — Null checks
    null_empi = pc.sum(pc.is_null(tbl["empi"]))
    null_mrn_norm = pc.sum(pc.is_null(tbl["mrn_norm"]))
    print(f"Null EMPIs: {null_empi.as_py():,}")
    print(f"Null MRNs (normalized): {null_mrn_norm.as_py():,}")

    # 3 — Duplicate MRN→EMPI pairs
    # A given MRN_norm should map to exactly one EMPI
    df = tbl.to_pandas()
    dup_counts = df.groupby("mrn_norm")["empi"].nunique()
    multi_map = dup_counts[dup_counts > 1]
    print(f"MRNs mapping to >1 EMPI: {len(multi_map):,}")
    if len(multi_map) > 0:
        print("Example conflicting MRNs:", multi_map.index[:10].tolist())

    # 4 — Most frequent MRNs (optional, just to see data distribution)
    top_mrns = df["mrn_norm"].value_counts().head(5)
    print("Top 5 most frequent MRNs:")
    print(top_mrns)

# Run the check
lookup_path = "./ehr_store/lookups/mrn_empi_map.parquet"
if Path(lookup_path).exists():
    check_mrn_lookup(lookup_path)
else:
    print("Lookup file not found:", lookup_path)
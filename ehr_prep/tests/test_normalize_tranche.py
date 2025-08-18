# How to run: 
# pytest -q (from project root where ehr_prep/ lives)
import os
from pathlib import Path
import pyarrow as pa
import pytest
import pyarrow.parquet as pq, glob


from ehr_prep.normalize import normalize_tranche

# ---------- helpers ----------

def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _read_all_parquet_rows(dirpath: Path):
    parts = sorted(dirpath.glob("*.parquet"))
    assert parts, f"No parquet parts in {dirpath}"
    total = 0
    tables = []
    for p in parts:
        t = pq.read_table(p)
        tables.append(t)
        total += t.num_rows
    # also return a concatenated table for convenience
    return total, pa.concat_tables(tables, promote=True)

# ---------- fixtures ----------

@pytest.fixture
def tranche_tmp(tmp_path: Path):
    """
    Creates a tiny tranche folder with:
      - RAD (free-text, 2 reports)
      - PHY (row-based, 3 rows)
      - ignore: LET + LOG
    """
    tdir = tmp_path / "tranche_01"

    # --- RAD-like free-text file (pipe header + free text ending with [report_end])
    rad_txt = """EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text
E1|P1|MGH|M1|R0001|5/9/2005 9:50:00 AM|CT CHEST|F|MRRADXR|First line of report
IMPRESSION: something
FINDINGS: bla

[report_end]

E1|P1|MGH|M1|R0002|2011-03-16 11:25 AM|XR SHOULDER|F|MRRADXR|Another report body
Line 2

[report_end]
"""
    _write(tdir / "FF12_2025_RAD.txt", rad_txt)

    # --- PHY-like row-based file (pipe CSV with one row = one entry)
    phy_txt = """EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Concept_Name|Code_Type|Code|Result|Units|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number
E1|P1|MGH|M1|10/14/2016|Flu-High Dose|EPIC|76|||Dr A|CL1|MGH|Outpatient|EPIC-001
E1|P1|MGH|M1|2017-01-03|Blood Pressure|EPIC|BP|120/80|mmHg|Dr B|CL2|MGH|Outpatient|EPIC-002
E1|P1|MGH|M1|03/01/2018|Weight|EPIC|WT|70|kg|Dr C|CL3|MGH|Inpatient|EPIC-003
"""
    _write(tdir / "FF12_2025_PHY.txt", phy_txt)

    # --- Files that should be ignored
    _write(tdir / "foo_let.txt", "this should be ignored\n")
    _write(tdir / "bar_log.txt", "this should be ignored\n")

    return tdir

# ---------- tests ----------

def test_normalize_tranche_outputs_expected_structure(tranche_tmp: Path, tmp_path: Path):
    out_root = tmp_path / "ehr_store" / "normalized"
    normalize_tranche(str(tranche_tmp), str(out_root), "T01")
    rad_parts = glob.glob((out_root / "tranche=T01" / "mod=RAD" / "*.parquet").as_posix())
    t = pq.read_table(rad_parts[0], columns=["raw_text"])
    print("\n--- RAD raw_text rows ---")
    for s in t.column("raw_text").to_pylist():
        print(repr(s))

    # directories per modality present
    rad_dir = out_root / "tranche=T01" / "mod=RAD"
    phy_dir = out_root / "tranche=T01" / "mod=PHY"
    assert rad_dir.exists(), "RAD output dir missing"
    assert phy_dir.exists(), "PHY output dir missing"

    # ignored modalities not present
    let_dir = out_root / "tranche=T01" / "mod=LET"
    assert not let_dir.exists(), "LET should be ignored"

    # parquet parts exist
    assert list(rad_dir.glob("*.parquet")), "No RAD parquet written"
    assert list(phy_dir.glob("*.parquet")), "No PHY parquet written"

def test_free_text_parsing_and_fields(tranche_tmp: Path, tmp_path: Path):
    out_root = tmp_path / "ehr_store" / "normalized"
    normalize_tranche(str(tranche_tmp), str(out_root), "T01")
    rad_parts = glob.glob(os.path.join(out_root.as_posix(), "tranche=T01", "mod=RAD", "*.parquet"))
    t = pq.read_table(rad_parts[0], columns=["raw_text"])
    print("\n--- RAD raw_text rows from L1 ---")
    for s in t.column("raw_text").to_pylist():
        print(repr(s))

    rad_dir = out_root / "tranche=T01" / "mod=RAD"
    n_rows, table = _read_all_parquet_rows(rad_dir)
    assert n_rows == 2, f"Expected 2 RAD reports, got {n_rows}"

    cols = {c for c in table.column_names}
    # Required schema columns
    for col in (
        "source_tranche","source_file","line_start","line_end",
        "patient_empi","patient_mrn","modality","doc_id","is_free_text",
        "raw_text","report_end_marker","doc_date","performed_date",
        "section_hints","modality_specific"
    ):
        assert col in cols, f"Missing RAD column: {col}"

    # Values
    mod = table.column("modality").to_pylist()
    assert set(mod) == {"RAD"}
    is_ft = table.column("is_free_text").to_pylist()
    assert all(is_ft), "RAD entries must be marked free-text"

    # doc_date parsed (both US-style and ISO-style)
    dates = table.column("doc_date").to_pylist()
    assert any(str(x) == "2005-05-09" for x in dates), "Did not parse 5/9/2005"
    assert any(str(x) == "2011-03-16" for x in dates), "Did not parse 2011-03-16"

    # [report_end] respected -> raw_text not empty
    texts = table.column("raw_text").to_pylist()
    assert all(isinstance(t, str) and len(t) > 5 for t in texts)

    # line_start < line_end
    ls = table.column("line_start").to_pylist()
    le = table.column("line_end").to_pylist()
    assert all(a < b for a, b in zip(ls, le)), "line ranges should increase"

def test_row_based_parsing_and_fields(tranche_tmp: Path, tmp_path: Path):
    out_root = tmp_path / "ehr_store" / "normalized"
    normalize_tranche(str(tranche_tmp), str(out_root), "T01")
    rad_parts = glob.glob(os.path.join(out_root.as_posix(), "tranche=T01", "mod=RAD", "*.parquet"))
    t = pq.read_table(rad_parts[0], columns=["raw_text"])
    print("\n--- RAD raw_text rows from L1 ---")
    for s in t.column("raw_text").to_pylist():
        print(repr(s))

    phy_dir = out_root / "tranche=T01" / "mod=PHY"
    n_rows, table = _read_all_parquet_rows(phy_dir)
    assert n_rows == 3, f"Expected 3 PHY rows, got {n_rows}"

    # Row-based flags
    is_ft = table.column("is_free_text").to_pylist()
    assert not any(is_ft), "PHY rows should be non-free-text"
    end_marker = table.column("report_end_marker").to_pylist()
    assert not any(end_marker), "Row-based files must not set report_end_marker"

    # Date parsing across formats
    dates = [str(x) for x in table.column("doc_date").to_pylist()]
    assert "2016-10-14" in dates
    assert "2017-01-03" in dates
    assert "2018-03-01" in dates

    # modality_specific JSON should carry small fields like Concept_Name
    meta = table.column("modality_specific").to_pylist()
    assert any(m and "Concept_Name" in m for m in meta), "Expected Concept_Name in meta JSON"

def test_ignore_let_and_log_files(tranche_tmp: Path, tmp_path: Path):
    out_root = tmp_path / "ehr_store" / "normalized"
    normalize_tranche(str(tranche_tmp), str(out_root), "T01")
    rad_parts = glob.glob(os.path.join(out_root.as_posix(), "tranche=T01", "mod=RAD", "*.parquet"))
    t = pq.read_table(rad_parts[0], columns=["raw_text"])
    print("\n--- RAD raw_text rows from L1 ---")
    for s in t.column("raw_text").to_pylist():
        print(repr(s))

    # ensure no output folder got created for ignored files
    forbidden = [
        out_root / "tranche=T01" / "mod=LET",
        out_root / "tranche=T01" / "mod=LOG",
    ]
    for p in forbidden:
        assert not p.exists(), f"Should not create output for ignored modality: {p}"
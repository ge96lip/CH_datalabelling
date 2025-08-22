# ehr_prep/tests/test_layer2_from_layer1_flow.py
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from ehr_prep.normalize import normalize_tranche
from ehr_prep.build_timeline import build_timeline_ds, load_patient_text_from_dataset

# ---------- helpers ----------
def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# ---------- fixtures (reusing Layer-1 temp raw tranche, but richer) ----------
@pytest.fixture
def tranche_tmp(tmp_path: Path):
    """
    Creates a tiny tranche folder with:
      - RAD (free-text, 3 reports E1, 2 reports E2) with [report_end] and blank lines
      - PHY (row-based, 2 rows E1, 1 row E2) with short 'raw_text' content
      - ignore: LET + LOG
    """
    tdir = tmp_path / "tranche_01"

    # RAD free-text file — mixed date styles, sections, blank lines, special chars
    rad_txt = """EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text
E1|P1|MGH|M1|R0001|5/9/2005 9:50:00 AM|CT CHEST|F|MRRADXR|Header ignored by parser body starts below
IMPRESSION: mild atelectasis
FINDINGS: lungs clear; no consolidation.

[report_end]

E1|P1|MGH|M1|R0002|2011-03-16 11:25 AM|XR SHOULDER|F|MRRADXR|Header ignored by parser body starts below
IMPRESSION: suspected tendinopathy
FINDINGS: shoulder joint space preserved.

[report_end]

E1|P1|MGH|M1|R0003|2011-04-01 09:00 AM|CT CHEST|F|MRRADXR|Header ignored by parser body starts below
IMPRESSION: improving findings
FINDINGS: small nodules; recommend follow-up 6-12 mo.

[report_end]

E2|P2|MGH|M2|R1001|2010-12-31 08:00 AM|XR CHEST|F|MRRADXR|Header ignored
IMPRESSION: baseline exam
FINDINGS: no acute process.

[report_end]

E2|P2|MGH|M2|R1002|2011-01-15 10:30 AM|XR CHEST|F|MRRADXR|Header ignored
IMPRESSION: subtle interstitial markings
FINDINGS: likely chronic.

[report_end]
"""
    _write(tdir / "FF12_2025_RAD.txt", rad_txt)

    # PHY row-based file — include short texts so Layer-2 concat is meaningful
    phy_txt = """EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Concept_Name|Code_Type|Code|Result|Units|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number
E1|P1|MGH|M1|2011-03-20|Blood Pressure|EPIC|BP|120/80|mmHg|Dr Alpha|CL1|MGH|Outpatient|EPIC-001
E1|P1|MGH|M1|2011-03-25|Weight|EPIC|WT|70|kg|Dr Beta|CL2|MGH|Outpatient|EPIC-002
E2|P2|MGH|M2|2011-01-01|Heart Rate|EPIC|HR|75|bpm|Dr Gamma|CLX|MGH|Inpatient|EPIC-009
"""
    _write(tdir / "FF12_2025_PHY.txt", phy_txt)

    # Ignored files
    _write(tdir / "foo_let.txt", "ignore me\n")
    _write(tdir / "bar_log.txt", "ignore me\n")

    return tdir

# ---------- tests ----------

def test_end_to_end_layer2_from_layer1(tranche_tmp: Path, tmp_path: Path):
    """
    Full flow:
      raw tranche -> normalize_tranche (Layer-1) -> build_timeline_ds (Layer-2)
    Then verify:
      - both patients exist,
      - texts concatenated chronologically,
      - manifests consistent.
    """
    # 1) Build Layer-1 from the raw temp tranche
    out_root = tmp_path / "ehr_store" / "normalized"
    normalize_tranche(str(tranche_tmp), str(out_root), "T01")
    rad_parts = list((out_root / "tranche=T01" / "mod=RAD").glob("*.parquet"))
    t = pq.read_table(rad_parts[0], columns=["patient_empi", "raw_text", "doc_date"])
    print("\n--- L1 RAD rows ---")
    for empi, body, d in zip(t.column("patient_empi").to_pylist(),
                            t.column("raw_text").to_pylist(),
                            t.column("doc_date").to_pylist()):
        print("EMPI=", empi, "date=", d, "text starts:", repr((body or "")[:40]))

    # Sanity: L1 parts exist for both modalities
    assert list((out_root / "tranche=T01" / "mod=RAD").glob("*.parquet")), "No RAD L1 parquet"
    assert list((out_root / "tranche=T01" / "mod=PHY").glob("*.parquet")), "No PHY L1 parquet"

    # 2) Build Layer-2 from the Layer-1 parquet
    l1_glob = (out_root / "tranche=*" / "mod=*" / "*.parquet").as_posix()
    l2_dir  = (tmp_path / "ehr_store" / "timeline_ds").as_posix()
    build_timeline_ds(l1_glob=l1_glob, l2_dir=l2_dir, n_buckets=16, overwrite=True)

    # 3) Patient completeness via dataset and manifest
    dataset = ds.dataset(l2_dir, format="parquet")
    tbl = dataset.to_table(columns=["patient_id"])
    patients = sorted(set(tbl.column("patient_id").to_pylist()))
    assert patients == ["E1", "E2"], f"Unexpected patients: {patients}"

    #m_dir = tmp_path / "ehr_store" / "manifests"
    m_dir = (Path(l2_dir).parent / "manifests")
    idx = pq.read_table(m_dir / "v1_patient_index.parquet")
    assert sorted(set(idx.column("patient_id").to_pylist())) == ["E1", "E2"]

    # 4) Load each patient as single text (chronological concatenation)
    text_e1 = load_patient_text_from_dataset("E1", ds_root=l2_dir)
    text_e2 = load_patient_text_from_dataset("E2", ds_root=l2_dir)

    # For E1, expected chronological order:
    # 2005-05-09 RAD, 2011-03-16 RAD, 2011-03-20 PHY, 2011-03-25 PHY, 2011-04-01 RAD
    exp_e1 = "\n\n".join([
        "IMPRESSION: mild atelectasis\nFINDINGS: lungs clear; no consolidation.",
        "IMPRESSION: suspected tendinopathy\nFINDINGS: shoulder joint space preserved.",
        "Blood Pressure 120/80 mmHg",  # PHY row-based text comes from Result column via normalizer -> raw_text
        "Weight 70 kg",
        "IMPRESSION: improving findings\nFINDINGS: small nodules; recommend follow-up 6-12 mo.",
    ])
    assert text_e1 == exp_e1, f"E1 text mismatch.\nGot:\n{text_e1}\nExpected:\n{exp_e1}"

    # For E2, expected chronological order:
    # 2010-12-31 RAD, 2011-01-01 PHY, 2011-01-15 RAD
    exp_e2 = "\n\n".join([
        "IMPRESSION: baseline exam\nFINDINGS: no acute process.",
        "Heart Rate 75 bpm",
        "IMPRESSION: subtle interstitial markings\nFINDINGS: likely chronic.",
    ])
    assert text_e2 == exp_e2, f"E2 text mismatch.\nGot:\n{text_e2}\nExpected:\n{exp_e2}"

    # 5) Manifest vs dataset row counts
    ds_rows = dataset.count_rows()
    mf = pq.read_table(m_dir / "v1_timeline_manifest.parquet")
    m_rows = sum(mf.column("n_rows").to_pylist())
    assert ds_rows == m_rows, f"Manifest rows {m_rows} != dataset rows {ds_rows}"

def test_filters_work_on_layer2(tranche_tmp: Path, tmp_path: Path):
    """
    Verify include_modalities and exclude_after filters against the dataset built
    from the tranche_tmp Layer-1.
    """
    out_root = tmp_path / "ehr_store" / "normalized"
    normalize_tranche(str(tranche_tmp), str(out_root), "T01")
    rad_parts = list((out_root / "tranche=T01" / "mod=RAD").glob("*.parquet"))
    t = pq.read_table(rad_parts[0], columns=["patient_empi", "raw_text", "doc_date"])
    print("\n--- L1 RAD rows ---")
    for empi, body, d in zip(t.column("patient_empi").to_pylist(),
                            t.column("raw_text").to_pylist(),
                            t.column("doc_date").to_pylist()):
        print("EMPI=", empi, "date=", d, "text starts:", repr((body or "")[:40]))

    l1_glob = (out_root / "tranche=*" / "mod=*" / "*.parquet").as_posix()
    l2_dir  = (tmp_path / "ehr_store" / "timeline_ds").as_posix()
    build_timeline_ds(l1_glob=l1_glob, l2_dir=l2_dir, n_buckets=16, overwrite=True)

    # Only RAD for E1 (should drop the PHY values 120/80 and 70)
    only_rad = load_patient_text_from_dataset("E1", ds_root=l2_dir, include_modalities=["RAD"])
    norm = only_rad.replace("–", "-").strip()

    expected_snippets = [
        "IMPRESSION: mild atelectasis",
        "FINDINGS: lungs clear; no consolidation.",
        "IMPRESSION: suspected tendinopathy",
        "FINDINGS: shoulder joint space preserved.",
        "IMPRESSION: improving findings",
        "FINDINGS: small nodules; recommend follow-up 6-12 mo.",
    ]
    start = 0
    for snip in expected_snippets:
        pos = norm.find(snip, start)
        assert pos != -1, f"Missing snippet in RAD text: {snip}\nGot:\n{norm}"
        start = pos
    # Cut off after 2011-03-20 (inclusive): expect RAD(2005), RAD(2011-03-16), PHY(2011-03-20)
    cutoff = load_patient_text_from_dataset("E1", ds_root=l2_dir, exclude_after="2011-03-20")
    exp_cutoff = "\n\n".join([
        "IMPRESSION: mild atelectasis\nFINDINGS: lungs clear; no consolidation.",
        "IMPRESSION: suspected tendinopathy\nFINDINGS: shoulder joint space preserved.",
        "Blood Pressure 120/80 mmHg",
    ])
    assert cutoff == exp_cutoff
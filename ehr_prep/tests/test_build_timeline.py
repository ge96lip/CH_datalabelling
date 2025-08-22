# ehr_prep/tests/test_build_timeline_ds.py
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
import pytest

from build_timeline import build_timeline_ds, load_patient_text_from_dataset

# Minimal set of L1 columns the builder reads/casts
_L1_COLS = [
    "source_tranche", "source_file", "line_start", "line_end",
    "patient_empi", "patient_mrn", "modality", "doc_id", "is_free_text",
    "raw_text", "report_end_marker", "doc_date", "performed_date",
    "section_hints", "modality_specific", "hospital", "inpatient_outpatient"
]

def _write_l1_part(path: Path, rows: list[dict]):
    """ 
    Writes a Layer-1 part to the given path.
    Ensures the directory exists and normalizes the rows to _L1_COLS.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure all columns present, fill missing with None
    norm_rows = []
    for r in rows:
        nr = {k: r.get(k) for k in _L1_COLS}
        norm_rows.append(nr)
    table = pa.Table.from_pylist(norm_rows)
    pq.write_table(table, path)

@pytest.fixture
def l1_fixture(tmp_path: Path):
    """
    Creates a tiny Layer-1 layout under:
      tmp/ehr_store/normalized/tranche=T01/mod=RAD/part-1.parquet
      tmp/ehr_store/normalized/tranche=T01/mod=PHY/part-1.parquet

    Patients:
      - EMPI_A: 3 events total (RAD@2020-01-05, PHY@2020-01-10, RAD@2020-02-01)
      - EMPI_B: 2 events total (RAD@2019-12-31, PHY@2020-01-15)
    """
    root = tmp_path / "ehr_store" / "normalized_v2"
    # RAD (free-text-like)
    _write_l1_part(
        root / "tranche=T01" / "mod=RAD" / "part-1.parquet",
        [
            # EMPI_A
            dict(source_tranche="T01", source_file="rad.txt", line_start=1, line_end=5,
                 patient_empi="EMPI_A", patient_mrn="M1", modality="RAD", doc_id="A_RAD_1",
                 is_free_text=True, raw_text="RAD A @ 2020-01-05", report_end_marker=True,
                 doc_date="2020-01-05", performed_date=None,
                 section_hints=None, modality_specific='{"exam_type":"CT"}', hospital="MMG", inpatient_outpatient=None),
            dict(source_tranche="T01", source_file="rad.txt", line_start=6, line_end=12,
                 patient_empi="EMPI_A", patient_mrn="M1", modality="RAD", doc_id="A_RAD_2",
                 is_free_text=True, raw_text="RAD A @ 2020-02-01", report_end_marker=True,
                 doc_date="2020-02-01", performed_date=None,
                 section_hints=None, modality_specific='{"exam_type":"XR"}', hospital="MMG", inpatient_outpatient=None),
            # EMPI_B
            dict(source_tranche="T01", source_file="rad.txt", line_start=13, line_end=20,
                 patient_empi="EMPI_B", patient_mrn="M2", modality="RAD", doc_id="B_RAD_1",
                 is_free_text=True, raw_text="RAD B @ 2019-12-31", report_end_marker=True,
                 doc_date="2019-12-31", performed_date=None,
                 section_hints=None, modality_specific='{"exam_type":"CT"}', hospital="MMG", inpatient_outpatient=None),
        ]
    )

    # PHY (row-based-like; text present to test concatenation)
    _write_l1_part(
        root / "tranche=T01" / "mod=PHY" / "part-1.parquet",
        [
            dict(source_tranche="T01", source_file="phy.txt", line_start=1, line_end=1,
                 patient_empi="EMPI_A", patient_mrn="M1", modality="PHY", doc_id="A_PHY_1",
                 is_free_text=False, raw_text="PHY A @ 2020-01-10", report_end_marker=False,
                 doc_date="2020-01-10", performed_date=None,
                 section_hints=None, modality_specific='{"Concept_Name":"Vitals"}', hospital="MMG", inpatient_outpatient="Inpatient"),
            dict(source_tranche="T01", source_file="phy.txt", line_start=2, line_end=2,
                 patient_empi="EMPI_B", patient_mrn="M2", modality="PHY", doc_id="B_PHY_1",
                 is_free_text=False, raw_text="PHY B @ 2020-01-15", report_end_marker=False,
                 doc_date="2020-01-15", performed_date=None,
                 section_hints=None, modality_specific='{"Concept_Name":"BP"}', hospital="MMG", inpatient_outpatient="Outpatient"),
        ]
    )

    return tmp_path


def test_build_and_load_single_patients(l1_fixture: Path):
    tmp = l1_fixture
    l1_glob = (tmp / "ehr_store" / "normalized_v2" / "tranche=*" / "mod=*" / "*.parquet").as_posix()
    l2_dir  = (tmp / "ehr_store" / "timeline_ds_v2").as_posix()

    # Build Layer-2 (partitioned dataset)
    build_timeline_ds(l1_glob=l1_glob, l2_dir=l2_dir, n_buckets=16, overwrite=True)

    # --- Verify manifests exist
    m_dir = tmp / "ehr_store" / "manifests"
    assert (m_dir / "v1_timeline_manifest.parquet").exists()
    assert (m_dir / "v1_patient_index.parquet").exists()

    # --- Count patients via manifest
    idx = pq.read_table(m_dir / "v1_patient_index.parquet")
    n_patients = idx.num_rows  # one row per patient_id
    assert n_patients == 2, f"Expected 2 patients, got {n_patients}"
    print(f"[INFO] Patients present in timeline_ds: {n_patients}")

    # --- Sanity: distinct patients directly from dataset
    dataset = ds.dataset(l2_dir, format="parquet")
    tbl = dataset.to_table(columns=["patient_id"])
    distinct = set(tbl.column("patient_id").to_pylist())
    assert distinct == {"EMPI_A", "EMPI_B"}

    # --- Load each patient back as single text
    text_a = load_patient_text_from_dataset("EMPI_A", ds_root=l2_dir)
    text_b = load_patient_text_from_dataset("EMPI_B", ds_root=l2_dir)

    # Expected chronological concatenation (A: RAD 01-05, PHY 01-10, RAD 02-01)
    exp_a = "\n\n".join([
        "RAD A @ 2020-01-05",
        "PHY A @ 2020-01-10",
        "RAD A @ 2020-02-01",
    ])
    assert text_a == exp_a, f"EMPI_A text mismatch.\nGot:\n{text_a}\nExpected:\n{exp_a}"

    # Expected chronological concatenation (B: RAD 12-31, PHY 01-15)
    exp_b = "\n\n".join([
        "RAD B @ 2019-12-31",
        "PHY B @ 2020-01-15",
    ])
    assert text_b == exp_b, f"EMPI_B text mismatch.\nGot:\n{text_b}\nExpected:\n{exp_b}"

def test_modalities_and_cutoff_filters(l1_fixture: Path):
    tmp = l1_fixture
    l1_glob = (tmp / "ehr_store" / "normalized_v2" / "tranche=*" / "mod=*" / "*.parquet").as_posix()
    l2_dir  = (tmp / "ehr_store" / "timeline_ds_v2").as_posix()

    # Rebuild (idempotent/overwrite)
    build_timeline_ds(l1_glob=l1_glob, l2_dir=l2_dir, n_buckets=16, overwrite=True)

    # Only RAD for EMPI_A
    only_rad = load_patient_text_from_dataset("EMPI_A", ds_root=l2_dir, include_modalities=["RAD"])
    assert only_rad == "\n\n".join(["RAD A @ 2020-01-05", "RAD A @ 2020-02-01"])

    # Cut off after 2020-01-10 (inclusive)
    cutoff = load_patient_text_from_dataset("EMPI_A", ds_root=l2_dir, exclude_after="2020-01-10")
    assert cutoff == "\n\n".join(["RAD A @ 2020-01-05", "PHY A @ 2020-01-10"])

def test_bucket_completeness(l1_fixture: Path):
    """
    Ensures all patients appear in some bucket and that the manifest's
    row counts equal the dataset's row counts.
    """
    tmp = l1_fixture
    l1_glob = (tmp / "ehr_store" / "normalized_v2" / "tranche=*" / "mod=*" / "*.parquet").as_posix()
    l2_dir  = (tmp / "ehr_store" / "timeline_ds_v2").as_posix()
    m_dir   = tmp / "ehr_store" / "manifests"

    build_timeline_ds(l1_glob=l1_glob, l2_dir=l2_dir, n_buckets=16, overwrite=True)

    # Dataset total rows
    ds_rows = ds.dataset(l2_dir, format="parquet").count_rows()

    # Sum rows from the per-bucket manifest
    mf = pq.read_table(m_dir / "v1_timeline_manifest.parquet")
    m_rows = sum(mf.column("n_rows").to_pylist())

    assert ds_rows == m_rows, f"Manifest row sum {m_rows} != dataset rows {ds_rows}"

    # Print total patients (as requested)
    idx = pq.read_table(m_dir / "v1_patient_index.parquet")
    print(f"[INFO] Total patients in timeline_ds: {idx.num_rows}")
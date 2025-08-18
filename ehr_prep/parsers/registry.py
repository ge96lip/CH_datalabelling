# ehr_prep/parsers/registry.py
from dataclasses import dataclass
from typing import Optional, Tuple
import re, os

@dataclass(frozen=True)
class ModalitySpec:
    code: str
    free_text: bool
    header_date_keys: Tuple[str, ...]        # where to read doc_date from the header/row
    performed_date_keys: Tuple[str, ...]     # optional secondary date
    keep_meta: Tuple[str, ...]               # small fields to keep in meta.modality_specific
    row_text_fields: Tuple[str, ...] = ()

# map filename suffix (case-insensitive) → spec
REGISTRY = {
    # --- Row-based (no narrative body) ---
    "dem": ModalitySpec(
        "DEM", False,
        ("Date_of_Birth",), (),
        ("Gender_Legal_Sex", "Vital_status", "Date_Of_Death"),
        row_text_fields=()  # keep empty; we don't want to surface PII here
    ),
    "enc": ModalitySpec(
        "ENC", False,
        ("Admit_Date", "Discharge_Date"), ("Admit_Date", "Discharge_Date"),
        ("Hospital", "Inpatient_Outpatient", "Encounter_number", "Service_Line", "Attending_MD"),
        row_text_fields=("Service_Line", "Inpatient_Outpatient")
    ),
    "med": ModalitySpec(
        "MED", False,
        ("Medication_Date", "Medication_Date_Detail"), (),
        ("Medication", "Code_Type", "Code", "Quantity", "Hospital", "Inpatient_Outpatient", "Encounter_number"),
        row_text_fields=("Medication",)
    ),
    "phy": ModalitySpec(
        "PHY", False,
        ("Date",), (),
        ("Concept_Name", "Code_Type", "Code", "Result", "Units", "Hospital", "Inpatient_Outpatient", "Encounter_number"),
        row_text_fields=("Result",)  # simplest; if you want "70 kg" we can add a composer later
    ),
    "prc": ModalitySpec(
        "PRC", False,
        ("Date",), (),
        ("Procedure_Name", "Code_Type", "Code", "Procedure_Flag", "Quantity", "Hospital", "Inpatient_Outpatient", "Encounter_number"),
        row_text_fields=("Procedure_Name",)
    ),
    "ptd": ModalitySpec(
        "PTD", False,
        ("Start_Date",), ("End_Date",),
        ("Patient_Data_Type", "Description", "Result", "Encounter_ID"),
        row_text_fields=("Description", "Result")
    ),
    "rdt": ModalitySpec(
        "RDT", False,
        ("Date",), (),
        ("Mode", "Group", "Test_Code", "Test_Description", "Accession_Number", "Hospital", "Inpatient_Outpatient"),
        row_text_fields=("Test_Description",)
    ),
    "rfv": ModalitySpec(
        "RFV", False,
        ("Start_date",), ("End_date",),
        ("Chief_complaint", "Concept_id", "Encounter_number", "Hospital", "Clinic"),
        row_text_fields=("Chief_complaint",)
    ),
    "dia": ModalitySpec(
        "DIA", False,
        ("Date",), (),
        ("Diagnosis_Name", "Code_Type", "Code", "Diagnosis_Flag", "Provider", "Clinic", "Hospital", "Inpatient_Outpatient", "Encounter_number"),
        row_text_fields=("Diagnosis_Name",)
    ),
    # Free-text families (have Report_Text & [report_end])
    "dis": ModalitySpec("DIS", True, ("Report_Date_Time",), (), ("Report_Description","Report_Status","Report_Type")),
    "hnp": ModalitySpec("HNP", True, ("Report_Date_Time",), (), ("Report_Description","Report_Status","Report_Type")),
    "opn": ModalitySpec("OPN", True, ("Report_Date_Time",), (), ("Report_Description","Report_Status","Report_Type")),
    "pat": ModalitySpec("PAT", True, ("Report_Date_Time",), (), ("Report_Description","Report_Status","Report_Type")),
    "prg": ModalitySpec("PRG", True, ("Report_Date_Time",), (), ("Report_Description","Report_Status","Report_Type")),
    "pul": ModalitySpec("PUL", True, ("Report_Date_Time",), (), ("Report_Description","Report_Status","Report_Type")),
    "rad": ModalitySpec("RAD", True, ("Report_Date_Time",), (), ("Report_Description","Report_Status","Report_Type")),
    "vis": ModalitySpec("VIS", True, ("Report_Date_Time",), (), ("Report_Description","Report_Status","Report_Type")),
    "end": ModalitySpec("END", True, ("Report_Date_Time",), (), ("Report_Description","Report_Status","Report_Type")),
    "lno": ModalitySpec("LNO", True, ("LMRNote_Date",), (), ("Status","Author","Institution","Subject")),
    # DIA = Diagnoses (row-based)
    # ignore “let” (none), “log” (download logs), “qry/desc”
}

_MODALITY_RE = re.compile(
    r'(?:^|[._-])(' + '|'.join(map(re.escape, REGISTRY.keys())) + r')' +
    r'(?:[._-][^/\\]+|\d+)?\.txt$',
    flags=re.IGNORECASE,
)

# precompile boundary-aware patterns once
_BOUNDARY = r'(?:^|[._-])'   # start or one of dot/underscore/dash
def _make_pat(key: str) -> re.Pattern:
    return re.compile(rf'{_BOUNDARY}{re.escape(key)}(?:$|[._-])')

_PATTERNS = {key: _make_pat(key) for key in REGISTRY}

def guess_modality_from_name(filename: str) -> Optional[ModalitySpec]:
    base = os.path.basename(filename).lower()
    stem = base[:-4] if base.endswith(".txt") else base  # drop extension, match on stem
    for key, spec in REGISTRY.items():
        if _PATTERNS[key].search(stem):
            return spec
    return None
# ehr_prep/parsers/registry.py
from dataclasses import dataclass
from typing import Optional, Tuple
import re, os

@dataclass(frozen=True)
class ModalitySpec:
    code: str
    free_text: bool
    # TODO: exclude the Tuple because it's max one string anyways 
    header_date_keys: Tuple[str, ...]        # where to read doc_date from the header/row
    performed_date_keys: Tuple[str, ...]     # optional secondary date
    keep_meta: Tuple[str, ...]               # small fields to keep in meta.modality_specific
    row_text_fields: Tuple[str, ...] = ()    # only used for row-based parsers

REGISTRY = {
    # ---------------------------
    # Row-based (no narrative body)
    # ---------------------------
    "dem": ModalitySpec(
        "DEM", False,
        ("Date_of_Birth",), ("Date_Of_Death",),  # start: date_of_birth; end: date_of_death
        (
            "MRN_Type",
            "Date_of_Birth", "Age", "Sex_At_Birth",
            "Is_a_veteran", "Zip_code", "Country",
            "Vital_status", "Date_Of_Death",
        ),
        row_text_fields=()  # keep DEM body empty to avoid leaking extra PII
    ),

    "enc": ModalitySpec(
        "ENC", False,
        ("Admit_Date",), ("Discharge_Date",),  # split, as you suggested
        (
            "MRN_Type", "Encounter_Status", "Inpatient_Outpatient",
            "Admit_Date", "Discharge_Date",
            "Clinic_Name", "Admit_Source", "Discharge_Disposition",
            "Admitting_Diagnosis", "Principal_Diagnosis",
            "Diagnosis_1", "Diagnosis_2", # excluded _3-_10
            "DRG", "Patient_Type", "Referrer_Discipline",
        ),
        # a compact narrative cue (optional; trim if to get fewer tokens)
        row_text_fields=("Principal_Diagnosis", "Referrer_Discipline", "Clinic_Name")
    ),

    "med": ModalitySpec(
        "MED", False,
        ("Medication_Date",), ("Medication_Date_Detail",),
        (
            "MRN_Type", "Medication_Date", "Medication_Date_Detail",
            "Medication", "Code_Type", "Code", "Quantity",
            "Provider", "Clinic", "Inpatient_Outpatient", "Additional_Info",
        ),
        row_text_fields=("Medication",)  # short cue for the narrative
    ),

    "phy": ModalitySpec(
        "PHY", False,
        ("Date",), (),
        (
            "MRN_Type", "Date",
            "Concept_Name", "Code_Type", "Code",
            "Result", "Units",
            "Clinic", "Inpatient_Outpatient",
        ),
        row_text_fields=("Concept_Name", "Result", "Units")
    ),

    "prc": ModalitySpec(
        "PRC", False,
        ("Date",), (),
        (
            "MRN_Type", "Date",
            "Procedure_Name", "Code_Type", "Code",
            "Procedure_Flag", "Quantity",
            "Clinic", "Inpatient_Outpatient",
        ),
        row_text_fields=("Procedure_Name",)
    ),

    "ptd": ModalitySpec(
        "PTD", False,
        ("Start_Date",), ("End_Date",),
        (
            "MRN_Type", "Patient_Data_Type", "Description", "Result",
            "Start_Date", "End_Date",
        ),
        row_text_fields=("Description", "Result")
    ),

    "rdt": ModalitySpec(
        "RDT", False,
        ("Date",), (),
        (
            "MRN_Type", "Date", "Mode", "Group",
            "Test_Code", "Test_Description",
            "Clinic", "Inpatient_Outpatient",
        ),
        row_text_fields=("Test_Description",)
    ),

    "rfv": ModalitySpec(
        "RFV", False,
        ("Start_date",), ("End_date",),
        (
            "MRN_Type",    
            "Start_date", "End_date",
            "Clinic", "Chief_complaint", "Concept_id", "Comments",
        ),
        row_text_fields=("Chief_complaint",)
    ),

    "dia": ModalitySpec(
        "DIA", False,
        ("Date",), (),
        (
            "MRN_Type", "Date",
            "Diagnosis_Name", "Code_Type", "Code", "Diagnosis_Flag",
            "Provider", "Inpatient_Outpatient",
        ),
        row_text_fields=("Diagnosis_Name",)
    ),

    # ---------------------------
    # Free-text families (have Report_Text & [report_end])
    # row_text_fields are ignored for these; keep_meta is persisted.
    # ---------------------------
    "dis": ModalitySpec(
        "DIS", True,
        ("Report_Date_Time",), (),
        ("MRN_Type", "Report_Description", "Report_Status", "Report_Type")
    ),
    "hnp": ModalitySpec(
        "HNP", True,
        ("Report_Date_Time",), (),
        ("MRN_Type", "Report_Description", "Report_Status", "Report_Type")
    ),
    "opn": ModalitySpec(
        "OPN", True,
        ("Report_Date_Time",), (),
        ("MRN_Type", "Report_Description", "Report_Status", "Report_Type")
    ),
    "pat": ModalitySpec(
        "PAT", True,
        ("Report_Date_Time",), (),
        ("MRN_Type", "Report_Description", "Report_Status", "Report_Type")
    ),
    "prg": ModalitySpec(
        "PRG", True,
        ("Report_Date_Time",), (),
        ("MRN_Type", "Report_Description", "Report_Status", "Report_Type")
    ),
    "pul": ModalitySpec(
        "PUL", True,
        ("Report_Date_Time",), (),
        ("MRN_Type", "Report_Description", "Report_Status", "Report_Type")
    ),
    "rad": ModalitySpec(
        "RAD", True,
        ("Report_Date_Time",), (),
        ("MRN_Type", "Report_Description", "Report_Status", "Report_Type")
    ),
    "vis": ModalitySpec(
        "VIS", True,
        ("Report_Date_Time",), (),
        ("MRN_Type", "Report_Number", "Report_Description", "Report_Status", "Report_Type")
    ),
    "end": ModalitySpec(
        "END", True,
        ("Report_Date_Time",), (),
        ("MRN_Type", "Report_Status", "Report_Type")
    ),
    "lno": ModalitySpec(
        "LNO", True,
        ("LMRNote_Date",), (),
        ("MRN_Type", "Status", "Subject", "Comments")
    ),
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
# ehr_prep/parsers/registry.py
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModalitySpec:
    code: str
    free_text: bool
    header_date_keys: tuple        # where to read doc_date from the header/row
    performed_date_keys: tuple     # optional secondary date
    # meta fields that are worth keeping small in meta.modality_specific
    keep_meta: tuple

# map filename suffix (case-insensitive) → spec
REGISTRY = {
    "dem": ModalitySpec("DEM", False, ("Date_of_Birth",), (), ("Gender_Legal_Sex","Vital_status","Date_Of_Death")),
    "enc": ModalitySpec("ENC", False, ("Admit_Date","Discharge_Date"), ("Admit_Date","Discharge_Date"), ("Hospital","Inpatient_Outpatient","Encounter_number","Service_Line","Attending_MD")),
    "med": ModalitySpec("MED", False, ("Medication_Date","Medication_Date_Detail"), (), ("Medication","Code_Type","Code","Quantity","Hospital","Inpatient_Outpatient","Encounter_number")),
    "phy": ModalitySpec("PHY", False, ("Date",), (), ("Concept_Name","Code_Type","Code","Result","Units","Hospital","Inpatient_Outpatient","Encounter_number")),
    "prc": ModalitySpec("PRC", False, ("Date",), (), ("Procedure_Name","Code_Type","Code","Procedure_Flag","Quantity","Hospital","Inpatient_Outpatient","Encounter_number")),
    "ptd": ModalitySpec("PTD", False, ("Start_Date",), ("End_Date",), ("Patient_Data_Type","Description","Result","Encounter_ID")),
    "rdt": ModalitySpec("RDT", False, ("Date",), (), ("Mode","Group","Test_Code","Test_Description","Accession_Number","Hospital","Inpatient_Outpatient")),
    "rfv": ModalitySpec("RFV", False, ("Start_date",), ("End_date",), ("Chief_complaint","Concept_id","Encounter_number","Hospital","Clinic")),
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
    "dia": ModalitySpec("DIA", False,("Date",),(),("Diagnosis_Name", "Code_Type", "Code", "Diagnosis_Flag","Provider", "Clinic", "Hospital", "Inpatient_Outpatient", "Encounter_number")),
    # ignore “let” (none), “log” (download logs), “qry/desc”
}

def guess_modality_from_name(filename: str) -> Optional[ModalitySpec]:
    low = filename.lower()
    for key, spec in REGISTRY.items():
        if low.endswith(f"_{key}.txt") or low.endswith(f"-{key}.txt") or low.endswith(f".{key}.txt"):
            return spec
    return None
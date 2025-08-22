# ehr_prep/parsers/row_based.py
import csv, ctypes
import sys
from io_utils import ParquetSink, parse_date_safe, stable_id, pack_meta_small
from .registry import ModalitySpec
def _set_csv_field_limit():
    try:
        # Works on POSIX where C long is 64-bit
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        # Windows: C long is 32-bit; compute the max representable value
        bits = ctypes.sizeof(ctypes.c_long) * 8
        max_c_long = (1 << (bits - 1)) - 1          # e.g., 2_147_483_647 on Windows
        csv.field_size_limit(max_c_long)

_set_csv_field_limit()

def _compose_row_text(row, fields):
    vals = []
    for f in fields or ():
        v = row.get(f)
        if v is not None and str(v).strip() != "":
            vals.append(str(v).strip())
    return " ".join(vals)

def normalize_row_file(in_path: str, out_dir: str, tranche: str, spec: ModalitySpec, sink: ParquetSink):
    with open(in_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        rdr = csv.DictReader(f, delimiter="|")
        for i, row in enumerate(rdr, start=1):
            doc_date, perf_date = None, None
            for k in spec.header_date_keys:
                doc_date = doc_date or parse_date_safe(row.get(k))
            for k in spec.performed_date_keys:
                perf_date = perf_date or parse_date_safe(row.get(k))

            raw_text = row.get("Report_Text") or _compose_row_text(row, spec.row_text_fields)
            if not raw_text and getattr(spec, "row_text_fields", ()):
                raw_text = _compose_row_text(row, spec.row_text_fields)

            sink.write_one({
                "source_tranche": tranche,
                "source_file": in_path.split("/")[-1],
                "line_start": i,
                "line_end": i,
                "patient_empi": row.get("EMPI") or None,
                "patient_mrn": row.get("MRN") or None,
                "modality": spec.code,
                "doc_id": stable_id(tranche, spec.code, i),
                "report_sequence": None,
                "is_free_text": False,
                "raw_text": raw_text or "",    # some row-based have this col
                "report_end_marker": False,
                "doc_date": doc_date,
                "performed_date": perf_date,
                "ingest_timestamp": None,                   # set by driver if desired
                "section_hints": None,
                "modality_specific": pack_meta_small(row, spec.keep_meta),
            })
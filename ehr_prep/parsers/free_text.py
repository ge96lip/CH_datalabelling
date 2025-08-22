# ehr_prep/parsers/free_text.py
from io_utils import ParquetSink, parse_date_safe, stable_id, pack_meta_small
from .registry import ModalitySpec

END_TOKEN = "[report_end]"

def iter_free_text_records(path: str, tranche: str, spec: ModalitySpec):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = next(f)  # pipe header
        cols = [c.strip() for c in header.split("|")]
        cur = None
        buf = []
        line_no = 1

        for line in f:
            line_no += 1
            # beginning of a new report: header line with the same number of pipes
            if cur is None:
                if line.count("|") != len(cols)-1:
                    # skip stray blank lines
                    continue
                row = dict(zip(cols, line.rstrip("\n").split("|")))
                cur = {
                    "source_tranche": tranche,
                    "source_file": path.split("/")[-1],
                    "line_start": line_no,
                    "line_end": None,
                    "patient_empi": row.get("EMPI") or None,
                    "patient_mrn": row.get("MRN") or None,
                    "modality": spec.code,
                    "doc_id": stable_id(tranche, spec.code, line_no),
                    "report_sequence": None,
                    "is_free_text": True,
                    "raw_text": None,                # filled on close
                    "report_end_marker": True,
                    "doc_date": None,
                    "performed_date": None,
                    "ingest_timestamp": None,        # filled by caller
                    "section_hints": ["IMPRESSION","FINDINGS"] if spec.code=="RAD" else None,
                    "modality_specific": pack_meta_small(row, spec.keep_meta),
                }
                # dates from header
                for k in spec.header_date_keys:
                    cur["doc_date"] = cur["doc_date"] or parse_date_safe(row.get(k))
                for k in spec.performed_date_keys:
                    cur["performed_date"] = cur["performed_date"] or parse_date_safe(row.get(k))
                buf.clear()
                continue

            # streaming body
            if line.strip().lower() == END_TOKEN.lower():
                cur["raw_text"] = "".join(buf).rstrip()
                cur["line_end"] = line_no
                yield cur
                # consume the mandated blank line
                #_ = next(f, None); line_no += 1
                cur = None
                buf.clear()
            else:
                buf.append(line)

def normalize_free_text_file(in_path: str, out_dir: str, tranche: str, spec: ModalitySpec, sink: ParquetSink):
    for rec in iter_free_text_records(in_path, tranche, spec):
        rec["ingest_timestamp"] = None  # optional; fill in the driver if you want now()
        sink.write_one(rec)
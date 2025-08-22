# ehr_prep/normalize.py
from pathlib import Path
from tqdm import tqdm
from parsers.registry import guess_modality_from_name
from parsers.free_text import normalize_free_text_file
from parsers.row_based import normalize_row_file
from io_utils import ParquetSink, file_signature, ok_marker_path, write_ok_marker, read_ok_marker, failed_dir_for, timestamp_tag

IGNORE_KEYS = ("let", "log", "qry", "desc")

def normalize_tranche(tranche_dir: str, out_root: str, tranche_code: str):
    """
    tranche_dir: ./raw_files/tranche_01
    out_root   : ehr_store/normalized
    tranche_code: 'T01'...'T11'
    """
    tdir = Path(tranche_dir)
    failed = []
    failed_log = failed_dir_for(out_root) / f"failed-{tranche_code}-{timestamp_tag()}.txt"
    for path in tqdm(sorted(tdir.glob("*.txt")), desc=f"Normalize {tranche_code}"):
        low = path.name.lower()
        if any(low.endswith(f"_{k}.txt") for k in IGNORE_KEYS):
            tqdm.write(f"[SKIP ignore] {path.name} in IGNORE_KEYS")
            continue

        spec = guess_modality_from_name(path.name)
        if not spec:
            tqdm.write(f"[WARN unknown modality] {path.name} - CONTINUE")
            continue

        out_dir = Path(out_root) / f"tranche={tranche_code}" / f"mod={spec.code}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- resume/skip logic ---
        sig = file_signature(str(path))
        ok_path = ok_marker_path(str(out_dir), path.name)
        ok = read_ok_marker(ok_path)
        if ok and ok.get("sha256") == sig["sha256"] and ok.get("size") == sig["size"] and ok.get("mtime") == sig["mtime"]:
            tqdm.write(f"[SKIP ok] {spec.code} <- {path.name}; hash and size match")
            continue

        # --- process with a fresh sink; write to parts; then mark OK
        sink = ParquetSink(str(out_dir))
        try:
            if spec.free_text:
                normalize_free_text_file(str(path), str(out_dir), tranche_code, spec, sink)
            else:
                normalize_row_file(str(path), str(out_dir), tranche_code, spec, sink)
            sink.close()

            write_ok_marker(ok_path, {
                **sig,
                "tranche": tranche_code,
                "modality": spec.code,
                "source_file": path.name,
                "processed_at": timestamp_tag(),
            })
            tqdm.write(f"[OK] {spec.code} <- {path.name}, finished processing")

        except Exception as e:
            sink.close()
            failed.append(f"{path} :: {e}")
            tqdm.write(f"[FAIL] {spec.code} <- {path.name} :: {e}")

    if failed:
        failed_log.write_text("\n".join(failed), encoding="utf-8")
        print(f"[SUMMARY] {len(failed)} file(s) failed. See {failed_log}")
    else:
        print("[SUMMARY] All processed files OK (skipped previously completed ones).")
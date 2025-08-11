import os
import copy
import argparse
import math
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, balanced_accuracy_score, f1_score

# local
from model import RecurrenceModel, LabeledDataset, InferenceDataset

# ---------- ENV/SETUP ----------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TF"] = "1"

PATIENT_DIR = "./data/patient_txts"
REC_FILE    = "./data/labels/recurrence.txt"
NONREC_FILE = "./data/labels/non_recurrence.txt"
MAYBE_FILE  = "./data/labels/maybe_recurrence.txt"   # (optional) excluded

CHECKPOINT_PATH = "/Users/carlotta/physionet.org/files/dfci-cancer-outcomes-ehr/1.0.0/models/DFCI-student-medonc/dfci-student-medonc.pt"


# ---------- UTILS ----------
def seed_everything(seed: int = 42):
    import random
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True  # speeds up fixed-size inputs


def get_device(prefer: str = "auto") -> torch.device:
    prefer = prefer.lower()
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Silicon fallback
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return torch.device("mps")
        return torch.device("cpu")
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def _read_mrn_list(path):
    """
    Reads a list of MRNs from a file, filters out comments and empty lines,
    removes trailing '.0' from numeric MRNs, and returns a list of unique MRNs.

    Args:
        path (str): The file path to read MRNs from.

    Returns:
        list: A list of unique, cleaned MRNs as strings.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        mrns = [ln.strip() for ln in f.readlines()]
    mrns = [m for m in (x.strip() for x in mrns) if m and not m.startswith("#")]
    mrns = [m[:-2] if m.endswith(".0") and m.replace(".","",1).isdigit() else m for m in mrns]
    return list(dict.fromkeys(mrns))


def load_patient_text(mrn, patient_dir=PATIENT_DIR):
    """
    Loads and returns the text data for a patient given their medical record number (MRN).

    Args:
        mrn (str): The medical record number of the patient.
        patient_dir (str, optional): The directory containing patient text files. Defaults to PATIENT_DIR.

    Returns:
        str or None: The contents of the patient's text file as a string, or None if the file does not exist.
    """
    path = os.path.join(patient_dir, f"{mrn}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def create_df(path_recurrence=REC_FILE,
    path_no_recurrence=NONREC_FILE,
    path_maybe=MAYBE_FILE,
    patient_dir=PATIENT_DIR
    ):
    """
    Constructs a pandas DataFrame containing patient MRNs, associated text data, and recurrence labels.

    This function reads lists of patient MRNs from specified files for recurrence, non-recurrence, and uncertain cases ("maybe").
    It excludes MRNs present in the "maybe" list from both recurrence and non-recurrence lists, and removes any overlap between recurrence and non-recurrence lists.
    For each MRN, it loads the corresponding patient text data from the given directory. Only MRNs with available text data are included in the resulting DataFrame.

    Parameters
    ----------
    path_recurrence : str, optional
        Path to the file containing MRNs with recurrence (default: REC_FILE).
    path_no_recurrence : str, optional
        Path to the file containing MRNs without recurrence (default: NONREC_FILE).
    path_maybe : str, optional
        Path to the file containing MRNs with uncertain recurrence status (default: MAYBE_FILE).
    patient_dir : str, optional
        Directory containing patient text files (default: PATIENT_DIR).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - 'mrn': Patient MRN
            - 'text': Patient text data
            - 'label': Recurrence label (1 for recurrence, 0 for non-recurrence)

    Warns
    -----
    Prints warnings if MRNs are present in both recurrence and non-recurrence lists, or if text data is missing for any MRN.

    """
    
    rec_ids    = _read_mrn_list(path_recurrence)
    nonrec_ids = _read_mrn_list(path_no_recurrence)
    maybe_ids  = set(_read_mrn_list(path_maybe))
    print(f"Size of recurrence list: {len(rec_ids)}")
    print(f"Size of non-recurrence list: {len(nonrec_ids)}")

    rec_ids    = [m for m in rec_ids    if m not in maybe_ids]
    nonrec_ids = [m for m in nonrec_ids if m not in maybe_ids]
    print(f"After excluding maybes → rec={len(rec_ids)} nonrec={len(nonrec_ids)}")

    overlap = set(rec_ids).intersection(nonrec_ids)
    if overlap:
        print(f"[WARN] {len(overlap)} MRNs in BOTH lists. Removing from nonrec.")
        nonrec_ids = [m for m in nonrec_ids if m not in overlap]

    rows, missing = [], []
    for mrn, label in [(m, 1) for m in rec_ids] + [(m, 0) for m in nonrec_ids]:
        text = load_patient_text(mrn, patient_dir)
        if not text:
            missing.append(mrn); continue
        rows.append({"mrn": mrn, "text": text, "label": int(label)})

    if missing:
        print(f"[WARN] Missing/empty TXT for {len(missing)} MRNs (first 10): {missing[:10]}")

    df = pd.DataFrame(rows)
    print(f"Used in dataset: rec={df[df.label==1].mrn.nunique()} nonrec={df[df.label==0].mrn.nunique()} total_rows={len(df)}")
    return df


def train_val_split_df(df, val_size: float = 0.2, seed: int = 42):
    # (bug fix) test_size=None breaks stratify; set an explicit holdout
    return train_test_split(df, test_size=val_size, stratify=df["label"], random_state=seed)


def check_tokenization(train_df):
    from transformers import AutoTokenizer
    sample = train_df.iloc[0]
    tok = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer", truncation_side='left')
    enc = tok(sample['text'], padding='max_length', truncation=True, return_tensors='pt')
    print(tok.convert_ids_to_tokens(enc['input_ids'][0][:20]))


def build_model(load_checkpoint: bool = True) -> RecurrenceModel:
    """
    Builds and returns a RecurrenceModel instance, optionally loading weights from a checkpoint.

    If `load_checkpoint` is True and a checkpoint exists at `CHECKPOINT_PATH`, loads the model weights
    from the checkpoint, excluding weights for the 'recurrence_head'. Otherwise, initializes the model
    with default weights.

    After loading weights, freezes all model parameters except for those in the recurrence head and the
    last two encoder layers of the Longformer backbone, which are unfrozen for fine-tuning.

    Args:
        load_checkpoint (bool, optional): Whether to load weights from a checkpoint. Defaults to True.

    Returns:
        RecurrenceModel: The initialized and partially unfrozen model ready for training.
    """
    model = RecurrenceModel()
    if load_checkpoint and os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in ckpt.items() if k in model_dict and 'recurrence_head' not in k}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
        print(f"Loaded base weights from {CHECKPOINT_PATH} (excluding recurrence_head).")
    else:
        print("[INFO] No base checkpoint loaded.")

    # Freeze all, then unfreeze recurrence head + last 2 encoder layers
    for p in model.parameters(): p.requires_grad = False
    for p in model.recurrence_head.parameters(): p.requires_grad = True
    for name, p in model.longformer.named_parameters():
        if "encoder.layer.10." in name or "encoder.layer.11." in name:
            p.requires_grad = True
    return model


@dataclass
class TrainConfig:
    batch_size: int = 8
    val_batch_size: int = 8
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    num_workers: int = 2
    amp: bool = True  # mixed precision on CUDA/MPS
    save_best: bool = True
    out_dir: str = "./outputs"
    device_pref: str = "auto"  # "auto" | "cuda" | "cuda:0" | "mps" | "cpu"


def get_logits_from_outputs(model_outputs):
    """
    Adjust this if your RecurrenceModel returns logits differently.
    Current code expects outputs[3] to be the recurrence logits.
    """
    return model_outputs[3].squeeze(1)  # shape: (B,)


def make_dataloaders(train_df, val_df, cfg: TrainConfig):
    train_ds = LabeledDataset(train_df.reset_index(drop=True))
    val_ds   = LabeledDataset(val_df.reset_index(drop=True))
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.val_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    return train_loader, val_loader


def compute_class_pos_weight(df):
    """
    Computes the positive class weight for imbalanced binary classification.

    This function calculates the ratio of negative to positive samples in the provided DataFrame,
    which can be used as the `pos_weight` parameter in loss functions such as `torch.nn.BCEWithLogitsLoss`
    to address class imbalance.

    Args:
        df (pandas.DataFrame): DataFrame containing a 'label' column with binary values (0 for negative, 1 for positive).

    Returns:
        torch.Tensor: A scalar tensor representing the positive class weight. If one class is empty, returns 1.0.

    Warns:
        Prints a warning if either the positive or negative class is empty.
    """
    n_pos = int(df["label"].sum())
    n_neg = int((df["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        print("[WARN] One class is empty; pos_weight=1.0")
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(n_neg / max(1, n_pos), dtype=torch.float32)


def epoch_eval(model, loader, device, loss_fn, desc="val", amp=False):
    model.eval()
    losses, all_logits, all_y = [], [], []
    use_amp = (amp and (device.type in ["cuda", "mps"]))
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch[0].to(device)
            attn_mask = batch[1].to(device)
            y_true    = batch[2].to(device).float()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids, attn_mask)
                logits  = get_logits_from_outputs(outputs)
                loss    = loss_fn(logits, y_true)
            losses.append(loss.item())
            all_logits.append(logits.detach().cpu())
            all_y.append(y_true.detach().cpu())

    if not all_logits:
        return np.nan, np.nan, np.nan, {"thr@bestJ":np.nan,"acc@bestJ":np.nan,"bal_acc@bestJ":np.nan}

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))

    # Accuracy at fixed 0.5 (current behavior)
    acc_05 = accuracy_score(y_true, (probs >= 0.5).astype(int))

    # Threshold by Youden’s J (maximizes sensitivity+specificity-1)
    fpr, tpr, thr = roc_curve(y_true, probs)
    J = tpr - fpr
    j_idx = int(np.argmax(J))
    thr_bestJ = float(thr[j_idx]) if j_idx < len(thr) else 0.5
    preds_J   = (probs >= thr_bestJ).astype(int)
    acc_J     = accuracy_score(y_true, preds_J)
    balacc_J  = balanced_accuracy_score(y_true, preds_J)

    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = np.nan

    extra = {"thr@bestJ": thr_bestJ, "acc@bestJ": float(acc_J), "bal_acc@bestJ": float(balacc_J), "acc@0.5": float(acc_05)}
    return float(np.mean(losses)), float(acc_05), float(auc), extra


def train(model, train_df, val_df, cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.out_dir, "metrics.csv")
    best_path    = os.path.join(cfg.out_dir, "recurrence_model_best.pt")
    last_path    = os.path.join(cfg.out_dir, "recurrence_model_last.pt")

    device = get_device(cfg.device_pref)
    print(f"Using device: {device}")
    model.to(device)

    train_loader, val_loader = make_dataloaders(train_df, val_df, cfg)

    # class imbalance handling
    pos_weight = compute_class_pos_weight(train_df).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer / scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * math.ceil(len(train_loader) / max(1, cfg.grad_accum_steps))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # log headers
    metrics_df = pd.DataFrame(columns=[
        "epoch","train_loss","val_loss","val_acc_0p5","val_auc",
        "val_acc_bestJ","val_balacc_bestJ","val_thr_bestJ","lr"
    ])
    metrics_df.to_csv(metrics_path, index=False)

    best_val_loss = -1.0
    best_val_score = -1.0

    for epoch in range(1, cfg.epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        running = 0.0
        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader),
                    desc=f"Epoch {epoch}/{cfg.epochs} [train]")
        optimizer.zero_grad(set_to_none=True)

        autocast_enabled = (cfg.amp and (device.type in ["cuda", "mps"]))
        for step, batch in pbar:
            input_ids = batch[0].to(device)
            attn_mask = batch[1].to(device)
            y_true    = batch[2].to(device).float()

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                outputs = model(input_ids, attn_mask)
                logits  = get_logits_from_outputs(outputs)
                loss    = loss_fn(logits, y_true) / cfg.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.grad_accum_steps == 0:
                # gradient clipping for stability
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * cfg.grad_accum_steps
            train_losses.append(loss.item() * cfg.grad_accum_steps)
            pbar.set_postfix({"loss": f"{running/step:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan

        # ---- Validate ----
        val_loss, val_acc_05, val_auc, extra = epoch_eval(model, val_loader, device, loss_fn, desc="val", amp=cfg.amp)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc_0p5": val_acc_05,
            "val_auc": val_auc,
            "val_acc_bestJ": extra["acc@bestJ"],
            "val_balacc_bestJ": extra["bal_acc@bestJ"],
            "val_thr_bestJ": extra["thr@bestJ"],
            "lr": scheduler.get_last_lr()[0],
        }
        pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=False, index=False)

        # (Optional) select best checkpoint by balanced accuracy at bestJ
        score_for_selection = extra["bal_acc@bestJ"]
        if cfg.save_best and score_for_selection > best_val_score:
            best_val_score = score_for_selection
            torch.save(model.state_dict(), best_path)
            # also persist the threshold alongside the checkpoint
            with open(os.path.join(cfg.out_dir, "best_threshold.json"), "w") as f:
                json.dump({"threshold": extra["thr@bestJ"], "epoch": epoch}, f)
        print(f"\nEpoch {epoch} → train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc_05:.4f} | val_auc={val_auc:.4f}")

        # append to CSV
        pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=False, index=False)

        # save best
        """if cfg.save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"[✓] Saved new best to {best_path}")"""

    # save last
    torch.save(model.state_dict(), last_path)
    print(f"[i] Saved last model to {last_path}")
    print(f"[i] Metrics logged to {metrics_path}")

    # optional plot
    try:
        import matplotlib.pyplot as plt
        m = pd.read_csv(metrics_path)
        fig = plt.figure()
        plt.plot(m["epoch"], m["train_loss"], label="train_loss")
        plt.plot(m["epoch"], m["val_loss"], label="val_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
        fig.savefig(os.path.join(cfg.out_dir, "loss_curve.png"), bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.plot(m["epoch"], m["val_acc"], label="val_acc")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Val Accuracy")
        fig.savefig(os.path.join(cfg.out_dir, "val_accuracy_curve.png"), bbox_inches="tight")
        plt.close(fig)
        print("[i] Saved plots to outputs/")
    except Exception as e:
        print(f"[WARN] Could not plot curves: {e}")


def run_inference(model_path: str, df, device_pref="auto", out_csv="training_set_inference_results.csv"):
    device = get_device(device_pref)
    model = RecurrenceModel()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    ds = InferenceDataset(df.reset_index(drop=True))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                        pin_memory=True if torch.cuda.is_available() else False)

    logits_all, labels_all = [], []
    thr = 0.5
    thr_path = os.path.join(os.path.dirname(model_path), "best_threshold.json")
    if os.path.exists(thr_path):
        try:
            thr = float(json.load(open(thr_path))["threshold"])
            print(f"[i] Using saved decision threshold: {thr:.3f}")
        except Exception:
            pass
        
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            input_ids, attn_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attn_mask)
            logits = get_logits_from_outputs(outputs)
            logits_all.extend(logits.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    out_df = df.copy()
    out_df["recurrence_logit"] = logits_all
    out_df["true_label"] = labels_all
    out_df.to_csv(out_csv, index=False)
    probs = 1 / (1 + np.exp(-np.array(logits_all)))
    preds = (probs >= thr).astype(int)
    acc   = accuracy_score(labels_all, preds)
    try:
        auc = roc_auc_score(labels_all, probs)
    except Exception:
        auc = float("nan")

    print(f"Inference complete — saved to {out_csv}")
    print(f"Val Accuracy: {acc:.4f} | AUROC: {auc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cuda | cuda:0 | mps | cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA/MPS)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--checkpoint", action="store_true", help="Load base checkpoint")
    parser.add_argument("--no-checkpoint", dest="checkpoint", action="store_false")
    parser.set_defaults(checkpoint=True)
    parser.add_argument("--do_infer", action="store_true")
    args = parser.parse_args()

    seed_everything(42)

    # Data
    df = create_df()
    print("Total usable rows:", len(df))
    train_df, val_df = train_val_split_df(df, val_size=args.val_size)

    # Sanity: tokenizer ok?
    check_tokenization(train_df)

    # Model
    model = build_model(load_checkpoint=args.checkpoint)

    cfg = TrainConfig(
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup,
        grad_accum_steps=args.accum,
        amp=args.amp,
        out_dir=args.out_dir,
        device_pref=args.device
    )

    # Train
    train(model, train_df, val_df, cfg)

    # Optional inference on full training set (or swap with val_df)
    if args.do_infer:
        best_path = os.path.join(cfg.out_dir, "recurrence_model_best.pt")
        last_path = os.path.join(cfg.out_dir, "recurrence_model_last.pt")
        model_path = best_path if os.path.exists(best_path) else last_path
        run_inference(model_path, train_df, device_pref=cfg.device_pref)


if __name__ == "__main__":
    main()
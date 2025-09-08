# mil_recurrence_timeaware.py
import random, datetime as dt

from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hierachical_MIL_model import PatientBagDataset, RecurrenceHierModel, load_backbone_and_heads_from_cls_ckpt, warm_start_recurrence_from_aux

CHECKPOINT_PATH = "Z:\CarlottaHoelzle\Task1\CH_datalabelling\models\dfci-student-medonc.pt"
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

def patient_collate(batch: List[Dict[str,Any]]) -> Dict[str,Any]:
    # We keep batch_size=1 for simplicity; extend as needed.
    assert len(batch) == 1, "Set DataLoader(batch_size=1) or write a ragged collator."
    return batch[0]


# -----------------------------
# 5) Training & inference
# -----------------------------

def train_one_epoch(model, loader, optimizer, device, args, pos_weight: float = 1.0):
    model.train()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    total_loss, n = 0.0, 0
    for step, batch in enumerate(loader, 1):
        # move to device
        
        N = batch["input_ids"].shape[0] 
        L = batch["input_ids"].shape[1] 
        pid = batch["pid"]

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        time_feat = batch["time_feat"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logit, attn, fused = model(input_ids, attn_mask, time_feat)  
        """
        logit, attn_visits, visit_mat = model(
            batch["input_ids"], batch["attention_mask"],
            row_idx=batch.get("row_idx"),
            modality_ids=batch.get("modality_ids"),
            visit_time=batch.get("visit_time"),
        )
        """

        loss = bce(logit.view(1), y.view(1))
        # reproducing rabjubg-but-not-calibrating: if probs hover around 0.4-0.6 and AUROC is high 
        with torch.no_grad():
            p = torch.sigmoid(logit).item()
        print(f"[train] step {step}/{len(loader)} pid={pid} y={int(y.item())} "
            f"prob={p:.3f} logit={logit.item():.3f} N={N} L={L} loss={loss.item():.4f}")
        # optional distillation 
        if getattr(args, "overfit_n", 0) > 0:
            pass  # no distillation during overfit sanity
        else:
            loss = loss + distill_losses(model, fused, attn, aux_mix=("progression_head",), tau=2.0, lambda_patient=0.1, lambda_attn=0.05)
        loss.backward()
        optimizer.step()
        total_loss += loss.item(); n += 1
        if step % 1 == 0:
                print(f"[train] step {step}/{len(loader)} pid={pid} chunks={N} len={L} loss={loss.item():.4f}")
    return total_loss / max(n,1)

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    from sklearn.metrics import roc_auc_score, average_precision_score
    ys, ps = [], []
    top_chunk_dates = {}  # pid -> (date, score)
    it = iter(loader)
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        time_feat = batch["time_feat"].to(device)
        y = float(batch["label"].item())
        pid = batch["pid"]

        logit, attn, _ = model(input_ids, attn_mask, time_feat)
        prob = torch.sigmoid(logit).item()
        print(f"[DBG] pid={pid} y={int(y)} prob={prob:.3f} logit={logit.item():.3f} "
                f"attn_max={float(attn.max().item()):.3f} chunks={len(batch['dates'])}")
        ys.append(y); ps.append(prob)

        # top contributing chunk and its date
        top_i = int(torch.argmax(attn).item())
        date_list = batch["dates"]

        top_date = date_list[top_i] if len(date_list) else None
        top_chunk_dates[pid] = (top_date, float(attn[top_i].item()))

    # Metrics (guard if all-one class in a small fold)
    try:
        auroc = roc_auc_score(ys, ps)
    except Exception:
        auroc = float('nan')
    try:
        auprc = average_precision_score(ys, ps)
    except Exception:
        auprc = float('nan')

    return {"AUROC": auroc, "AUPRC": auprc, "top_chunk_dates": top_chunk_dates}

def distill_losses(model, fused_chunks, attn, aux_mix=("progression_head",), tau=2.0, lambda_patient=0.1, lambda_attn=0.05):
    """
    fused_chunks: (N, 768) outputs before MIL (after time fusion)
    attn: (N,) learned attention from MIL
    Returns extra loss (scalar) using available aux heads on the model.
    """
    loss = torch.tensor(0.0, device=fused_chunks.device)
    if not hasattr(model, "aux_heads"):
        return loss

    # Patient-level: pass MIL patient vec into aux heads
    patient_vec = (attn.unsqueeze(1) * fused_chunks).sum(0)  # (768,)
    for name in aux_mix:
        if name in model.aux_heads:
            aux = model.aux_heads[name](patient_vec).squeeze(-1)   # scalar logit
            # Encourage small magnitude (or match a frozen teacher if you have one).
            # Here: L2 toward zero just to regularize; replace with teacher logits if available.
            loss = loss + lambda_patient * (aux.pow(2).mean())

    # Chunk-attention alignment (soft hint)
    with torch.no_grad():
        # teacher chunk logits from aux head(s), averaged
        t_logit = None
        count = 0
        for name in aux_mix:
            if name in model.aux_heads:
                z = model.aux_heads[name](fused_chunks).squeeze(-1)  # (N,)
                t_logit = z if t_logit is None else (t_logit + z)
                count += 1
        if count > 0:
            t_logit = t_logit / count
            t_soft = torch.softmax(t_logit / tau, dim=0)  # (N,)
        else:
            t_soft = None

    if t_soft is not None and t_soft.numel() == attn.numel():
        # KL(attn || t_soft). Note: attn is already a prob. Add eps for safety.
        eps = 1e-8
        loss = loss + lambda_attn * torch.sum(attn * (torch.log(attn + eps) - torch.log(t_soft + eps)))

    return loss

def run_training(
    train_pids: List[str],
    val_pids: List[str],
    labels_dict: Dict[str, Tuple[int, Optional[dt.date]]],
    ds_root: str,
    args,
    device="cuda:0",
    epochs=3,
    max_chunks_per_patient: Optional[int] = 512,   # tighter
    max_len: int = 512,                           # shorter than 4096
    encoder_microbatch: int = 2,
    amp_dtype: str = "none", #"fp16",
    backbone_device: str = "cuda:0", 
    freez_bb = True, 
    stride = 128,
):
    # Datasets
    train_ds = PatientBagDataset(
        train_pids, ds_root, labels_dict, stride=stride,
        max_chunks_per_patient=max_chunks_per_patient, 
        max_len=max_len,
    )
    val_ds = PatientBagDataset(
        val_pids, ds_root, labels_dict,stride=stride,
        max_chunks_per_patient=max_chunks_per_patient, 
        max_len=max_len,
    )

    fixed_order = True if getattr(args, "overfit_n", 0) > 0 else False
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=not fixed_order, collate_fn=patient_collate)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,          collate_fn=patient_collate)
    # Model
    model = RecurrenceHierModel(
        freeze_backbone=freez_bb, time_dim=128,
        encoder_microbatch=encoder_microbatch,
        amp_dtype=amp_dtype, dropout = 0,
    )
    # pick devices
    main_device = torch.device(device)               # e.g., "mps" or "cuda" or "cpu"
    bb_device   = torch.device(backbone_device)      # e.g., "cpu" (safe on Mac)
    print(f"[INFO] Using {main_device} as main device, using {bb_device} as backbone device.")
    model.place_modules(main_device, bb_device)

    load_backbone_and_heads_from_cls_ckpt(model, CHECKPOINT_PATH, load_heads=("progression_head"))
    # if recurrence head should be randomly initialized, skip warm start
    warm_start_recurrence_from_aux(model, src_head="progression_head")
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    wd = 0.0 if getattr(args, "overfit_n", 0) > 0 else 0.01
    print("weight decay is: ", wd)
    """optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd
    )"""
    bb_params, head_params = [], []
    for n, p in model.named_parameters():
        if n.startswith("backbone.") and p.requires_grad:
            bb_params.append(p)
        elif p.requires_grad:
            head_params.append(p)

    if freez_bb: 
        bb_params = []

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr, "weight_decay": 0.0},
            {"params": bb_params,  "lr": 1e-5,     "weight_decay": 0.0},
        ]
    )


    # Class imbalance (balanced subset => pos_weight=1.0)
    if getattr(args, "overfit_n", 0) > 0:
        pos_weight = 1.0
    else:
        n_pos = sum(labels_dict[pid][0] for pid in train_pids)
        n_neg = len(train_pids) - n_pos
        pos_weight = (n_neg / max(n_pos, 1)) if n_pos and n_neg else 1.0
    # Class imbalance
    n_pos = sum(labels_dict[pid][0] for pid in train_pids)
    n_neg = len(train_pids) - n_pos
    pos_weight = (n_neg / max(n_pos, 1)) if n_pos and n_neg else 1.0

    for ep in range(1, epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, args = args, pos_weight=pos_weight)
        val_metrics = evaluate(model, val_loader, device)
        print(f"[Epoch {ep}] loss={tr_loss:.4f}  AUROC={val_metrics['AUROC']:.4f}  AUPRC={val_metrics['AUPRC']:.4f}")

    return model

# -----------------------------
# 6) Inference: prediction + “fancy” date extraction
# -----------------------------

@torch.no_grad()
def predict_with_explanations(model, pid: str, ds_root: str, labels_dict: Dict[str, Tuple[int, Optional[dt.date]]], device="mps"):
    
    """
    Runs inference on a single patient using the provided model and returns prediction probability along with explanatory information.
    This method processes a patient's data through the model, computes the probability of recurrence, and extracts attention-based explanations such as the most influential chunk date and the top-K chunk dates with their corresponding attention scores.
    Args:
        model: The trained model used for prediction.
        pid (str): Patient identifier.
        ds_root (str): Root directory of the dataset.
        labels_dict (Dict[str, Tuple[int, Optional[dt.date]]]): Dictionary mapping patient IDs to their labels and optional dates.
        device (str, optional): Device to run the model on (e.g., "mps", "cuda", "cpu"). Defaults to "mps".
    Returns:
        dict: A dictionary containing:
            - "pid": The patient ID.
            - "prob_recurrence": The predicted probability of recurrence.
            - "top_chunk_date": The date corresponding to the chunk with the highest attention.
            - "topk_dates": A list of tuples (date, attention_score) for the top-K most attended chunks.
    """
    
    ds = PatientBagDataset([pid], ds_root, labels_dict, max_len=1536, stride=0, max_chunks_per_patient=512)
    batch = ds[0]
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    time_feat = batch["time_feat"].to(device)
    logit, attn, _ = model(input_ids, attn_mask, time_feat)
    prob = torch.sigmoid(logit).item()
    top_i = int(torch.argmax(attn).item())
    top_date = batch["dates"][top_i] if batch["dates"] else None



    # You can also return top-K dates:
    topk = torch.topk(attn, k=min(5, attn.numel()))
    topk_ix = topk.indices.tolist()
    topk_dates = [(batch["dates"][i], float(attn[i].item())) for i in topk_ix]
    return {
        "pid": pid,
        "prob_recurrence": prob,
        "top_chunk_date": top_date,
        "topk_dates": topk_dates
    }

# -----------------------------
# 7) Main: load labels → split → train → eval → save
# -----------------------------
def _load_labels_json(path: str) -> Dict[str, Tuple[int, Optional[dt.date]]]:
    import json
    import pandas as pd
    with open(path, "r") as f:
        raw = json.load(f)  # { empi: [label, "YYYY-MM-DD" or null] }
    labels: Dict[str, Tuple[int, Optional[dt.date]]] = {}
    for empi, pair in raw.items():
        label = int(pair[0])
        d = pair[1]
        rec_date = None
        if d:
            try:
                rec_date = pd.to_datetime(d).date()
            except Exception:
                rec_date = None
        labels[str(empi)] = (label, rec_date)
    return labels

def _stratified_split(pids: List[str], labels_dict: Dict[str, Tuple[int, Optional[dt.date]]],
                      val_frac=0.25, seed=42) -> Tuple[List[str], List[str]]:
    from sklearn.model_selection import train_test_split
    y = [labels_dict[pid][0] for pid in pids]
    train_pids, val_pids = train_test_split(
        pids, test_size=val_frac, random_state=seed, stratify=y
    )
    return list(train_pids), list(val_pids)

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        import torch.backends.mps as mps
        if getattr(mps, "is_available", lambda: False)():
            return "mps"
    except Exception:
        pass
    return "cpu"

def inspect_checkpoint_keys(ckpt_path, limit=60):
    import torch
    sd = torch.load(ckpt_path, map_location="cpu")
    ks = list(sd.keys())
    print(f"[CKPT] {len(ks)} keys")
    for k in ks[:limit]:
        print("  ", k)
    non_backbone = sorted({k.split('.')[0] for k in ks if not k.startswith("longformer.")})
    print("[CKPT] non-backbone top-level prefixes:", non_backbone)

# ---- OVERFIT HELPERS -------------------------------------------------
def _pick_balanced_subset(labels_dict: Dict[str, Tuple[int, Optional[dt.date]]],
                          n_per_class: int,
                          seed: int = 123) -> List[str]:
    """Return a fixed, deterministic, balanced list of pids: n_per_class positives + n_per_class negatives."""
    rng = random.Random(seed)
    pos = [pid for pid, (y, _) in labels_dict.items() if int(y) == 1]
    neg = [pid for pid, (y, _) in labels_dict.items() if int(y) == 0]

    if len(pos) < n_per_class or len(neg) < n_per_class:
        raise ValueError(f"Not enough examples: have pos={len(pos)}, neg={len(neg)}, want {n_per_class} each.")

    pos_sel = sorted(rng.sample(pos, n_per_class))
    neg_sel = sorted(rng.sample(neg, n_per_class))
    subset = pos_sel + neg_sel
    # fixed order: positives then negatives (or shuffle deterministically if you prefer)
    return subset


def _make_deterministic(seed: int = 123):
    import numpy as np, os, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN determinism (may reduce speed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def main():
    import argparse, os, json
    parser = argparse.ArgumentParser(description="Train/eval time-aware MIL recurrence model")
    parser.add_argument("--ds_root", default="ehr_prep/ehr_store/timeline_ds_STS", help="Path to L2 parquet dataset root")
    parser.add_argument("--labels_json",default= "data/labels/labels_hier.json", help="Path to labels_hier.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_frac", type=float, default=0.25)
    parser.add_argument("--max_chunks_per_patient", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_ckpt", default="recurrence_hier_mil.pt")
    parser.add_argument("--predict_k", type=int, default=5, help="Show top-K attention dates for K random val patients")
    parser.add_argument("--overfit_n", type=int, default=0,
                        help="If > 0, use a fixed balanced subset with N pos and N neg patients and try to overfit it.")
    parser.add_argument("--overfit_eval_on_train", action="store_true",
                        help="If set, evaluate on the same tiny subset (classic overfit sanity).")
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    # 0) Find which checkpoint keys are available
    # inspect_checkpoint_keys(CHECKPOINT_PATH, limit=20)

    # 1) Load labels (EMPI -> (label, recurrence_date))
    labels_dict = _load_labels_json(args.labels_json)

    # 2) Pick patient IDs present in labels
    all_pids = list(labels_dict.keys())
    if len(all_pids) == 0:
        raise RuntimeError("No patients found in labels JSON.")

    # 3) Stratified split
    _make_deterministic(args.seed)

    if args.overfit_n > 0:
        subset = _pick_balanced_subset(labels_dict, n_per_class=args.overfit_n, seed=args.seed)
        if args.overfit_eval_on_train:
            train_pids = list(subset)
            val_pids   = list(subset)   # evaluate on the same samples to confirm overfit
            args.lr = 3e-3
        else:
            # split the balanced subset 50/50 into train/val (still balanced)
            half = len(subset) // 2
            train_pids = subset[:half] * 32 # or increase epochs 
            val_pids   = subset[half:]
        print(f"[OVERFIT] Using balanced subset: pos={args.overfit_n}, neg={args.overfit_n}, "
              f"train={len(train_pids)}, val={len(val_pids)}")
    else:
        # 3) Stratified split (original behavior)
        train_pids, val_pids = _stratified_split(all_pids, labels_dict, val_frac=args.val_frac, seed=args.seed)
    print(f"[INFO] Train={len(train_pids)}  Val={len(val_pids)}  (total={len(all_pids)})")

    # 4) Train
    device = _detect_device()
    print(f"[INFO] Using device: {device}")
    model = run_training(
        train_pids=train_pids,
        val_pids=val_pids,
        labels_dict=labels_dict,
        ds_root=args.ds_root,
        device=device,
        epochs=args.epochs,
        max_chunks_per_patient=(args.max_chunks_per_patient if args.max_chunks_per_patient > 0 else None),
        stride=0, 
        freez_bb=True,
        args = args
    )
    torch.save(model.state_dict(), args.out_ckpt)
    print(f"[OK] Saved model weights → {args.out_ckpt}")
    eval_model = model
    """eval_model = RecurrenceHierModel(
        freeze_backbone=False,       # match what you trained with
        time_dim=128,
        encoder_microbatch=2,        # match training config
        amp_dtype="none",            # eval in full precision
        dropout=0.0                  # match what you actually used
    )
    # Place modules on devices exactly like in training
    eval_model.place_modules(torch.device(device), torch.device("cuda"))  # or your backbone_device

    # Load weights-only
    state = torch.load(args.out_ckpt, map_location=device)
    eval_model.load_state_dict(state, strict=True)
    """
    # Lock it down for eval-only
    for p in eval_model.parameters():
        p.requires_grad = False
    eval_model.eval()


    # 5) Final evaluation on val (with top-chunk dates)
    val_ds = PatientBagDataset(val_pids, args.ds_root, labels_dict, max_len=1053, stride=0, max_chunks_per_patient=1024)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=patient_collate)
    final_metrics = evaluate(eval_model, val_loader, device)
    print(f"[FINAL] AUROC={final_metrics['AUROC']:.4f}  AUPRC={final_metrics['AUPRC']:.4f}")

    # 6) Show a few example predictions with “fancy” date extraction
    sample_pids = random.sample(val_pids, min(args.predict_k, len(val_pids)))
    for pid in sample_pids:
        info = predict_with_explanations(model, pid, args.ds_root, labels_dict, device=device)
        print(f"[PRED] pid={pid} prob={info['prob_recurrence']:.3f}  top_date={info['top_chunk_date']}")
        if info["topk_dates"]:
            print("       topK_dates:", [(str(d), f"{w:.3f}") for d, w in info["topk_dates"]])

    # 7) Save checkpoint
    torch.save(model.state_dict(), args.out_ckpt)
    print(f"[OK] Saved model checkpoint → {args.out_ckpt}")

if __name__ == "__main__":
    main()
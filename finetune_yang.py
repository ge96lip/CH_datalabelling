import os

import pandas as pd
import numpy as np
import copy
import torch
import tqdm
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TF"] = "1"


import csv
from model import RecurrenceModel, LabeledDataset, InferenceDataset
from torch import sigmoid
from torch.optim import AdamW
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from scipy.special import expit

# ---------- CONFIG ----------
PATIENT_DIR = "./data/patient_txts"
REC_FILE    = "./data/labels/recurrence.txt"
NONREC_FILE = "./data/labels/non_recurrence.txt"
MAYBE_FILE  = "./data/labels/maybe_recurrence.txt"   # will be ignored/excluded if present


def _read_mrn_list(path):
    """Read a plain-text file with one MRN per line; ignore empties/comments."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        mrns = [ln.strip() for ln in f.readlines()]
    # strip blanks, comments, obvious junk
    mrns = [m for m in (x.strip() for x in mrns) if m and not m.startswith("#")]
    # normalize whitespace-only and trailing .0 (common Excel artifact)
    mrns = [m[:-2] if m.endswith(".0") and m.replace(".","",1).isdigit() else m for m in mrns]
    return list(dict.fromkeys(mrns))  # dedupe, keep order


def load_patient_text(mrn, patient_dir=PATIENT_DIR):
    """Read the entire TXT file for an MRN. Falls back to empty string if missing."""
    path = os.path.join(patient_dir, f"{mrn}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def create_df(
    path_recurrence=REC_FILE,
    path_no_recurrence=NONREC_FILE,
    path_maybe=MAYBE_FILE,
    patient_dir=PATIENT_DIR
):
    rec_ids    = _read_mrn_list(path_recurrence)
    nonrec_ids = _read_mrn_list(path_no_recurrence)
    maybe_ids  = set(_read_mrn_list(path_maybe))
    print(f"Size of recurrence list: {len(rec_ids)}")
    print(f"Size of non-recurrence list: {len(nonrec_ids)}")
    # exclude maybes from both sides
    rec_ids    = [m for m in rec_ids    if m not in maybe_ids]
    nonrec_ids = [m for m in nonrec_ids if m not in maybe_ids]
    print(f"Size of recurrence list after excluding maybes: {len(rec_ids)}")
    print(f"Size of non-recurrence list after excluding maybes: {len(nonrec_ids)}")

    # if any overlap between rec/nonrec, drop from nonrec
    overlap = set(rec_ids).intersection(nonrec_ids)
    if overlap:
        print(f"[WARN] {len(overlap)} MRNs found in BOTH recurrence and non_recurrence. "
              f"Removing them from non_recurrence: {sorted(list(overlap))[:5]}{' ...' if len(overlap)>5 else ''}")
        nonrec_ids = [m for m in nonrec_ids if m not in overlap]

    # Build rows
    rows = []
    missing_files = []
    for mrn, label in [(m, 1) for m in rec_ids] + [(m, 0) for m in nonrec_ids]:
        text = load_patient_text(mrn, patient_dir)
        if text is None or text == "":
            missing_files.append(mrn)
            continue
        rows.append({"mrn": mrn, "text": text, "label": label})

    df = pd.DataFrame(rows)

    # ---- Audit / sanity prints ----
    print(f"Label file counts (after excluding maybes/overlap): "
          f"rec={len(rec_ids)}, nonrec={len(nonrec_ids)}")
    if missing_files:
        print(f"[WARN] Missing/empty patient files for {len(missing_files)} MRNs "
              f"(first 10): {missing_files[:10]}")

    # Unique MRN counts actually used
    used_rec = df[df["label"] == 1]["mrn"].nunique()
    used_non = df[df["label"] == 0]["mrn"].nunique()
    print(f"Used in dataset: rec={used_rec}, nonrec={used_non}, total={len(df)}")

    return df


def train_test_split_df(df):
    """
    Splits the DataFrame into training and validation sets.
    The split is stratified based on the 'label' column.
    """

    train_df, val_df = train_test_split(df, test_size=None, stratify=df["label"], random_state=42)
    print(f"Train size: {len(train_df)} / Recurrence: {train_df.label.sum()} / No Recurrence: {(train_df.label == 0).sum()}")
    print(f"Val size: {len(val_df)} / Recurrence: {val_df.label.sum()} / No Recurrence: {(val_df.label == 0).sum()}")
    return train_df, val_df

def check_tokenization(train_df): 
    sample = train_df.iloc[0]
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer", truncation_side='left')
    encoding = tokenizer(sample['text'], padding='max_length', truncation=True, return_tensors='pt')
    print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0][:20]))  # show first 20 tokens
 
def overfitting_model(model, trainloader):
    
    batch = next(iter(trainloader))
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(20):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device).float()

        outputs = model(input_ids, attention_mask)
        recurrence_logits = outputs[3].squeeze(1)
        print(sigmoid(recurrence_logits))  # Show first 10 logits
        loss = F.binary_cross_entropy_with_logits(recurrence_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
def get_model(): 
    model = RecurrenceModel()
    # Load weights from original model (ignore recurrence_head)
    checkpoint = torch.load("/Users/carlotta/physionet.org/files/dfci-cancer-outcomes-ehr/1.0.0/models/DFCI-student-medonc/dfci-student-medonc.pt", map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and 'recurrence_head' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze recurrence head
    for param in model.recurrence_head.parameters():
        param.requires_grad = True

    # Unfreeze last 2 encoder layers
    for name, param in model.longformer.named_parameters():
        if "encoder.layer.10." in name or "encoder.layer.11." in name:
            param.requires_grad = True
    return model

def train(model, train_df, val_df, device, num_epochs=5, num_neg=0, num_pos=0):
 
    trainloader = DataLoader(
        LabeledDataset(train_df.reset_index(drop=True)),
        batch_size=8, shuffle=True, num_workers=0 # num_workers=0 for compatibility with notebook environments
    )

    validloader = DataLoader(
        LabeledDataset(val_df.reset_index(drop=True)),
        batch_size=4, shuffle=False, num_workers=0 # num_workers=0 for compatibility with notebook environments
    )
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    num_training_steps = num_epochs * len(trainloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.to(device)
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        # loop over all heads 
        """running_train_losses = [0.0 for i in range(4)]
        mean_train_losses = [0.0 for i in range(4)]
        running_valid_losses = [0.0 for i in range(4)]
        mean_valid_losses = [0.0 for i in range(4)]"""
        trainloader = DataLoader(
            LabeledDataset(train_df.reset_index(drop=True)),
            batch_size=8, shuffle=True, num_workers=0
        )
        running_train_loss = 0.0
        running_valid_loss = 0.0
        num_train_batches = len(trainloader)
                
        model.train()
        for i, batch in enumerate(trainloader):
            input_ids = batch[0].to(device)
            input_masks = batch[1].to(device)
            recurrence_true = batch[2].to(device).float()

            #optimizer.zero_grad()
            model.zero_grad()
            outputs_pred = model(input_ids, input_masks)
            recurrence_pred = outputs_pred[3].squeeze(1)
            pos_weight = torch.tensor([num_neg / num_pos]).to(device)
            #loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            #loss = loss_fn(recurrence_pred, recurrence_true)

            loss = F.binary_cross_entropy_with_logits(recurrence_pred, recurrence_true)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            running_train_loss += loss.item()

            print(
                f'Training Epoch: {epoch+1}, Batch {i+1}/{num_train_batches}, '
                f'Batch Loss: {loss.item():.4f}, Mean Loss: {running_train_loss/(i+1):.4f}',
                end='\r',
                #flush=True
            )
        
        print('\n')
        # eval on valid
        
        if validloader is not None:
            model.eval()
            running_valid_loss = 0.0
            num_valid_batches = len(validloader)

            with torch.no_grad():
                for i, batch in enumerate(validloader):
                    input_ids = batch[0].to(device)
                    input_masks = batch[1].to(device)
                    recurrence_true = batch[2].to(device)

                    outputs_pred = model(input_ids, input_masks)
                    recurrence_pred = outputs_pred[3].squeeze(1)

                    loss = F.binary_cross_entropy_with_logits(recurrence_pred, recurrence_true)
                    running_valid_loss += loss.item()

                    print(
                        f'Validation Epoch: {epoch+1}, Batch {i+1}/{num_valid_batches}, '
                        f'Batch Loss: {loss.item():.4f}, Mean Loss: {running_valid_loss/(i+1):.4f}',
                        end='\r',
                        #flush=True
                    )
        
    torch.save(model.state_dict(), "recurrence_model_finetuned.pt")
    
def inference(device):

    model = RecurrenceModel()
    model.load_state_dict(torch.load("recurrence_model_finetuned.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.to(device)
    model.eval()
    inference_dataset = InferenceDataset(train_df)
    inference_loader = DataLoader(inference_dataset, batch_size=8, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in inference_loader:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            # Process outputs as needed
    recurrence_logits = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(inference_loader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask)
            recurrence_pred = outputs[3].squeeze(1).cpu().numpy()
            recurrence_logits.extend(recurrence_pred)
            true_labels.extend(labels.cpu().numpy())

    # 5. Store results in DataFrame
    output_df = train_df.copy()
    output_df["recurrence_logit"] = recurrence_logits
    output_df["true_label"] = true_labels  # Should match original
    output_df.to_csv("training_set_inference_results.csv", index=False)
    print("Inference complete — saved to training_set_inference_results.csv")
    probs = expit(recurrence_logits)

    preds = (probs >= 0.5).astype(int)
    accuracy = accuracy_score(true_labels, preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    
if __name__ == "__main__":
    # Load and prepare data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading and preparing data...")
    df = create_df()
    print("Data loaded. Total records:", len(df))
    train_df, val_df = train_test_split_df(df)
    
    # Check tokenization
    check_tokenization(train_df)
    
    # Initialize model and train
    model = get_model()
    model_temp = copy.deepcopy(model)
    train(model_temp, train_df, val_df, device, num_epochs=5, num_neg=len(df[df['label'] == 0]), num_pos=len(df[df['label'] == 1]))
    
    # Uncomment to overfit on a small batch
    # overfitting_model(model, DataLoader(LabeledDataset(train_df.head(10)), batch_size=2))
    
    # Inference 
    inference(device)
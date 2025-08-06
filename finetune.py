
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

def detect_sep(path):
    with open(path, newline='') as csvfile:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(csvfile.readline())
    return dialect.delimiter

def load_patient_text(pid):
    path = f"data/patients/{pid}.csv"
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep)
    if "text" not in df.columns:
        raise ValueError(f"'text' column not found in file: {path}")
    return " ".join(df["text"].dropna().astype(str).tolist())

def create_df(path_recurrence = "data/label/id_recurrence.csv", path_no_recurrence = "data/label/id_no_recurrence.csv"): 
    
    rec_ids = pd.read_csv(path_recurrence, sep=";").iloc[:, 0].str.strip().tolist()
    no_rec_ids = pd.read_csv(path_no_recurrence, sep=";").iloc[:, 0].str.strip().tolist()


    # Create dataset
    texts, labels = [], []
    # text: combined patient notes
    # label: 0 (no recurrence) or 1 (recurrence)
    for pid in rec_ids:
        texts.append(load_patient_text(pid))
        labels.append(1)
    for pid in no_rec_ids:
        texts.append(load_patient_text(pid))
        labels.append(0)

    df = pd.DataFrame({"text": texts, "label": labels})
    num_neg= len(no_rec_ids)
    num_pos = len(rec_ids)
    print(f"Number of patients with recurrence: {num_pos}")
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
    checkpoint = torch.load("/Users/carlottaholzle/physionet.org/files/dfci-cancer-outcomes-ehr/1.0.0/models/DFCI-student-medonc/dfci-student-medonc.pt", map_location="cpu")
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

    df = create_df()
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
import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from transformers import AutoModel
from transformers import AutoTokenizer
from torch.utils import data


class RecurrenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.longformer = AutoModel.from_pretrained('yikuan8/Clinical-Longformer')
        
        self.any_cancer_head = Sequential(Linear(768, 128), ReLU(), Linear(128,1))
        self.response_head = Sequential(Linear(768, 128), ReLU(), Linear(128,1))
        self.progression_head = Sequential(Linear(768, 128), ReLU(), Linear(128,1))
        
        # NEW: recurrence head (randomly initialized)
        self.recurrence_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))

    def forward(self, input_ids, attention_mask):
        global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
        global_attention_mask[:, 0] = 1

        output = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]

        any_cancer_out = self.any_cancer_head(cls_token)
        response_out = self.response_head(cls_token)
        progression_out = self.progression_head(cls_token)
        recurrence_out = self.recurrence_head(cls_token)  # <--

        return any_cancer_out, response_out, progression_out, recurrence_out
    


class LabeledDataset(data.Dataset):
    def __init__(self, pandas_dataset):
        self.data = pandas_dataset.copy()
        self.indices = self.data.index.unique()
        self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer", truncation_side='left')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        this_index = self.indices[index]
        pand = self.data.loc[this_index, :]

        encoded = self.tokenizer(pand['text'], padding='max_length', truncation=True, return_tensors='pt')

        x_text_tensor = encoded.input_ids.squeeze(0)
        x_attention_mask = encoded.attention_mask.squeeze(0)

        y_label = torch.tensor(pand['label'], dtype=torch.float32)

        return x_text_tensor, x_attention_mask, y_label

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer", truncation_side='left')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, "text"]
        label = self.data.loc[idx, "label"]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.float32)
        )
import pandas as pd
from sklearn.model_selection import train_test_split

recurrence_ids = pd.read_csv('label_recurrence/id_recurrence.csv')['patient_id'].tolist()
no_recurrence_ids = pd.read_csv('label_recurrence/id_no_recurrence.csv')['patient_id'].tolist()

texts, labels = [], []
for pid in recurrence_ids + no_recurrence_ids:
    df = pd.read_csv(f'patients/{pid}.csv')
    notes = " ".join(df[df['type'].isin(['oncology_note', 'radiology_report'])]['text'])
    texts.append(notes)
    labels.append(1 if pid in recurrence_ids else 0)

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.5, random_state=42)
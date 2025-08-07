import pandas as pd
import os

# Load the CSV
df = pd.read_csv("/Users/carlotta/Desktop/Code_MT/CH_datalabelling/Recurrence Dataset_altered.csv")

# Columns to drop (AJ to AP)
columns_to_drop = [
    "Did they develop a second primary lung cancer?",
    "Second Primary Date",
    "Has there been recurrence?",
    "Where is the recurrence site?",
    "Date of Recurrence",
    "Date of CT that Noted Recurrence",
    "Accession # of CT Noting Recurrence", 
    "RADIOLOGY",  
    "PATHOLOGY", 
    "Follow-up Information"
]
df.drop(columns=columns_to_drop, inplace=True)

# Tagging map: {column name: prefix}
tag_map = {}

# ID | for A–C
id_cols = df.columns[0:3]
tag_map.update({col: f"ID | {col}" for col in id_cols})

# DEM | for D–H
dem_cols = df.columns[3:8]
tag_map.update({col: f"DEM | {col}" for col in dem_cols})

# PATH | for J–X
path_cols = df.columns[8:23]
tag_map.update({col: f"PATH | {col}" for col in path_cols})

# RAD | for Z and AA–AH (columns 23–33)
rad_cols = df.columns[23:32]
tag_map.update({col: f"RAD | {col}" for col in rad_cols})

# Follow-Up Information | for AQ–BA (col 42–52)
followup_cols = df.columns[32:43]
tag_map.update({col: f"Follow-Up Information | {col}" for col in followup_cols})

# Rename columns with their tags
df.rename(columns=tag_map, inplace=True)

# Create output directory
output_dir = "./data/patient_txts"
os.makedirs(output_dir, exist_ok=True)

# Write one file per patient
for _, row in df.iterrows():
    # Find the new column name that starts with "ID | Record ID"
    record_id_col = [col for col in df.columns if col.endswith("Record ID") and col.startswith("ID")][0]
    record_id = str(row[record_id_col])
    filename = os.path.join(output_dir, f"{record_id}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        for column, value in row.items():
            value_str = "" if pd.isna(value) else str(value)
            f.write(f"{column} | {value_str}\n")
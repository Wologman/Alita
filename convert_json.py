import json
import csv
from pathlib import Path

# Set paths
json_path = Path(r'E:\Alita\Data\Experiments\Exp_47\Runs\Run_02\Results\Run_02_class_names.json')
csv_path = Path(r'E:\Alita\Resources\class_names_from_classifier.csv')

# Read the JSON file
with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Write to CSV
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for item in data:
        writer.writerow([item])
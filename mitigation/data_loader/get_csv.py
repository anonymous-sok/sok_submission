import os
import json
import pandas as pd


start_dir = "../../src/SlowTrack/results"
output_csv = os.path.join(start_dir, "summary.csv")


json_files = []
for root, dirs, files in os.walk(start_dir):
    for file in files:
        if file.endswith(".json"):
            json_files.append(os.path.join(root, file))

json_files.sort() 

records = []
for path in json_files:
    try:
        with open(path, "r") as f:
            result = json.load(f)

        tracking_metrics = result["tracking_metrics"]

        record = {
            "file": os.path.basename(path),
            "total_track_time": tracking_metrics["total_track_time"],
            "total_time": tracking_metrics["total_time"],
            "mota": result["mota"],
            "mAP": result["mAP"],
            "mAP50": result["AP@0.50"]
        }
        records.append(record)

    except (KeyError, json.JSONDecodeError) as e:
        print(f"[Warning] Skipped: {path} → {e}")


df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)

print(f"[✓] Summary written to: {output_csv}")
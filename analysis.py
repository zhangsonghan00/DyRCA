
import pandas as pd
import os
import json

case_dir = '/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered/ts1-ts-train-service-pod-kill-stncr4'
env_path = os.path.join(case_dir, "env.json")
inj_path = os.path.join(case_dir, "injection.json")
metrics_path = os.path.join(case_dir, "abnormal_metrics.parquet")

with open(env_path) as f:
    env = json.load(f)
with open(inj_path) as f:
    inj = json.load(f)

metrics = pd.read_parquet(metrics_path)

print(metrics)

ab_start = pd.to_datetime(
                int(env["ABNORMAL_START"]), unit="s", utc=True
            ) - pd.Timedelta(hours=8)
ab_end = pd.to_datetime(
                int(env["ABNORMAL_END"]), unit="s", utc=True
            )
print(f"abnormal start: {ab_start}")
print(f"abnormal end: {ab_end}")

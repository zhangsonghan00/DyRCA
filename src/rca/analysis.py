import pandas as pd
import torch
import os
from pathlib import Path

data_root = r"/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered"

# 假设可以读取path目录
service_set = set()
service_select = [
    "preserve-service",
    "security-service",
    "order-service",
    "consign-service",
    "consign-price-service",
    "food-service",
    "station-food-service",
    "train-food-service",
    "assurance-service",
    "preserve-other-service",
    "order-other-service",
    "contact-service",
    "station-service",
    "seat-service",
]

service_path_name = []

# 遍历目录
for root, dirs, files in os.walk(data_root):
    for dir_name in dirs:
        if any(sample in dir_name for sample in service_select):
            service_path_name.append(dir_name)
            service_name = dir_name.split("-")[1:5]
            service_name = "-".join(service_name)
            service_set.add(service_name)

data_packs = [Path(data_root) / name for name in service_path_name]

print("Unique services:", service_set)
print("total services:", len(service_set))
print("Total:", len(service_path_name))

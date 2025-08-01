
import pandas as pd
import os
import json

path = r"/home/nn/rca/GraphRCA/data/RCABENCH/service_to_idx.pkl"
service_to_idx = pd.read_pickle(path)
print(service_to_idx)

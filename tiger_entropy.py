import pickle
import numpy as np
from collections import Counter
import sys
import os

dataset = sys.argv[1]

os.chdir(f'{os.environ["HOME"]}/GRID/SASRec.pytorch/python')

sys.path.insert(0, '.')
from utils import data_partition
data = data_partition(f'reviews_{dataset}')
itemnum = data[4]
print(f"itemnum: {itemnum}")

pkl_path = f'{os.environ["HOME"]}/GRID/tiger_inference/{dataset}/outputs/pickle/merged_predictions.pkl'
with open(pkl_path, 'rb') as f:
    predictions = pickle.load(f)

counter = Counter()
for entry in predictions:
    for sid in entry['semantic_ids']:
        counter[tuple(sid)] += 1

counts = np.array(list(counter.values()), dtype=float)
probs = counts / counts.sum()
entropy = -np.sum(probs * np.log2(probs + 1e-10))
max_entropy = np.log2(itemnum)

print(f"Dataset: {dataset}")
print(f"Recommendation entropy: {entropy:.4f}")
print(f"Max possible entropy:   {max_entropy:.4f}")
print(f"Normalized entropy:     {entropy/max_entropy:.4f}")
print(f"Unique items recommended: {len(counter)} / {itemnum}")
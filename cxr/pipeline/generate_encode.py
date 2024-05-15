import os
import pickle
import pandas as pd
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num_batch', type=int)
args = parser.parse_args()

num_batch = args.num_batch

NUM = 10550


sequences = []
for idx in tqdm(range(1, num_batch+1)):
    filename = f"./output/generations/generations_{NUM}_{idx}.pkl"
    with open(filename, 'rb') as f:
        seq = pickle.load(f)
        sequences.extend(seq)


cache_dir = "./output/trained_cxr_10"
os.makedirs(cache_dir, exist_ok=True)
pd.to_pickle(sequences, os.path.join(cache_dir, '0.pkl'))

print(len(sequences))









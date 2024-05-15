
import argparse
import glob
import json
import os

import pandas as pd
import torch
import tqdm
import transformers

import pickle
import pathlib

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
from datasets import load_dataset, Dataset
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument("--batch_size", default=None, type=int, required=True,
                    help="Number of questions in each batch")
parser.add_argument("--batch_num", default=None, type=int, required=True,
                    help="Number of batches")


args = parser.parse_args()

model_name = args.model
sequences = []

run_id = 0

cache_dir_par = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.seed}')

for idx in tqdm.tqdm(range(1, args.batch_num+1)):
    cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.seed}/batch_results/{args.batch_size}_{idx}')
    # old_results = glob.glob(os.path.join(cache_dir_par, '*.pkl'))
    # old_results = [_ for _ in old_results if '_partial' not in _ or str(idx) not in _]
    # run_id = len(old_results)
    file_dir = os.path.join(cache_dir, f'{run_id}_{args.batch_size}_{idx}.pkl')
    if os.path.isfile(file_dir):
        with open(file_dir, 'rb') as infile:
            seq = pickle.load(infile)
            print(seq)
            sequences.extend(seq)

print(f'number of items: {len(sequences)}')
pd.to_pickle(sequences, os.path.join(cache_dir_par, f'{run_id}.pkl'))


    
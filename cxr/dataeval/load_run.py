import torch
import functools
import os
from typing import Dict
import argparse

import tqdm

import pickle
import pathlib

from load import *

print(os.getcwd())

import sys
sys.path.append('.')

import persist_to_disk as ptd
ptd.config.generate_config()

import dataeval.load_worker as lw
import models

import models.nli as sc
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='opt-13b')
parser.add_argument("--idx", default=None, type=int, required=True,
                        help="Index of sequences of questions")
parser.add_argument("--batch_size", default=None, type=int, required=True,
                    help="Number of questions in each batch")
parser.add_argument('--data', type=str, default='coqa')
args = parser.parse_args()


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IGNORE_INDEX = -100


if __name__ == '__main__':
    import _settings 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = args.model
    data = args.data
    path =  _settings.GEN_PATHS[data][model]


    idx = args.idx
    batch_size = args.batch_size

    print(path)
    # organize the results.

    print('reading cleaned outputs...\n')
    cleaned_seq = read_cleaned_outputs_new(path)

    ids = [_['id'] for _ in cleaned_seq]
    print(len(ids))

    # compare each pair of generated responses - this could be sped up by using batches
    print('reading semantic similarities...\n')
    sem_sim = read_semantic_similarities_new(path, idx, batch_size, device=device)
    dict_add = read_semantic_similarities_new(path, idx+1, batch_size, device=device)
    for key,value in dict_add.items():
        sem_sim[key] = value

    # the lexical similarity baseline
    print('reading lexical similarities...\n')
    lex_sim = read_lexical_sim(path, idx, batch_size, parallel=True)
    # print(lex_sim)

    # for white-box method
    print('reading likelihood...\n')
    if model != 'gpt-3.5-turbo':
        read_loglikelihoods_and_more_new(path, idx, batch_size, device=device)

    # For evaluation of the generated responses
    print('reading rouge scores...\n')
    read_rouges_new(path, idx, batch_size, parallel=True) # compute the rougeL scores

import torch
import os
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode

import csv
import functools
import os
from collections import defaultdict
from importlib import reload

import evaluate
import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import torch
import tqdm
from pandarallel import pandarallel

import sys
sys.path.append('.')

import models
import models.nli as sc
import utils

rouge = evaluate.load('rouge', keep_in_memory=True)
exact_match_metric = evaluate.load("exact_match")

def _get_semantic_similarities_sample(sample, judge_model:sc.ClassifyWrapper, clean=False):
    text_key = 'text_cleaned' if clean else 'text'
    # _log_fn = lambda str: None if logger is None else logger.info(str)

    question = sample['question']
    has_semantically_different_answers = False
    all_ans = sample['generations'][text_key]
    unique_ans = sorted(list(set(all_ans)))
    semantic_set_ids = {ans: i for i, ans in enumerate(unique_ans)}
    _rev_mapping = semantic_set_ids.copy()
    sim_mat = torch.zeros((len(unique_ans), len(unique_ans),3))
    old_deberta_predictions = []

    # _log_fn("Number of unique answers: " + str(len(unique_ans)))

    for i, ans_i in enumerate(unique_ans):
        for j, ans_j in enumerate(unique_ans[i+1:], i+1):
            sim_mat[i,j] = judge_model.pred_qa(question, ans_i, ans_j)[0]
            sim_mat[j,i] = judge_model.pred_qa(question, ans_j, ans_i)[0]

            # original logic
            deberta_prediction = torch.stack([sim_mat[i,j], sim_mat[j,i]], 0).argmax(1)
            # _log_fn(f'Q: {question} || A1: {ans_i} || A2: {ans_j} || {deberta_prediction}')
            if deberta_prediction.min() == 0:
                has_semantically_different_answers = True
            else:
                semantic_set_ids[ans_j] = semantic_set_ids[ans_i]
            old_deberta_predictions.append([question, ans_i, ans_j, deberta_prediction.min().item()])
    return dict(
        id=sample['id'],
        mapping = [_rev_mapping[_] for _ in all_ans],
        sim_mat = sim_mat,
        old = {
        'has_semantically_different_answers': has_semantically_different_answers,
        'syntactic_similarities': _old_syntactic_similarities(sample['generations'][text_key])},
    ), old_deberta_predictions

@torch.no_grad()
def _get_semantic_similarities(samples, judge_model:sc.ClassifyWrapper, clean=False):
    utils.seed_everything(10)
    result_dict, deberta_predictions = {}, []
    for sample in tqdm.tqdm(samples):
        result_dict[sample['id']], deberta_predictions_ = _get_semantic_similarities_sample(sample, judge_model, clean, logger)
        deberta_predictions.extend(deberta_predictions_)
    return result_dict, pd.DataFrame(deberta_predictions, columns=['question', 'ans1', 'ans2', 'deberta_prediction'])


def get_semantic_similarities(path:str, idx, batch_size, clean=True):

    cleaned_sequences = 

    judge_model == 'microsoft/deberta-large-mnli'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc_model = sc.ClassifyWrapper(judge_model, device=device)
    semantic_sims, deberta_predictions = lw._get_semantic_similarities(cleaned_sequences, sc_model, clean)

    par_path = path.split('/').pop()
    pathlib.Path(f'{par_path}').mkdir(parents=True, exist_ok=True)
    with open(f'{par_path}/semantic_sims_{idx}_{batch_size}.pkl', 'wb') as outfile:
        pickle.dump(semantic_sims, outfile)

    return semantic_sims



def _get_lexical_similarities_sample(sample):
    all_ans = sample['pred']
    unique_ans = sorted(list(set(all_ans)))
    ans2i = {ans: i for i, ans in enumerate(unique_ans)}
    sim_mat = np.eye(len(unique_ans))
    for i, ans_i in enumerate(unique_ans):
        for j, ans_j in enumerate(unique_ans[i+1:], i+1):
            sim_mat[i,j] = sim_mat[j,i] = rouge.compute(predictions=[ans_i], references=[ans_j], rouge_types=['rougeL'])['rougeL']
    return {'sim_mat': sim_mat, 'mapping': [ans2i[_] for _ in all_ans]}


def _get_lexical_similarities(samples, clean=False):
    text_key = 'text_cleaned' if clean else 'text'
    df = pd.DataFrame({key: [sample[key] for sample in samples] for key in ['id', 'answer', 'question']})
    df['text_key'] = text_key
    df['pred'] = [sample['generations'][text_key] for sample in samples]
    ret = df.apply(_get_lexical_similarities_sample, axis=1)
    return ret.values.tolist()


def lexical_sim(path:str, idx, batch_size, clean=True):

    cleaned_sequences = 

    lexical_similarities = lw._get_lexical_similarities(cleaned_sequences[((idx-1) * batch_size) : (idx * batch_size)], clean)
    lexical_similarities = {_['id']: _eval for _, _eval in zip(cleaned_sequences[((idx-1) * batch_size) : (idx * batch_size)], lexical_similarities)}

    par_path = path.split('/').pop()
    pathlib.Path(f'{par_path}').mkdir(parents=True, exist_ok=True)
    with open(f'{par_path}/lexical_sims_{idx}_{batch_size}.pkl', 'wb') as outfile:
        pickle.dump(lexical_similarities, outfile)
    return lexical_similarities







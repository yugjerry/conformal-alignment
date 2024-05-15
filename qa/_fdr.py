import torch

import sys
from importlib import reload
import persist_to_disk as ptd
import os
ptd.config.set_project_path(os.path.abspath("."))
import tqdm
import pandas as pd
import numpy as np
import re
import utils
import seaborn as sns
import math
import random

from xgboost import XGBRFClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.linear_model import LogisticRegression

sns.set_theme()
sns.set_context("notebook")

import pipeline.clustering as pc
import pipeline.eval_uq as eval_uq

import argparse

import pickle
import pathlib

from _settings import GEN_PATHS

import matplotlib.pyplot as plt


import pipeline.uq_bb as uq_bb
reload(uq_bb)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='coqa')
parser.add_argument('--model', type=str, default='opt-13b')
parser.add_argument("--batch_num", default=400, type=int, required=True,
                        help="Number of batches")
parser.add_argument("--batch_size", default=20, type=int, required=True,
                    help="Number of questions in each batch")
parser.add_argument('--split_pr', type=float, default=0.5)
parser.add_argument('--split_pr_tune', type=float, default=0.2)
parser.add_argument('--N', type=int, default=1000, help="Number of reference data")
parser.add_argument('--repN', type=int, default=500)
parser.add_argument('--clf_name', type=str, default='logistic')
args = parser.parse_args()

# python3 -u _fdr.py --data "triviaqa" --model "llama-2-13b-chat-hf" --N 2000 --split_pr 0.5 --split_pr_tune 0.2

model = args.model
data = args.data
batch_num = args.batch_num
batch_size = args.batch_size
split_pr = args.split_pr
split_pr_tune = args.split_pr_tune
N = args.N
clf_name = args.clf_name
repN = args.repN

assert split_pr_tune + split_pr < 1


num_gens = 20
acc_name = 'generations|rougeL|acc'

path = GEN_PATHS[data][model] 

summ_kwargs = {
    'u+ea': {'overall': True, 'use_conf': False},
    'u+ia': {'overall': False, 'use_conf': False},
    'c+ia': {'overall': False, 'use_conf': True},
}['c+ia']

uq_list = [
        'generations|numsets', 
        'lexical_sim',
        'generations|spectral_eigv_clip|disagreement_w',
        'generations|eccentricity|disagreement_w',
        'generations|degree|disagreement_w',
        'generations|spectral_eigv_clip|agreement_w',
        'generations|eccentricity|agreement_w',
        'generations|degree|agreement_w',
        'generations|spectral_eigv_clip|jaccard',
        'generations|eccentricity|jaccard',
        'generations|degree|jaccard',
        'semanticEntropy|unnorm', 
        'self_prob',
]

q_seq = np.round(np.linspace(0.01,0.99,99),3)

num_thresholds = 100

def rt(lab_cal, sc_cal, sc_test, t):
    u = 1
    return ((u+np.sum((lab_cal==0) & (sc_cal>=t)))/(u+len(sc_cal))) * (len(sc_test) / max(1,np.sum((sc_test>=t))))

t_seq = np.linspace(0, 1, num=num_thresholds)


# SAMPLE SIZE

tune_size = math.floor(split_pr_tune * N)  # for choosing hyperparameters
train_size = math.floor(split_pr * N)
cal_size = N - tune_size - train_size
test_size = 500

print([tune_size, train_size, cal_size])

#################################################################
#################################################################

# reference kwargs
o = uq_bb.UQ_summ(path, batch_num=batch_num, batch_size=batch_size, clean=True, split='test', cal_size=tune_size, train_size=train_size, seed=0)

uq_kwargs_ref = summ_kwargs

if len(o.key) > 2:
    assert o.key[2] == 'test'
    self2 = o.__class__(o.path, o.batch_num, o.batch_size, o.key[1], 'val', o.key[3], o.key[3], o.key[5])
    self2.tunable_hyperparams = {k:v for k, v in o.tunable_hyperparams.items() if k in uq_list}
    tuned_hyperparams = self2._tune_params(num_gens=num_gens,
                                           metric=acc_name,
                                           overall=False, use_conf=True, curve='auarc')
    uq_kwargs_ref.update(tuned_hyperparams)
else:
    uq_kwargs_ref.update(o._get_default_params())

print(f'uq_kwargs_ref: {uq_kwargs_ref}')

par_path = '/'.join(path.split('/')[:-1])

filename = os.path.join(par_path, 'uq_result/result')
if not os.path.isdir(filename):
    os.makedirs(filename)



# load confidence/uncertainty scores

uq_res = []
for uq_ in uq_list:
    _, individual_res = o.get_uq(name=uq_, num_gens=num_gens, **uq_kwargs_ref.get(uq_,{}))
    print(individual_res.to_numpy().shape)
    uq_res.append(individual_res.to_numpy())

print(uq_res.index)
all_ids = o.ids

uq_res = np.array(uq_res)
uq_res = np.swapaxes(uq_res,0,1)
print(f'shape of uq_res: {uq_res.shape}')

label = o.get_acc(acc_name)[1]



fdp_rep = []
power_rep = []

for rep_idx in tqdm.tqdm(range(repN)):

    train_set = np.random.choice(uq_res.shape[0], train_size, replace=False)
    test_set = set(np.arange(uq_res.shape[0])) - set(train_set)

    train_ids = [all_ids[_] for _ in train_set]
    test_ids = [all_ids[_] for _ in test_set]


    train_label = label.loc[train_ids,:].to_numpy(dtype=int)
    test_label = label.loc[test_ids,:].to_numpy(dtype=int)
    uq_score = []

    print('fitting classifier...')
    for ids_gen in range(num_gens):
        X_train = uq_res[train_set,:,ids_gen]
        X_cal = uq_res[list(test_set),:,ids_gen]
        y_train = train_label[:,ids_gen]

        print(np.mean(y_train.ravel()))

        if np.mean(y_train.ravel())==0:
            uq_score.append([1 for i in range(X_cal.shape[0])])
        elif np.mean(y_train.ravel())==1:
            uq_score.append([0 for i in range(X_cal.shape[0])])
        else:
            if clf_name == 'rf':
                rf_depth = 30
                clf = RandomForestClassifier(max_depth=rf_depth, random_state=2024).fit(X_train, y_train)
            if clf_name == 'logistic':
                clf = LogisticRegression(random_state=0).fit(X_train, y_train)
            if clf_name == 'xgbrf':
                clf = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2).fit(X_train, y_train)

            
            clf_prob_ = clf.predict_proba(X_cal)
            uq_score.append(clf_prob_[:,0])
        

    uq_score = np.array(uq_score).T

    scores = pd.DataFrame(uq_score, test_ids)

    scores = 1 - np.array(scores)
    labels = test_label

    print(scores.shape)
    print(labels.shape)

    
    scs = scores[:,0]
    labs = labels[:,0]

    cal_idx = np.random.choice(len(labs), cal_size, replace=False)
    test_idx = list(set(range(len(labs))) - set(cal_idx))
    label_cal = labs[cal_idx]
    score_cal = scs[cal_idx]
    label_test = labs[test_idx]
    score_test = scs[test_idx]

    test_idx = np.random.choice(len(label_test), test_size, replace=False)
    label_test = label_test[test_idx]
    score_test = score_test[test_idx]

    fdp = []
    power = []

    for q in q_seq:

        rt_seq = np.array([rt(label_cal, score_cal, score_test, t) for t in t_seq])
        idx_set = np.where(rt_seq <= q)[0]
        if len(idx_set) == 0:
            tau = 1
        else:
            tau = t_seq[idx_set[0]]

        fdp.append(np.sum((label_test==0)&(score_test>=tau))/max(1,np.sum(score_test>=tau)))
        power.append(np.sum((label_test==1)&(score_test>=tau))/max(1,np.sum(label_test==1)))

    fdp_rep.append(fdp)
    power_rep.append(power)

fdp_rep = np.array(fdp_rep)
power_rep = np.array(power_rep)

fdp_seq = np.mean(fdp_rep, axis=0)
power_seq = np.mean(power_rep, axis=0)

fdp_std = np.sqrt(np.var(fdp_rep, axis=0))
power_std = np.sqrt(np.var(power_rep, axis=0))


pd.to_pickle(fdp_seq, os.path.join(par_path, f'uq_result/result/fdr_{model}_{data}_{clf_name}_{N}_{split_pr_tune}_{split_pr}_{repN}.pkl'))
pd.to_pickle(power_seq, os.path.join(par_path, f'uq_result/result/power_{model}_{data}_{clf_name}_{N}_{split_pr_tune}_{split_pr}_{repN}.pkl'))
pd.to_pickle(fdp_std, os.path.join(par_path, f'uq_result/result/fdr_std_{model}_{data}_{clf_name}_{N}_{split_pr_tune}_{split_pr}_{repN}.pkl'))
pd.to_pickle(power_std, os.path.join(par_path, f'uq_result/result/power_std_{model}_{data}_{clf_name}_{N}_{split_pr_tune}_{split_pr}_{repN}.pkl'))

fig, ax = plt.subplots(figsize=(4, 4))

ax.plot(q_seq, fdp_seq, 'bo--', label='FDR')
ax.plot(q_seq, power_seq, 'rs--', label='Power')
ax.fill_between(q_seq, fdp_seq-fdp_std, fdp_seq+fdp_std, alpha=0.5, 
    edgecolor='lightblue', facecolor='lightblue')
ax.fill_between(q_seq, power_seq-power_std, power_seq+power_std, alpha=0.5, 
    edgecolor='lightpink', facecolor='lightpink')
ax.set_xlabel("Target level of FDR")
ax.set_ylabel("FDR and Power")

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'g--', alpha=0.75, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlim((0,1))
ax.set_ylim((0,1))
ax.legend(loc='best')

plt.savefig(f'{par_path}/uq_result/fdp_plot_{model}_{data}_{clf_name}_{N}_{split_pr_tune}_{split_pr}_{repN}.pdf', dpi=400, bbox_inches='tight')



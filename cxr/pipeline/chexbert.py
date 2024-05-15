import torch
import json, os
import argparse
import sys
sys.path.append('.')
sys.path.append('./utils')

from chexbert_utils import CheXbert
import pandas as pd

# parser = argparse.ArgumentParser()
# parser.add_argument('--base_dir', type=str, default='/Users/yugui/Library/CloudStorage/Dropbox/conformal_align/codes/cxr')
# parser.add_argument('--NUM', type=int)
# parser.add_argument('--split_subset', type=str, default="dev")
# parser.add_argument('--cxrmate', type=str, default='')
# parser.add_argument('--num_seq', type=int, default=2)
# args = parser.parse_args()

# base_dir = args.base_dir
# NUM = args.NUM
# cxrmate_ = args.cxrmate
# split_subset = args.split_subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

chexbert_path = './models/chexbert.pth'

bert_path = "bert-base-uncased"

# path_jsonl = os.path.join(base_dir, f'code/output/output_{cxrmate_}{split_subset}/generation_{split_subset}_{NUM}_{args.num_seq}.jsonl')

batch_size = 16

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""


def chexbert_eval(y_hat, y, study_id):

    CONDITIONS = [
        'enlarged_cardiomediastinum',
        'cardiomegaly',
        'lung_opacity',
        'lung_lesion',
        'edema',
        'consolidation',
        'pneumonia',
        'atelectasis',
        'pneumothorax',
        'pleural_effusion',
        'pleural_other',
        'fracture',
        'support_devices',
        'no_finding',
    ]

    # with open(path_jsonl) as f:
    #     data = [json.loads(line) for line in f]

    model = CheXbert(
        bert_path=bert_path,
        chexbert_path=chexbert_path,
        device=device,
    ).to(device)

    table = {'chexbert_y_hat': [], 'chexbert_y': [], 'y_hat': [], 'y': [], 'study_id': []}
    # for batch in minibatch(data, batch_size):
    table['chexbert_y_hat'].extend([i + [j] for i, j in zip(model(list(y_hat)).tolist(), list(study_id))])
    table['chexbert_y'].extend([i + [j] for i, j in zip(model(list(y)).tolist(), list(study_id))])
    table['y_hat'].extend(y_hat)
    table['y'].extend(y)
    table['study_id'].extend(study_id)


    columns = CONDITIONS + ['study_id']
    df_y_hat = pd.DataFrame.from_records(table['chexbert_y_hat'], columns=columns)
    df_y = pd.DataFrame.from_records(table['chexbert_y'], columns=columns)

    # df_y_hat.to_csv(path_jsonl.replace('.jsonl', '_chexbert_y_hat.csv'))
    # df_y.to_csv(path_jsonl.replace('.jsonl', '_chexbert_y.csv'))

    df_y_hat = df_y_hat.drop(['study_id'], axis=1)
    df_y = df_y.drop(['study_id'], axis=1)

    df_y_hat = (df_y_hat == 1)
    df_y = (df_y == 1)

    tp = (df_y_hat == df_y).astype(float)

    fp = (df_y_hat == ~df_y).astype(float)
    fn = (~df_y_hat == df_y).astype(float)

    tp_cls = tp.sum()
    fp_cls = fp.sum()
    fn_cls = fn.sum()

    tp_eg = tp.sum(1)
    fp_eg = fp.sum(1)
    fn_eg = fn.sum(1)

    precision_class = (tp_cls / (tp_cls + fp_cls)).fillna(0)
    recall_class = (tp_cls / (tp_cls + fn_cls)).fillna(0)
    f1_class = (tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls))).fillna(0)

    scores = {
        'ce_precision_macro': precision_class.mean(),
        'ce_recall_macro': recall_class.mean(),
        'ce_f1_macro': f1_class.mean(),
        'ce_precision_micro': tp_cls.sum() / (tp_cls.sum() + fp_cls.sum()),
        'ce_recall_micro': tp_cls.sum() / (tp_cls.sum() + fn_cls.sum()),
        'ce_f1_micro': tp_cls.sum() / (tp_cls.sum() + 0.5 * (fp_cls.sum() + fn_cls.sum())),
        'ce_precision_example': (tp_eg / (tp_eg + fp_eg)).fillna(0).mean(),
        'ce_recall_example': (tp_eg / (tp_eg + fn_eg)).fillna(0).mean(),
        'ce_f1_example': (tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).fillna(0).mean(),
        'ce_num_examples': float(len(df_y_hat)),
    }

    class_scores_dict = {
       **{'ce_precision_' + k: v for k, v in precision_class.to_dict().items()},
       **{'ce_recall_' + k: v for k, v in recall_class.to_dict().items()},
       **{'ce_f1_' + k: v for k, v in f1_class.to_dict().items()},
    }
    # pd.DataFrame(class_scores_dict, index=['i',]).to_csv(path_jsonl.replace('.jsonl', '_chexbert_scores.csv'))

    tp = (df_y_hat == df_y).astype(float)
    tp_eg = tp.sum(1)

    return df_y, df_y_hat, f1_class, scores
    

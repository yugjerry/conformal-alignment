# conformal-alignment

This repo implements the Conformal Alignment procedure in the tasks of question answering and radiology report generation. 
Given any pre-trained model and new units with model-generated outputs, Conformal Alignment leverages a set of reference data with ground-truth alignment status to train an alignment predictor. It then selects new units whose predicted alignment scores surpass a data-dependent threshold, certifying their corresponding outputs as trustable. It is guaranteed that on average, a prescribed fraction of selected units indeed meet the alignment criterion, regardless of the foundation model or the data distribution.

Answer generations in question answering and calculation of confidence/uncertainty scores follow the implementation in <https://github.com/zlin7/UQ-NLG>.

# Question answering

This repo supports the [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) and [CoQA](https://stanfordnlp.github.io/coqa/) datasets.

You need to specify the large language model (LLM) and dataset in use, e.g. ```model='llama-2-13b-chat-hf'```, ```dataset='triviaqa'```. Use the following command to generate answers by batch (```idx``` the index of each batch).

> python3 -m pipeline/generate --model $model --dataset $dataset --batch_size 20 --idx $SGE_TASK_ID

After the generation step,  use the following command to obtain self-evaluation scores and uncertainty/confidence scores.

> python3 -m dataeval/load_run.py --batch_size $bsize --data $data --model $model --idx $SGE_TASK_ID

The script ```_fdr.py``` implements the Conformal Alignment procedure and calculates FDR and power.

> python3 -m _fdr.py --data "triviaqa" --model "llama-2-13b-chat-hf" --N 2000 --split_pr 0.5 --split_pr_tune 0.2


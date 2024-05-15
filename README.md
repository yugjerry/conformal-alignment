# conformal-alignment

This repo implements the Conformal Alignment procedure in the tasks of question answering and radiology report generation. 
Given any pre-trained model and new units with model-generated outputs, Conformal Alignment leverages a set of reference data with ground-truth alignment status to train an alignment predictor. It then selects new units whose predicted alignment scores surpass a data-dependent threshold, certifying their corresponding outputs as trustable. It is guaranteed that on average, a prescribed fraction of selected units indeed meet the alignment criterion, regardless of the foundation model or the data distribution.

Answer generating in question answering and calculation of confidence/uncertainty scores follow the implementation in <https://github.com/zlin7/UQ-NLG>.

# Question answering (```qa/```)

## Dataset and LLM preparation

This repo supports the [**TriviaQA**](https://nlp.cs.washington.edu/triviaqa/) and [**CoQA**](https://stanfordnlp.github.io/coqa/) datasets, which will be prepared in ```qa/pipeline/generate.py```. Two large language models (LLM) used in the paper are [**OPT-13B**](https://huggingface.co/facebook/opt-13b) and [**LLaMA-2-13B-chat**](https://llama.meta.com/llama-downloads/).

## Answer generating

You need to specify the LLM and dataset in use, e.g. ```model='llama-2-13b-chat-hf'```, ```dataset='triviaqa'```. 
Use the following command to generate answers by batch (```idx``` is the index of each batch).

```bash
python3 -m pipeline/generate --model $model --dataset $dataset --batch_size 20 --idx $SGE_TASK_ID
```

## Scores extraction

After the generation step,  use the following command to obtain self-evaluation scores and uncertainty/confidence scores.

```bash
python3 -m dataeval/load_run.py --batch_size $bsize --data $data --model $model --idx $SGE_TASK_ID
```

## Conformal Alignment implementation
The script ```_fdr.py``` implements the Conformal Alignment procedure and calculates FDR and power.

```bash
python3 -m _fdr.py --data "triviaqa" --model "llama-2-13b-chat-hf" --N 2000 --split_pr 0.5 --split_pr_tune 0.2
```

# Chest X-ray (CXR) report generating (```cxr/```)

## Dataset and LLM preparation

Vision-language model fine-tuning is implemented in the notebook ```cxr/vlm_finetune.ipynb``` following [conformal language modeling](https://arxiv.org/abs/2306.10193), in which we use a [**Vision Transformer** (ViT) pretrained on ImageNet-21k](https://huggingface.co/google/vit-base-patch16-224-in21k) as the image encoder and **GPT2** as the text decoder.

[**MIMIC-CXR**](https://www.nature.com/articles/s41597-019-0322-0) dataset needs access. See the [PhysioNet project page](https://physionet.org/content/mimic-cxr/2.0.0/).

## Report generating

After specifying the fine-tuned model (```model='trained'```) and dataset (```data='cxr'```), use the following command to generate and concatenate outputs (```bnum``` is the number of batches and ```bsize``` is the batch size).

```bash
python3 -m pipeline/generate --idx $SGE_TASK_ID --batch_size $bsize
python3 -m pipeline/generate_encode.py --num_batch $bnum
```

## Scores extraction

After the generation step,  use the following command to obtain self-evaluation scores and uncertainty/confidence scores.

```bash
python3 -m dataeval/load_run.py --idx $SGE_TASK_ID --batch_size $bsize --data $data --model $model
```

## Conformal Alignment implementation
The script ```_fdr.py``` implements the Conformal Alignment procedure and calculates FDR and power.

```bash
python3 -m _fdr.py --data "cxr" --model "trained" --N 2000 --split_pr 0.5 --split_pr_tune 0.2
```





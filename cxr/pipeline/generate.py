import torch
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from datasets import load_dataset, Dataset
from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer
from transformers import AutoImageProcessor

sys.path.append('.')
sys.path.append('./utils')
sys.path.append('./pipeline')

from _utils import *
from chexbert import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size.')
parser.add_argument('--idx', type=int, default=0,
                    help='index of batch.')
args = parser.parse_args()

mimic_dir = '../../UQ-CXR/physionet.org/files/mimic-cxr-resized-224/'
ckpt = "./models/trained_p101112"

split_name = 'dev'

num_gens = 10


# load model
print("Loading model...")
trained_model = VisionEncoderDecoderModel.from_pretrained(ckpt)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        use_fast=True
    )
tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
tokenizer.pad_token = tokenizer.eos_token

print("Loading image processor...")
image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
)


# Load dataset
print("Loading dataset...")
base = "./ap_pa_per_dicom_id"
dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(base, "train.jsonl"),
        "dev": os.path.join(base, "dev.jsonl"),
        "validate": os.path.join(base, "validate.jsonl"),
        "test": os.path.join(base, "test.jsonl"),
    }
)


data_sub1 = dataset['dev']
data_sub2 = dataset['test']
dt1 = [data_sub1[i] for i in range(len(data_sub1))]
dt2 = [data_sub2[i] for i in range(len(data_sub2))]
dt1.extend(dt2)
data_sub = Dataset.from_list(dt1)
print(len(data_sub))


filename = f"./output/generations/generations_{len(data_sub)}_{args.idx}.pkl"
os.makedirs(os.path.dirname(filename), exist_ok=True)

sequences = []
bs = 1
for i in tqdm(range((args.idx-1)*args.batch_size, args.idx*args.batch_size, bs), dynamic_ncols=True):
    test_sub = data_sub[i : min(len(data_sub), i + bs)]

    true_finding = [''.join(text.split(get_before_findings(text))).strip()  for text in test_sub['report']]

    test_prompts_sub = [get_before_findings(text) for text in test_sub['report']]
    tokenizer.padding_side = "left"
    test_input_ids_sub, test_attention_masks_sub = tokenizer(test_prompts_sub, return_tensors="pt", padding=True).values()

    images_sub = [read_image(os.path.join(mimic_dir, image_path), mode=ImageReadMode.RGB) for image_path in test_sub['image_path']]
    # images_sub = [read_image(os.path.join(mimic_dir, test_sub['image_path']), mode=ImageReadMode.RGB)]

    image_size = image_processor.size['height']
    inference_image_transformations = Transform(image_size, image_processor.image_mean, image_processor.image_std)
    inference_image_transformations = torch.jit.script(inference_image_transformations)

    def inference_jit_image_processor(images):
        return torch.stack([inference_image_transformations(image) for image in images], dim=0)

    test_pixel_values_sub = inference_jit_image_processor(images_sub)

    test_sub["pixel_values"] = test_pixel_values_sub
    test_sub["decoder_input_ids"] = test_input_ids_sub

    input_length = test_input_ids_sub.shape[1]

    input_datum = {
        "pixel_values": test_pixel_values_sub,
        "decoder_input_ids": test_input_ids_sub,
        "decoder_attention_mask": test_attention_masks_sub,
    }

    most_likely_generations = trained_model.generate(
        **input_datum, 
        max_length=512, 
        do_sample=False,
        num_beams=4,
        num_return_sequences=1
        )

    test_sub['generated'] = tokenizer.decode(most_likely_generations[0], skip_special_tokens=True)

    most_likely_generations = most_likely_generations.cpu()[0, input_length:]
    
    generations = []
    num_g = num_gens
    while num_g > 0:
        _ =  trained_model.generate(
        **input_datum, 
        max_length=512, 
        do_sample=True,
        num_return_sequences=1
        )
        generations.append(_[:, input_length:].cpu())
        num_g -= len(_)

    generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
    generations = generations.reshape(-1, generations.shape[-1])[:num_gens]
    generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
    
    # remember the data
    curr_seq = dict(
        prompt=test_input_ids_sub.cpu()[0],
        id=test_sub['dicom_id'][0],
        question=test_prompts_sub[0],
        answer=true_finding[0],
        image=test_pixel_values_sub.cpu()[0],
        attention_masks=test_attention_masks_sub[0],
        additional_answers=[],
    )
    curr_seq.update(
        dict(
            most_likely_generation_ids = most_likely_generations,
            generations_ids=generations,
        )
    )
    curr_seq.update(
        dict(
            most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
            generations=generated_texts,
        )
    )

    y_hat, y, study_id = test_sub['generated'], test_sub['report'], test_sub['study_id']
    df_y, df_y_hat, f1_class, scores = chexbert_eval(y_hat, y, study_id)

    tp = (df_y_hat == df_y).astype(float)
    tp_eg_most_likely = tp.sum(1).to_list()


    chexbert_score_list = []
    for gens in generated_texts:
        y_hat, y, study_id = gens, test_sub['report'], test_sub['study_id']
        df_y, df_y_hat, f1_class, scores = chexbert_eval(y_hat, y, study_id)

        tp = (df_y_hat == df_y).astype(float)
        tp_eg = tp.sum(1).to_list()

        chexbert_score_list.extend(tp_eg)

    curr_seq.update(
        dict(
            chexbert_score = tp_eg_most_likely,
            chexbert_score_generations = chexbert_score_list
        )
    )
    sequences.append(curr_seq)

print(sequences)
pd.to_pickle(sequences, filename)

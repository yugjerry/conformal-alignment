import getpass
import os
import sys

_BASE_DIR = '/home/mnt/projects/'

DATA_FOLDER = os.path.join(_BASE_DIR, 'UQ-NLG-main')
LLAMA_PATH = f'{_BASE_DIR}/alignment/llama/'

GENERATION_FOLDER = os.path.join(DATA_FOLDER, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

# After running pipeline/generate.py, update the following paths to the generated files if necessary.
GEN_PATHS = {
    'coqa': {
        'llama-2-13b-chat-hf': f'{GENERATION_FOLDER}/llama-2-13b-chat-hf_coqa_10/0.pkl',
        'opt-13b': f'{GENERATION_FOLDER}/opt-13b_coqa_10/0.pkl',
    },
    'triviaqa': {
        'llama-2-13b-chat-hf': f'{GENERATION_FOLDER}/llama-2-13b-chat-hf_triviaqa_10/0.pkl',
        'opt-13b': f'{GENERATION_FOLDER}/opt-13b_triviaqa_10/0.pkl',
    }
}
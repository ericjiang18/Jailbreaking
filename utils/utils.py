import subprocess
import re
import pandas as pd
import GPUtil
import torch.nn.functional as F
import yaml
from itertools import product
import numpy as np
import json
from transformers import AutoTokenizer

def get_tokenizer_from_path(path):

    if 'mixtral' in path:
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
    elif 'mistral' in path:
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
    elif 'llama2-7b' in path:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    elif 'llama2-13b' in path:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
    elif 'llama2-70b' in path:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat-hf')
    elif 'Yi-34b' in path:
        tokenizer = AutoTokenizer.from_pretrained('01-ai/Yi-34B-Chat')
    elif 'Yi-6b' in path:
        tokenizer = AutoTokenizer.from_pretrained('01-ai/Yi-6B-Chat')
    elif 'gemma-7b-it' in path:
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b-it')
        
    return tokenizer







def entropy(labels, base=None):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0
    # Compute entropy
    base = np.e if base is None else base
    for i in probs:
        ent -= i * np.log(i) / np.log(base)
    return ent

def read_json_to_df(output_file):
    # read json file
    with open(output_file) as f:
        # load json file where the lines are separated by \
            # n
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    df = pd.DataFrame(data)
    return df
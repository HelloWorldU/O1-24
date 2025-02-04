"""
Train Qwen2.5 to test on Game24.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import numpy as np
import re
import torch
import sys
import json
import subprocess
from collections import defaultdict
from functools import partial

class LanguageModel:
    def __init__(self, model_path, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs =self.model.generate()

# 数据预处理
def create_ragged_dataset(dataset, tokenizer, batch_size=10):
    def tokenize_and_process(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]

        processed_inputs = [f"{inst}\n{inp}" for inst, inp in zip(instructions, inputs)]
        outputs = examples['output']
        
        input_encodings = tokenizer(
            processed_inputs,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None
        )
        output_encodings = tokenizer(
            outputs,
            truncation=False,
            padding=False,  
            return_tensors=None
        )
        
        processed_examples = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': output_encodings['input_ids'],
            'length': [len(ids) for ids in output_encodings['input_ids']]
        }
        
        return processed_examples

    processed_dataset = dataset.map(
        tokenize_and_process,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names
    )
    
    return processed_dataset


def dynamic_padding_collator(examples, tokenizer):
    max_len = min(1265, max(len(ex['labels']) for ex in examples))
    batch = tokenizer.pad(
        {"input_ids": [ex['input_ids'] for ex in examples], "attention_mask": [ex['attention_mask'] for ex in examples]},
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    
    labels_list = [ex['labels'] for ex in examples]
    padded_labels = torch.full((len(labels_list), max_len), -100)
    for i, label in enumerate(labels_list):
        padded_labels[i, :len(label)] = torch.tensor(label)
    batch["labels"] = padded_labels

    return batch


def prepare_datasets(train_dataset, validation_dataset, tokenizer):
    processed_train = create_ragged_dataset(train_dataset, tokenizer)
    processed_val = create_ragged_dataset(validation_dataset, tokenizer)
    
    return processed_train, processed_val


def load_datasets(train_dataset, validation_dataset, tokenizer):
    processed_train, processed_val = prepare_datasets(train_dataset, validation_dataset, tokenizer)
    return processed_train, processed_val
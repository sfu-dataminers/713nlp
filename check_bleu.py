"""
First make sure you have a Python3 program for your answer in ./answer/

Then run:

    python3 zipout.py

This will create a file `output.zip`.

To customize the files used by default, run:

    python3 zipout.py -h
"""
import sys, os, optparse, logging, tempfile, subprocess, shutil


# Import Statement
import numpy as np
import pandas as pd
import os, time
from tqdm.auto import tqdm
import torch
from datasets import Dataset, load_dataset, load_metric
from transformers import (
    MarianTokenizer,
    MBart50TokenizerFast,
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    utils)
utils.logging.set_verbosity(50)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

pretrained_models = {
        "zh":"Helsinki-NLP/opus-mt-en-zh",
        "ja":"Helsinki-NLP/opus-tatoeba-en-ja"
    }


def compute_bleu(eval_preds, xxlang= 'zh'):
    metric = load_metric("sacrebleu")
    tokenizer= load_tokenizer(pretrained_models[xxlang])
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels] 
        
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": round(result["score"], 4)}
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = round(np.mean(prediction_lens), 4)
    return result

def load_tokenizer(tk_model):
    mr_tknzr = MarianTokenizer.from_pretrained(tk_model)
    with open('data/tokens.txt') as fp:
        tokens = fp.read().split("\n")
    mr_tknzr.add_tokens(tokens, special_tokens=True)
    return mr_tknzr

def data_preproces(examples, pretrained_tknzr):
    ip_sent = [s for s in examples["Sent_en"]]
    target_sent = [s for s in examples["Sent_yy"]]
    model_ip = pretrained_tknzr(ip_sent, max_length=128, truncation=True, padding="max_length")
    labels = pretrained_tknzr(target_sent, max_length=128, truncation=True, padding="max_length")
    if len(examples['Sent_en']) > 1 and (len(model_ip['input_ids'][0]) != len(model_ip["input_ids"][1])):
        print ("Error!", )
    model_ip["labels"] = labels["input_ids"]
    return model_ip


def load_data(data_path, yylang = 'hi', batch_size = 8, xxlang = 'zh'):
    pretrained_tknzr = load_tokenizer(pretrained_models[xxlang])
    data_files = {"train": "train.parquet", "test": "test.parquet", "eval": "eval.parquet"}
    dataset = load_dataset(data_path, data_files= data_files)
    data_subset = dataset.filter(lambda example: example['lang_yy'] == yylang)
    data_subset = data_subset.map(data_preproces, batched=True, batch_size=batch_size*3, fn_kwargs={'pretrained_tknzr': pretrained_tknzr})
    data_subset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
    return data_subset


def load_model_tokenizer(model_dir):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)#.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def get_score(model, tokenizer, data_subset, model_name = 'temp'):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    args = Seq2SeqTrainingArguments(
        output_dir = model_name,
        seed = 99,
        evaluation_strategy = "epoch",
        log_level = 'warning',
        disable_tqdm = False,
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        warmup_steps = 10,
        num_train_epochs=10,
        predict_with_generate=True,
        remove_unused_columns = True,
        fp16=False,
        save_total_limit=3,
        save_strategy = "epoch",
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=data_subset["train"],
        eval_dataset=data_subset["eval"],
        data_collator=data_collator,
        tokenizer= tokenizer,
        compute_metrics= compute_bleu
    )
    score = trainer.predict(data_subset["test"])
    return score


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data",  default= 'data', help="Path to data")
    #Change the file names, as per the loaded model
    optparser.add_option("-m", "--model", default='NMT_PFT_en-sh-to-ms', help="Path to model directory")
    # Similarly, change the target language as per the model
    optparser.add_option("-t", "--targetlang", default='ms', help="Path to model directory")
    optparser.add_option("-b", "--batch_size", default=8, help="Path to model directory")
    optparser.add_option("-x", "--xlang", default='zh', help="Path to model directory")

    (opts, _) = optparser.parse_args()

    datag = load_data(opts.data, opts.targetlang, opts.batch_size, opts.xlang)
    modelg,tokenizer = load_model_tokenizer(opts.model)
    scr = get_score(modelg, tokenizer, datag)
    print(scr)


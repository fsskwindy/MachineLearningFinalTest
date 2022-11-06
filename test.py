# encoding=gbk
import re
import json
import torch
import random
import numpy as np
import pandas as pd
from rouge import Rouge
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ���� rouge_L ����
def print_rouge_L(output, label):
    rouge = Rouge()
    rouge_score = rouge.get_scores(output, label)

    rouge_L_f1 = 0
    rouge_L_p = 0
    rouge_L_r = 0
    for d in rouge_score:
        rouge_L_f1 += d["rouge-l"]["f"]
        rouge_L_p += d["rouge-l"]["p"]
        rouge_L_r += d["rouge-l"]["r"]
    print("rouge_f1:%.2f" % (rouge_L_f1 / len(rouge_score)))
    print("rouge_p:%.2f" % (rouge_L_p / len(rouge_score)))
    print("rouge_r:%.2f" % (rouge_L_r / len(rouge_score)))


# --------------------------��ȡ����--------------------------
train_path = 'data/train_dataset.csv'  # �Զ���ѵ����·��
with open(train_path, 'r', encoding='utf-8') as f:
    train_data_all = f.readlines()

test_path = 'data/test_dataset.csv'  # �Զ�����Լ�·��
with open(test_path, 'r', encoding='utf-8') as f:
    test_data = f.readlines()

train = pd.DataFrame([], columns=["Index", "Text", "Abstract"])
test = pd.DataFrame([], columns=["Index", "Text"])
for idx, rows in enumerate(train_data_all):
    train.loc[idx] = rows.split("\t")
for idx, rows in enumerate(test_data):
    test.loc[idx] = rows.split("\t")

# --------------------------���ض�����T5ģ��--------------------------
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

model_name = "./PretrainModel/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # --------------------------�鿴��������Ԥ����-------------------------
i = 0  # �Ե�һ���������ݽ���Ԥ��
article_text = train["Text"][i]
article_abstract = train["Abstract"][i]

input_ids = tokenizer(
    [WHITESPACE_HANDLER(article_text)],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=512,
    min_length=int(len(article_text) / 32),
    no_repeat_ngram_size=3,
    num_beams=5
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(f"Generate��\n{summary}")
print(f"Label��\n{article_abstract}")
print_rouge_L(summary, article_abstract)
#
# # --------------------------����10��������Rouge_L-------------------------
# multi_sample = 10
# for idx, article_text in tqdm(enumerate(train["Text"][:multi_sample]), total=multi_sample):
#     input_ids = tokenizer(
#         [WHITESPACE_HANDLER(article_text)],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=512
#     )["input_ids"]
#
#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_length=512,
#         min_length=int(len(article_text) / 32),
#         no_repeat_ngram_size=3,
#         num_beams=5
#     )[0]
#
#     summary = tokenizer.decode(
#         output_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#     train.loc[idx, "summary"] = summary
#
# print_rouge_L(train["summary"][:multi_sample], train["Abstract"][:multi_sample])

# --------------------------Ԥ��-------------------------
# for idx, article_text in tqdm(enumerate(test["Text"]), total=1000):
#     input_ids = tokenizer(
#         [WHITESPACE_HANDLER(article_text)],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=768
#     )["input_ids"]
#
#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_length=512,
#         min_length=int(len(article_text) / 32),
#         no_repeat_ngram_size=3,
#         num_beams=5
#     )[0]
#
#     summary = tokenizer.decode(
#         output_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#     test.loc[idx, "Text"] = summary
#
# test[["Index", "Text"]].to_csv("summit.csv", index=False, header=False, sep="\t")

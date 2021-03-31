import os
import sys
sys.path.append("../")
import json
# import gzip
# import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
import collections
import random
from nltk.corpus import stopwords
from allennlp_models.my_project.dataset_readers.corenlp import CoreNLP

#sentenceとquestionで共通するノンストップワードが一つもない場合False
def check_overlap(sentence,question, stop_words):
    for w in question.split():
        if sentence.find(w)!=-1 and w not in stop_words:
            return True
    return False


def answer_find(context_text,answer_start,answer_end):
    print(f"context_text = {context_text}")
    context=sent_tokenize(context_text)
    start_p=0

    #start_p:対象となる文の文字レベルでの始まりの位置
    #end_p:対象となる文の文字レベルでの終端の位置
    #answer_startがstart_pからend_pの間にあるかを確認。answer_endも同様
    for i,sentence in enumerate(context):
        end_p=start_p+len(sentence)
        if start_p<=answer_start and answer_start<=end_p:
            sentence_start_id=i
        if start_p<=answer_end and answer_end<=end_p:
            sentence_end_id=i
        #スペースが消えている分の追加、end_pの計算のところでするべきかは不明
        start_p+=len(sentence)+1

    #得られた文を結合する（大抵は文は一つ）
    answer_sentence=" ".join(context[sentence_start_id:sentence_end_id+1])

    print(f"answer_sentence = {answer_sentence}")
    print("")
    return answer_sentence

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in word_tokenize(sent)]

def data_process(input_path,interro_path,train=False):
    with open(input_path,"r") as f:
        data=json.load(f)
    with open(interro_path,"r") as f:
        interro_data=json.load(f)

    # corenlp=CoreNLP()

    questions=[]
    answers=[]
    sentences=[]

    stop_words = stopwords.words('english')


    all_count=0

    #context_text:文章
    #question_text:質問
    #answer_text:解答
    #answer_start,answer_end:解答の文章の中での最初と最後の位置
    for topic in tqdm(data["data"]):
        topic=topic["paragraphs"]
        for paragraph in topic:
            context_text=paragraph["context"].lower()
            for qas in paragraph["qas"]:

                question_text=qas["question"].lower()
                a=qas["answers"][0]
                answer_text=a["text"].lower()
                answer_start=a["answer_start"]
                answer_end=a["answer_start"]+len(a["text"])
                interro=interro_data[all_count]["interro"]
                non_interro=interro_data[all_count]["non_interro"]
                #contextの中からanswerが含まれる文を見つけ出す
                sentence_text=answer_find(context_text,answer_start,answer_end)

                question_text=" ".join(tokenize(question_text))
                sentence_text=" ".join(tokenize(sentence_text))
                answer_text=" ".join(tokenize(answer_text))

                # print(f"question_text = {question_text}")
                # print(f"sentence_text = {sentence_text}")
                # print(f"answer_text = {answer_text}")

                # print("")

                # #ゴミデータ(10個程度)は削除
                # if len(question_text)<=5:
                #     continue
                #
                # #テキストとノンストップワードが一つも重複してないものは除去
                # if check_overlap(sentence_text,question_text,stop_words):
                #     continue

                sentences.append(sentence_text)
                questions.append(question_text)
                answers.append(answer_text)
    # print("=" * 110)
    # print("src")
    # print("=" * 110)
    # for i, line in enumerate(sentences):
    #     print(f"{i}) " + line + "\n")
    #
    # print("")
    # print("="* 110)
    # print("")
    #
    #
    # print("=" * 110)
    # print("trg")
    # print("=" * 110)
    #
    # for i, line in enumerate(questions):
    #
    #     print(f"{i}) " + line + "\n")
    #

    # if train==True:
    #     with open("data/squad-src-train.txt","w")as f:
    #         for line in sentences:
    #             f.write(line+"\n")
    #     with open("data/squad-tgt-train.txt","w")as f:
    #         for line in questions:
    #             f.write(line+"\n")
    #
    # if train==False:
    #     random_list=list(range(len(questions)))
    #     random.shuffle(random_list)
    #     val_num=int(len(random_list)*0.5)
    #     with open(FIXTURES_ROOT / "rc" / "squad-src-val.txt","w")as f:
    #         for line in sentences[0:val_num]:
    #             f.write(line+"\n")
    #     with open(FIXTURES_ROOT / "rc" / "squad-tgt-val.txt","w")as f:
    #         for line in questions[0:val_num]:
    #             f.write(line+"\n")
    #     with open(FIXTURES_ROOT / "rc" / "squad-src-test.txt","w")as f:
    #         for line in sentences[val_num:]:
    #             f.write(line+"\n")
    #     with open(FIXTURES_ROOT / "rc" / "squad-tgt-test.txt","w")as f:
    #         for line in questions[val_num:]:
    #             f.write(line+"\n")

from tests import FIXTURES_ROOT

data_process(input_path=FIXTURES_ROOT / "rc" / "squad.json",
             interro_path=FIXTURES_ROOT / "rc" / "squad-interro-train",
             train=False
             )


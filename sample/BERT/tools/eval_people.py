#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import sys
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from bert4torch.snippets import sequence_padding
import time
from bert4torch.tokenizers import Tokenizer
import argparse

os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))
__dir__ = os.path.dirname(os.path.abspath("../tools/eval_people.py"))
sys.path.append(__dir__)
sys.path.append(__dir__ + "/../")
sys.path.append(__dir__ + "/../python")
maxlen=256
from python.bert_sail import dataset
def pre_process_dateset(inputs,tokenizer):#pre_process_dataset test

    token_ids_, labels_ = [], []
    for d in inputs:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        mapping = tokenizer.rematch(d[0], tokens)#label mappping
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)

        labels = ['O']*len(token_ids)
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = 'B-'+label
                labels[start + 1:end + 1] = ['I-'+label]*(end-start)#prepare labels
                 
        token_ids_.append(token_ids)
        labels_.append(labels)
    token_ids_ =sequence_padding(token_ids_)
 
    return token_ids_,labels_

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--test_path', type=str, default="../datasets/china-people-daily-ner-corpus/example.test", help='test_path')
    parser.add_argument('--input_path', type=str, default="../python/results/bert4torch_output_fp32_1b.bmodel_sail_python_result.txt", help='cpp output path')
    parser.add_argument('--dict_path', type=str, default="../models/pre_train/chinese-bert-wwm/vocab.txt", help='pre_train_vab_path')
    opt = parser.parse_args()
    return opt
def test(path,y_trues):#test txt
    f=open(path,'r')
    y_preds_=[]
    y_trues_=[]
    tot=0
    for line in f.readlines():
        
        if(len(eval(line))!=len(y_trues[tot])):
            tot+=1
            continue
       
        y_trues_.append(y_trues[tot])
        y_preds_.append(eval(line))
       
        tot+=1

    print("accuary: ", accuracy_score(y_trues_, y_preds_))
    print("p: ", precision_score(y_trues_, y_preds_))
    print("r: ", recall_score(y_trues_, y_preds_))
    print("f1: ", f1_score(y_trues_, y_preds_))
    print("classification report: ")
    print(classification_report(y_trues_, y_preds_))
def main(opt):
    
    D=dataset.load_data(opt.test_path)
    tokenizer = Tokenizer(opt.dict_path,do_lower_case=True)
    _,trues=pre_process_dateset(D,tokenizer)
    test(opt.input_path,trues)
   
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    print('all done.')
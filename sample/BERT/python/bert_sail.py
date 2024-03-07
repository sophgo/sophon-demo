#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import sophon.sail as sail
import torch
import argparse
import time

from utils.dataset import MyDataset
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
import numpy as np
from bert4torch.layers import CRF
from bert4torch.tokenizers import Tokenizer
from collections import OrderedDict
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
tot_pre,tot_infer,tot_post,tot_time=0,0,0,0
maxlen = 256
categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
categories_id2label = {i: k for i, k in enumerate(categories)}
categories_label2id = {k: i for i, k in enumerate(categories)}
dataset=MyDataset()

class BERT:
    def __init__(self,model_path,dict_path,if_crf=True):
        self.io_mode = sail.IOMode.SYSIO
        self.device=0
        self.engine=sail.Engine(model_path,self.device,self.io_mode)
        self.handle=self.engine.get_handle()
        self.bmcv=sail.Bmcv(self.handle)
        self.graph_names=self.engine.get_graph_names()
        self.input_names={}
        self.output_names={}
        self.max_input_shape={}
        self.input_shapes={}
        self.input_dtypes={}
        self.img_dtypes={}
        self.input_scales={}
        self.max_output_shape={}
        self.output_shapes={}
        self.output_dtypes={}
        self.output_scales={}
        self.dict_path=dict_path

        self.tokenizer = Tokenizer(self.dict_path,do_lower_case=True)
        self.if_crf=if_crf
        if(self.if_crf):
            self.crf = CRF(len(categories))
        self.token2text={}
    
      
    
        for graph_name in self.graph_names:
            self.input_names[graph_name]=self.engine.get_input_names(graph_name)
            self.output_names[graph_name]=self.engine.get_output_names(graph_name)
            self.max_input_shape[graph_name]=self.engine.get_max_input_shapes(graph_name)
            input_shape_dict={}
            for input_name in self.input_names[graph_name]:
                input_shape_dict[input_name]=self.engine.get_input_shape(graph_name,input_name)
            self.input_shapes[graph_name]=input_shape_dict
            output_shape_dict={}
            for output_name in self.output_names[graph_name]:
                output_shape_dict[output_name]=self.engine.get_output_shape(graph_name,output_name)
            self.output_shapes[graph_name]=output_shape_dict
            input_dtype_dict={}
            img_dtypes_dict={}
            for input_name in self.input_names[graph_name]:
                input_dtype_dict[input_name]=self.engine.get_input_dtype(graph_name,input_name)
                img_dtypes_dict[input_name] = self.bmcv.get_bm_image_data_format(input_dtype_dict[input_name])
                
            self.input_dtypes[graph_name]=input_dtype_dict
            self.img_dtypes[graph_name]=img_dtypes_dict
            output_dtype_dict={}
            for output_name in self.output_names[graph_name]:
                output_dtype_dict[output_name]=self.engine.get_output_dtype(graph_name,output_name)
            self.output_dtypes[graph_name]=output_dtype_dict
            input_scale_dict={}
            for input_name in self.input_names[graph_name]:
                input_scale_dict[input_name]=self.engine.get_input_scale(graph_name,input_name)
            self.input_scales[graph_name]=input_scale_dict
            output_scale_dict={}
            for output_name in self.output_names[graph_name]:
                output_scale_dict[output_name]=self.engine.get_output_scale(graph_name,output_name)
            self.output_scales[graph_name]=output_scale_dict
            
        self.batch_size = self.input_shapes[self.graph_names[0]][self.input_names[self.graph_names[0]][0]][0]
        suppoort_batch_size = [1, 8]
        if self.batch_size not in suppoort_batch_size:
            raise ValueError('batch_size must be {} for bmcv, but got {}'.format(suppoort_batch_size, self.batch_size))
        
    def pre_process_text(self,input):#pre_process_dev test

        tokenizer=self.tokenizer
     
        tokens = tokenizer.tokenize(input, maxlen=maxlen)
        token_ids = tokenizer.tokens_to_ids(tokens)#tokenize
       
        for i in range(maxlen-len(token_ids)):
            token_ids.append(0)#padding
        return tokens,token_ids
    def pre_process_dataset(self,inputs):#pre_process_dataset test

        token_ids_, labels_ = [], []
        tokenizer=self.tokenizer
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
    def post_postprocess(self,out_infer):#post_postprocess
        emission_score, attention_mask=out_infer
        if self.if_crf:#crf layer
            best_path = self.crf.decode(torch.tensor(emission_score,dtype=torch.float), torch.tensor(attention_mask,dtype=torch.float))  # [btz, seq_len]
        else :
            best_path=torch.tensor(emission_score.argmax(axis=2))
        return best_path
    def get_input_feed_numpy(self, input_names, inputs):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param inputs: image_numpy
        :return:
        """
        input_feed = {}
        for i, (name, input) in enumerate(zip(input_names, inputs)):
            input_feed[name] = np.ascontiguousarray(input)
        return input_feed
    def infer_numpy(self, input_data):
        """
        input_data: [input0, input1, ...]
        Args:
            input_data:

        Returns:

        """
        # logger.debug("input_data shape: {}".format(input_data.shape))
        inputs_feed = self.get_input_feed_numpy(self.input_names[self.graph_names[0]], input_data)
        # print(inputs_feed['token_ids.1'].shape,"----------")
        outputs = self.engine.process(self.graph_names[0], inputs_feed)
        outputs_dict = OrderedDict()
        for name in self.output_names[self.graph_names[0]]:
            outputs_dict[name] = outputs[name]
        # logger.debug(outputs.keys())
        return outputs_dict
        # return self.outputToList_numpy(outputs)
    
    def softmax(self,x):
        """ softmax function """
        x = np.exp(x) / np.sum(np.exp(x), axis = 2, keepdims = True)
        return x
    def test_text(self,text):#one test
        
        tokens,token_ids=self.pre_process_text(text)
        token_ids=[token_ids]
        token_ids_b=[token_ids[i:i+self.batch_size] for i in range(0,len(token_ids),self.batch_size)]
        if(len(token_ids_b[-1])%self.batch_size):
            zero=np.zeros((self.batch_size-len(token_ids_b[-1]),256))
            zero[:,0]=101
            zero[:,0]=102
            token_ids_b[-1]=np.concatenate((token_ids_b[-1],zero))
        out=self.infer_numpy([token_ids_b[0]])
        lis=[]
        for i in out.keys():
            lis.append(out[i])
        lis[0]=self.softmax(lis[0])

        ans=self.post_postprocess(lis)
        
        ans=self.trans_entity2tuple(ans,tokens,token_ids)
        return ans
    def test_dataset(self,texts):#dataset test
        global tot_pre,tot_infer,tot_post,tot_time
        y_trues=[]
        y_preds=[]
        s=time.time()
        ss=time.time()
        token_ids,labels=self.pre_process_dataset(texts)
        
        token_ids_b=[token_ids[i:i+self.batch_size] for i in range(0,len(token_ids),self.batch_size)]
        labels_b=[labels[i:i+self.batch_size] for i in range(0,len(labels),self.batch_size)]
        if(len(token_ids_b[-1])%self.batch_size):
            zero=np.zeros((self.batch_size-len(token_ids_b[-1]),256))
            zero[:,0]=101
            zero[:,0]=102
            token_ids_b[-1]=np.concatenate((token_ids_b[-1],zero))
        tot_pre=time.time()-s
        count = 0
        for token_id,label in zip(token_ids_b,labels_b):
            print("processed: {}/{}".format(count * self.batch_size, len(token_ids)))
            count += 1
            s=time.time()
            out=self.infer_numpy([token_id])
            tot_infer+=time.time()-s
            s=time.time()
           
            lis=[]
            for i in out.keys():
                lis.append(out[i])
            
            lis[0]=self.softmax(lis[0])
            ans=self.post_postprocess(lis)
            ans=self.trans_entity2label(ans)
          
            y_true=label
            for i in range(min(len(label),self.batch_size)):
                ans[i]=ans[i][:len(label[i])]
            
                y_trues.append(y_true[i])
                y_preds.append(ans[i])
               
            tot_post+=time.time()-s

        
        tot_time=time.time()-ss
        return y_preds
    def trans_entity2label(self,scores):
        '''translate entitys to label
        '''
        entity_ids_= []
        for i in range(self.batch_size):

            entity_ids = []
            for j, item in enumerate(scores[i]):
                flag_tag = categories_id2label[item.item()]
                entity_ids.append(flag_tag)
            entity_ids_.append(entity_ids)
        return entity_ids_
    def trans_entity2tuple(self,scores,tokens,token_ids):
        '''translate entitys to tuple
        '''
        batch_entity_ids = set()
        for i, one_samp in enumerate(scores):
            entity_ids = []
            for j, item in enumerate(one_samp):
                # if(item.item()==0):break
                flag_tag = categories_id2label[item.item()]
                if flag_tag.startswith('B-'):  # B
                    entity_ids.append([i, j, j, flag_tag[2:]])
                elif len(entity_ids) == 0:
                    continue
                elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:]==entity_ids[-1][-1]):  # I
                    entity_ids[-1][-2] = j
                elif len(entity_ids[-1]) > 0:
                    entity_ids.append([])

            for i in entity_ids:
                string=""
                
                if i:
                    for j in range(i[1],i[2]+1):
                        string +=tokens[j]
                    batch_entity_ids.add(tuple([string,i[3]]))
        return batch_entity_ids
    

def parse_args():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default="../models/BM1684/bert4torch_output_fp32_1b.bmodel", help='bmodel path')
    parser.add_argument('--dev_id', type=int, default=0, help='device id')
    parser.add_argument('--input', type=str, default="../datasets/china-people-daily-ner-corpus/example.test", help='test_path')
    parser.add_argument('--dict_path', type=str, default="../models/pre_train/chinese-bert-wwm/vocab.txt", help='pre_train_vab_path')
    parser.add_argument('--if_crf', type=bool, default=True, help='if using crf')
    args = parser.parse_args()
    return args
def main(args):
    # check params
    
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    if not os.path.exists(args.dict_path):
        raise FileNotFoundError('{} is not existed.'.format(args.dict_path))
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
  
    bert=BERT(args.bmodel,args.dict_path)
    if(args.input[-4:]=='test'):
        if not os.path.exists(args.input):
            raise FileNotFoundError('{} is not existed.'.format(args.input))
        D=dataset.load_data(args.input)
        y_preds=bert.test_dataset(D)
        print("avg_tot_time",tot_time/len(D))
        print("avg_pre_time",tot_pre/len(D))
        print("avg_infer_time",tot_infer/len(D))
        print("avg_post_time",tot_post/len(D))
        output_path="../python/results/"+args.bmodel.split('/')[-1]+"_sail_python_result.txt"
        f=open(output_path,'w+')
        for i in y_preds:
            f.write(str(i)+'\n')
    else:
        if not os.path.exists(args.input):
            raise FileNotFoundError('{} is not existed.'.format(args.input))
        f=open(args.input)
        text=f.readlines()[0]
        ans=bert.test_text(text)
        print(ans)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('all done.')

import os
import sys
import json
import torch
import random
import string
import logging
import numpy as np
import pickle as pkl
from shutil import copyfile
from my_utils.tokenizer import UNK_ID

def load_meta(opt, meta_path):
    with open(meta_path, 'rb') as f:
        meta = pkl.load(f)
    embedding = torch.Tensor(meta['embedding'])
    opt['pos_vocab_size'] = len(meta['vocab_tag'])
    opt['ner_vocab_size'] = len(meta['vocab_ner'])
    opt['vocab_size'] = len(meta['vocab'])
    return embedding, opt

def load_meta_with_vocab(opt, meta_path):
    with open(meta_path, 'rb') as f:
        meta = pkl.load(f)
    embedding = torch.Tensor(meta['embedding'])
    opt['pos_vocab_size'] = len(meta['vocab_tag'])
    opt['ner_vocab_size'] = len(meta['vocab_ner'])
    opt['vocab_size'] = len(meta['vocab'])
    return embedding, opt, meta['vocab']

class BatchGen:
    def __init__(self, data_path, batch_size, gpu, is_train=True, doc_maxlen=1000, dropout_w=0.05, dw_type=0,
                 with_label=False,data_json=None):
        self.batch_size = batch_size
        self.doc_maxlen = doc_maxlen
        self.is_train = is_train
        self.gpu = gpu


        if data_path is None:
            self.data = data_json
        else:
            self.data_path = data_path
            self.data = self.load(self.data_path, is_train, doc_maxlen)

        self.dropout_w = dropout_w
        self.dw_type = dw_type

        if is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            data = [self.data[i] for i in indices]

        data = [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]
        self.data = data
        self.offset = 0
        self.with_label = with_label


    def load(self, path, is_train=True, doc_maxlen=1000):
        with open(path, 'r', encoding='utf-8') as reader:
            # filter
            data = []
            cnt = 0
            for line in reader:
                sample = json.loads(line)
                cnt += 1
                if is_train and (len(sample['doc_tok']) > doc_maxlen or \
                                 sample['start'] is None or sample['end'] is None):
                    import pdb; pdb.set_trace()
                    print(sample['uid'])
                    continue
                data.append(sample)
            print('Loaded {} samples out of {}'.format(len(data), cnt))
            return data

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
        self.offset = 0

    def __random_select__(self, arr):
        if self.dropout_w > 0:
            if self.dw_type > 0:
                ids = list(set(arr))
                ids_size = len(ids)
                random.shuffle(ids)
                ids = set(ids[:int(ids_size * self.dropout_w)])
                return [UNK_ID if e in ids else e for e in arr]
            else:
                return [UNK_ID if random.uniform(0, 1) < self.dropout_w else e for e in arr]
        else: return arr

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            batch_size = len(batch)
            batch_dict = {}

            doc_len = max(len(x['doc_tok']) for x in batch)
            # feature vector
            feature_len = len(eval(batch[0]['doc_fea'])[0]) if len(batch[0].get('doc_fea', [])) > 0 else 0
            doc_id = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_tag = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_ent = torch.LongTensor(batch_size, doc_len).fill_(0)
            doc_feature = torch.Tensor(batch_size, doc_len, feature_len).fill_(0)
            query_len = max(len(x['query_tok']) for x in batch)
            query_id = torch.LongTensor(batch_size, query_len).fill_(0)

            for i, sample in enumerate(batch):
                select_len = min(len(sample['doc_tok']), doc_len)
                doc_tok = sample['doc_tok']
                query_tok = sample['query_tok']

                if self.is_train:
                    doc_tok = self.__random_select__(doc_tok)
                    query_tok = self.__random_select__(query_tok)

                doc_id[i, :select_len] = torch.LongTensor(doc_tok[:select_len])
                doc_tag[i, :select_len] = torch.LongTensor(sample['doc_pos'][:select_len])
                doc_ent[i, :select_len] = torch.LongTensor(sample['doc_ner'][:select_len])
                for j, feature in enumerate(eval(sample['doc_fea'])):
                    doc_feature[i, j, :] = torch.Tensor(feature)

                select_len = min(len(query_tok), query_len)
                query_id[i, :len(sample['query_tok'])] = torch.LongTensor(query_tok[:select_len])

            doc_mask = torch.eq(doc_id, 0)
            query_mask = torch.eq(query_id, 0)
            
            batch_dict['doc_tok'] = doc_id
            batch_dict['doc_pos'] = doc_tag
            batch_dict['doc_ner'] = doc_ent
            batch_dict['doc_fea'] = doc_feature
            batch_dict['query_tok'] = query_id
            batch_dict['doc_mask'] = doc_mask
            batch_dict['query_mask'] = query_mask

            if self.is_train:
                start = [sample['start'] for sample in batch]
                end = [sample['end'] for sample in batch]
                batch_dict['start'] = torch.LongTensor(start)
                batch_dict['end'] = torch.LongTensor(end)
                if self.with_label:
                    label = [sample['label'] for sample in batch]
                    batch_dict['label'] = torch.FloatTensor(label)

            if self.gpu:
                for k, v in batch_dict.items():
                    batch_dict[k] = v.pin_memory()
            if not self.is_train:
                batch_dict['text'] = [sample['context'] for sample in batch]
                try:
                    batch_dict['span'] = [sample['span'] for sample in batch]
                except:
                    pass
            batch_dict['uids'] = [sample['uid'] for sample in batch]
            self.offset += 1

            yield batch_dict

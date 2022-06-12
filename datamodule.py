# -*- coding: utf-8 -*-

import json, os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from typing import List

class SumDataset(Dataset):
    def __init__(self, config, tokenizer, mode = 'train'):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

        data = self._read_data(mode)
        self.data = [self._convert_to_feature(dp['document'], dp['summary']) for dp in data]

    def _read_data(self, mode):
        data_path = os.path.join('./data', mode + '.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def _convert_to_feature(self, document: str, summary: str):
        '''
            Inputs
                document, summary
            Outputs
                enc_ids, enc_mask, dec_ids, dec_mask, label_ids

            Guide
                1. tokenizer는 __init__ 에서 선언되어 있는 tokenizer 사용.
                2. HW1과 같이 tokenizer.tokenize, tokenizer.convert_tokens_to_ids method를 이용하여 구현.
                3. config에 지정한 길이에 맞춰 truncation or padding 진행.
                4. dec_ids와 label_ids는 각각 bos_token_id와 eos_token_id를 잘 고려하여 구현.
                5. dec_ids의 padding 부분의 label_ids는 loss가 흐르지 않게 하기 위해 label -100 부여.
            
            Example (enc_len=10, dec_len=5)
                document : '오늘 점심 엄청 맛있다. 또 먹자.'
                summary : '점심 맛있다.'
                           오늘    점심   엄청   맛있다. 또   먹자
                enc_ids : [21882, 482, 5221, 11482, 899, 7221, 1, 1, 1, 1]
                enc_mask : [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
                          bos 점심  맛있다.
                dec_ids : [0, 482, 11482, 1, 1]
                dec_mask : [1, 1, 1, 0, 0]
                             점심  맛있다. eos
                label_ids : [482, 11482, 2, -100, -100]
        '''
        tokenizer = self.tokenizer
        enc_len = self.config.enc_len
        dec_len = self.config.dec_len

        bos_token_id = tokenizer.bos_token_id # 0
        eos_token_id = tokenizer.eos_token_id # 2
        pad_token_id = tokenizer.pad_token_id # 1
        ignore_token_id = -100

        enc_ids, enc_mask = None, None
        dec_ids, dec_mask, label_ids = None, None, None
        
        ############################################## EDIT ################################################
        #21882, 482 ,,,,
        doc_tokenized_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(document))
        sum_tokenized_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(summary))

        #enc_ids, enc_mask
        enc_ids = doc_tokenized_idx[:enc_len]
        enc_mask = [1 for _ in range(enc_len)]
        for i in range(len(enc_ids), enc_len):
          #뒤에 패딩
          enc_ids.insert(i, 1)
          enc_mask[i] = 0
        
        #dec_ids, dec_mask, label_ids
        dec_ids = sum_tokenized_idx[:dec_len-1]
        label_ids = sum_tokenized_idx[:dec_len-1]
        dec_ids.insert(0,0) #bos token, dec_ids 길이는 dec_len
        label_ids.append(2) #eos token, label_ids 길이는 dec_len
        dec_mask = [1 for _ in range(dec_len)]
        for i in range(len(dec_ids), dec_len):
          dec_ids.insert(i,1)
          label_ids.insert(i,-100)
          dec_mask[i]=0

        #print(document)
        #print(enc_ids)
        #print(enc_mask)
        #print(summary)
        #print(dec_ids)
        #print(dec_mask)
        #print(label_ids)

        ############################################## EDIT ################################################

        assert len(enc_ids) == self.config.enc_len
        assert len(enc_mask) == self.config.enc_len
        assert len(dec_ids) == self.config.dec_len
        assert len(dec_mask) == self.config.dec_len
        assert len(label_ids) == self.config.dec_len

        return {
            'enc_ids' : enc_ids,
            'enc_mask' : enc_mask,
            'dec_ids' : dec_ids,
            'dec_mask' : dec_mask,
            'label_ids' : label_ids,
            'summary' : summary
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        output = {
            'enc_ids' : torch.tensor(self.data[i]['enc_ids'], dtype = torch.long),
            'enc_mask' : torch.tensor(self.data[i]['enc_mask'], dtype = torch.long),
            'dec_ids' : torch.tensor(self.data[i]['dec_ids'], dtype = torch.long),
            'dec_mask' : torch.tensor(self.data[i]['dec_mask'], dtype = torch.long),
            'label_ids' : torch.tensor(self.data[i]['label_ids'], dtype = torch.long),
            'summary' : self.data[i]['summary']
        }

        return output

class DataModule:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.val_dataset = None

    def train_dataloader(self):
        train_dataset = SumDataset(self.config, self.tokenizer, mode = 'train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size = self.config.train_batch_size, sampler = train_sampler)

        return train_dataloader

    def val_loss_dataloader(self):
        self.val_dataset = SumDataset(self.config, self.tokenizer, mode = 'val') if self.val_dataset is None else self.val_dataset
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, batch_size = self.config.val_batch_size, sampler = val_sampler)

        return val_dataloader
    
    def val_rouge_dataloader(self):
        self.val_dataset = SumDataset(self.config, self.tokenizer, mode = 'val') if self.val_dataset is None else self.val_dataset
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, batch_size = 1, sampler = val_sampler)

        return val_dataloader

    def test_dataloader(self):
        test_dataset = SumDataset(self.config, self.tokenizer, mode = 'test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size = 1, sampler = test_sampler)

        return test_dataloader

    def submit_dataloader(self):
        submit_dataset = SumDataset(self.config, self.tokenizer, mode = 'submit')
        submit_sampler = SequentialSampler(submit_dataset)
        submit_dataloader = DataLoader(submit_dataset, batch_size = 1, sampler = submit_sampler)

        return submit_dataloader
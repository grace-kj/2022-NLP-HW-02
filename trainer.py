# -*- coding: utf-8 -*-

import random
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import Adam
from transformers import AutoTokenizer
from rouge_score import rouge_scorer, scoring

from datamodule import DataModule
from transformer import TransformerForConditionalGeneration

def calculate_rouge(output_lns, reference_lns, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

class Trainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        self.datamodule = DataModule(self.config, self.tokenizer)
        self.model = TransformerForConditionalGeneration(self.config, self.tokenizer).to('cuda')

        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_loss_dataloader = self.datamodule.val_loss_dataloader()
        self.val_rouge_dataloader = self.datamodule.val_rouge_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()
        self.submit_dataloader = self.datamodule.submit_dataloader()

        self.optimizer = Adam(self.model.parameters(), lr = self.config.lr, eps = self.config.eps, weight_decay = self.config.weight_decay)

    def train(self):
        for epoch in tqdm(range(self.config.num_epochs)):
            self.model.zero_grad()
            self.model.train(True)

            epoch_train_loss = 0.0

            for batch in tqdm(self.train_dataloader, desc = 'Train Epoch', leave = True, position = 0):
                inputs = {
                    'enc_ids' : batch['enc_ids'].to('cuda'),
                    'enc_mask' : batch['enc_mask'].to('cuda'),
                    'dec_ids' : batch['dec_ids'].to('cuda'),
                    'dec_mask' : batch['dec_mask'].to('cuda'),
                    'label_ids' : batch['label_ids'].to('cuda')
                }

                loss = self.model(**inputs) 

                epoch_train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

            print("|{:^79}|".format(" Epoch / Total Epoch : {} / {} ".format(epoch, self.config.num_epochs)))
            print("|{:^79}|".format(" Train Loss : {:.4f} ".format(epoch_train_loss / len(self.train_dataloader))))

            self.model.train(False)

            self.valid_loss()
            # modify
            if epoch % 5 == 0:
                self.valid_rouge()


    def valid_loss(self):
        self.model.eval()

        val_loss = 0.0

        for batch in tqdm(self.val_loss_dataloader, desc = 'Validation Loss', leave = True, position = 0):
            inputs = {
                'enc_ids' : batch['enc_ids'].to('cuda'),
                'enc_mask' : batch['enc_mask'].to('cuda'),
                'dec_ids' : batch['dec_ids'].to('cuda'),
                'dec_mask' : batch['dec_mask'].to('cuda'),
                'label_ids' : batch['label_ids'].to('cuda')
            }

            loss = self.model(**inputs)

            val_loss += loss.item()

        print("|{:^79}|".format(" Valid Loss : {:.4f} ".format(val_loss / len(self.val_loss_dataloader))))

    def valid_rouge(self):
        self.model.eval()

        hyp_list = []
        sum_list = []

        for batch in tqdm(self.val_rouge_dataloader, desc = 'Validation Rouge', leave = True, position = 0):
            inputs = {
                'enc_ids' : batch['enc_ids'].to('cuda'),
                'enc_mask' : batch['enc_mask'].to('cuda'),
            }

            hypothesis = self.model.generate(**inputs)

            summary = batch['summary'][0]

            hyp_list.append(hypothesis)
            sum_list.append(summary)

        for i in range(10):
          print(hyp_list[i])
        score = calculate_rouge(hyp_list, sum_list)
        rouge1, rouge2, rougeL = score['rouge1'], score['rouge2'], score['rougeL']

        print("|{:^79}|".format(" Valid Rouge1 : {:.4f} | Valid Rouge2 : {:.4f} | Valid RougeL : {:.4f} ".format(rouge1, rouge2, rougeL)))
        

    def test(self):
        self.model.eval()

        hyp_list = []
        sum_list = []

        for batch in tqdm(self.test_dataloader, desc = 'Test', leave = True, position = 0):
            inputs = {
                'enc_ids' : batch['enc_ids'].to('cuda'),
                'enc_mask' : batch['enc_mask'].to('cuda'),
            }

            hypothesis = self.model.generate(**inputs)

            summary = batch['summary'][0]

            hyp_list.append(hypothesis)
            sum_list.append(summary)

        score = calculate_rouge(hyp_list, sum_list)
        rouge1, rouge2, rougeL = score['rouge1'], score['rouge2'], score['rougeL']

        print("|{:^79}|".format(" Test Rouge1 : {:.4f} | Test Rouge2 : {:.4f} | Test RougeL : {:.4f} ".format(rouge1, rouge2, rougeL)))

    def submit(self):
        self.model.eval()

        f = open('./submit.txt', 'w')

        for batch in tqdm(self.submit_dataloader, desc = 'Make Submit File', leave = True, position = 0):
            inputs = {
                'enc_ids' : batch['enc_ids'].to('cuda'),
                'enc_mask' : batch['enc_mask'].to('cuda')
            }        

            hypothesis = self.model.generate(**inputs)
            
            f.write(hypothesis + "\n")
        
        f.close()

    def process(self):
        self.train()
        self.test()
        self.submit()
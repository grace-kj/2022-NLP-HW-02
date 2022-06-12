# -*- coding: utf-8 -*-

import argparse, random, time
import numpy as np
import torch

from trainer import Trainer

def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class Dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def main():
    start = time.time()
    ############################################## EDIT ################################################
    config = {
        'seed' : 42,
        'num_epochs' : 50,
        'train_batch_size' : 32,
        'val_batch_size' : 32,
        'lr' : 1e-4,
        'eps' : 1e-8,
        'weight_decay' : 0.0,

        'enc_len' : 128,
        'dec_len' : 32,

        'd_model' : 512,
        'num_heads' : 8,
        'num_encoder_layers' : 3,
        'num_decoder_layers' : 3,
        'vocab_size' : 50265, # don't edit
        'pad_token_id' : 1 # don't edit
    }
    ############################################## EDIT ################################################

    config = Dotdict(config)

    set_seed(config)

    trainer = Trainer(config)
    trainer.process()

    print("Execution time : {:.4f} sec".format(time.time() - start))

if __name__ == '__main__':
    main()
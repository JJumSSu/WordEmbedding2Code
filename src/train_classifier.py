import os
import sys
import pickle
import argparse
import numpy as np

import torch
import pdb

from module import Classifier
from torchnlp.datasets import imdb_dataset

from torch.autograd import Variable


def load(args):

    with open(args.imdb_file, 'r') as f:
        data = csv.reader(f)
        for lines in data:
            pdb.set_trace()


    train_data = imdb_dataset(train=True)
    test_data  = imdb_dataset(Test=True)
    pdb.set_trace()

    # preprocessing

    # len < 400
    # generate intersected word lists (75k)
    # filter my file and glove file respectively


def to_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x
    
def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def run(args, glove_embedding, train_loader, valid_loader, test_loader):

    model       = to_cuda(classifier(args, glove_embedding))
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_ftn    = torch.nn.NLLLoss()
    best_loss   = float('inf')
    train_steps = 0
    max_train_steps = args.max_epoch * len(train_loader)
    
    for cur_epoch in range(1, args.max_epoch):
        model.train()
        train_loss  = []
        for train_step, (data, label) in enumerate(train_loader):            
            label -= 1 
            model.zero_grad()
            data, label  = to_cuda(data), to_cuda(label)
            data, label  = Variable(data), Variable(label)
            logits       = model(data)
            loss         = loss_ftn(logits, label)
            train_loss.append(loss.data.clone().item())
            train_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            
            print('| training progress [{}/{}]'.format(train_steps, max_train_steps), end='\r', flush=True)
                
        model.eval()
        valid_loss = []
        for valid_step, (data, label) in enumerate(valid_loader):
            label -= 1
            model.zero_grad()
            data, label  = to_cuda(data), to_cuda(label)
            data, label  = Variable(data), Variable(label)
            logits       = model(data)
            loss         = loss_ftn(logits, label)
            valid_loss.append(loss.data.clone().item())
        
        train_loss = np.mean(train_loss) 
        valid_loss = np.mean(valid_loss) 
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), args.path_output + args.model_name + "_best_ckpt.pt")
            
        print('| [epoch{}] trian_loss : {:.2f} valid_loss : {:.2f} best_valid_loss : {:.2f}'.format(cur_epoch, train_loss, valid_loss, best_loss))
    
    correct  = 0
    num_test = 0 
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):            
        label -= 1 
        model.zero_grad()
        data, label = to_cuda(data), to_cuda(label)
        data, label = Variable(data), Variable(label)
        logits      = model(data)
        preds       = logits.data.max(1)[1]
        corect     += torch.sum(preds == label).item()
        num_test   += len(label)
    
    print("Test Accuracy: {:.3f}".format(correct/num_test))
            
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_word2code',   action='store_true', help='to use codes?')

    parser.add_argument('--batch_size',      type=int,   default=128,      help='batch size')
    parser.add_argument('--max_epoch',       type=int,   default=15,       help='number of epochs to train')
    parser.add_argument('--lr',              type=float, default=0.0001,   help='learning rate')
    parser.add_argument('--seed',            type=int,   default=1,        help='random seed')
    parser.add_argument('--hidden_size',     type=int,   default=150,      help='number of dimensions of lstm')
    
    parser.add_argument('--code_book_len',   type=int,   default=32,       help='number of codebooks')
    parser.add_argument('--cluster_num',     type=int,   default=16,       help='length of a codebook')

    parser.add_argument('--glove_file',      type=str,   default='',                    help='input data path')
    parser.add_argument('--imdb_file',       type=str,   default='../',                    help='imdb data path')
    parser.add_argument('--path_glove',      type=str,   default='../data/',            help='input data path')
    parser.add_argument('--path_output',     type=str,   default='../output/',          help='path output codes')
    parser.add_argument('--model_name',      type=str,   default='glove.6B.300d.txt',   help='glove_file')
    
    
    args = parser.parse_args()

    if args.use_word2code:
        set_seed(args)
        train_laoder, valid_loader = load(args)
        train(args, train_loader, valid_loader)

    else:
        set_seed(args)
        train_laoder, valid_loader = load(args)
        evaluate(args, eval_loader, idx2word)


if __name__ == '__main__':
    main()

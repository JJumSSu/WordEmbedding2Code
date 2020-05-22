#-*- coding: utf-8 -*-
import os
import sys
import pickle
import argparse
import numpy as np

import torch
import pdb

from module import CODE_AE
from torch.autograd import Variable
from keras.datasets import imdb


def load_GloVe(args): 
    
    GloVe_dir = args.path_glove + args.glove_file

    if not os.path.exists(GloVe_dir[:-4] + "_glove_embeddings.pt"):
        print("| Processing GloVe File")
        f = open(GloVe_dir,'rb')

        vectors  = []
        idx2word = []
        word2idx = {}

        for idx, line in enumerate(f):
            splitLines = line.split()
            word  = splitLines[0]
            embed = np.array([float(value) for value in splitLines[1:]])

            vectors.append(embed)
            idx2word.append(word)
            word2idx[word] = idx

        vectors = np.array(vectors)
        vectors = torch.from_numpy(vectors).type('torch.FloatTensor')
        torch.save(vectors, GloVe_dir[:-4] + "_glove_embeddings.pt")

        with open(GloVe_dir[:-4] + "_word2idx.pkl", 'wb') as f:
            pickle.dump(word2idx, f)
            f.close()

        with open(GloVe_dir[:-4] + "_idx2word.pkl", 'wb') as f:
            pickle.dump(idx2word, f)
            f.close()

    else:
        print("| Loading Processed File")

        vectors  = torch.load(GloVe_dir[:-4] + "_glove_embeddings.pt")

        with open(GloVe_dir[:-4] + "_idx2word.pkl", 'rb') as f:
            idx2word = pickle.load(f)
            f.close()

        with open(GloVe_dir[:-4] + "_word2idx.pkl", 'rb') as f:
            word2idx = pickle.load(f)
            f.close()
               
    return vectors, idx2word, word2idx


def to_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x
    

def set_seed(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def load(args):

    vectors, idx2word, word2idx  = load_GloVe(args)
    valid_ids      = np.random.choice(range(len(idx2word)), args.batch_size*10, replace=False)
    word_vec_train = torch.utils.data.TensorDataset(vectors, vectors)
    word_vec_valid = torch.utils.data.TensorDataset(vectors[valid_ids], vectors[valid_ids])
    
    train_loader = torch.utils.data.DataLoader(word_vec_train, shuffle=True,  batch_size=args.batch_size, num_workers=10)
    valid_loader = torch.utils.data.DataLoader(word_vec_valid, shuffle=False, batch_size=args.batch_size, num_workers=10)
    eval_loader  = torch.utils.data.DataLoader(word_vec_train, shuffle=False, batch_size=args.batch_size, num_workers=10)

    return train_loader, valid_loader, eval_loader, idx2word


def train(args, train_loader, valid_loader):
       
    print("| Train")

    model       = to_cuda(CODE_AE(args))  # training with CPU available?
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_ftn    = torch.nn.MSELoss(reduction='sum')
    best_loss   = float("Inf")
    train_steps = 0
    max_train_steps = args.max_epoch * len(train_loader)

    for cur_epoch in range(1, args.max_epoch):
        model.train()
        train_loss  = []
        for train_step, (data, target) in enumerate(train_loader):            
            model.zero_grad()
            data, target    = to_cuda(data), to_cuda(target)
            data, target    = Variable(data), Variable(target)
            logits, rec_emb = model(data)
            loss            = loss_ftn(rec_emb, target).div(len(data))
            train_loss.append(loss.data.clone().item())
            train_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            
            print('| training progress [{}/{}]'.format(train_steps, max_train_steps), end='\r', flush=True)
                
        model.eval()
        valid_loss = []
        for valid_step, (data, target) in enumerate(valid_loader):
            model.zero_grad()
            data, target    = to_cuda(data), to_cuda(target)
            data, target    = Variable(data), Variable(target)
            logits, rec_emb = model(data)
            loss            = loss_ftn(rec_emb, target).div(len(data))
            valid_loss.append(loss.data.clone().item())
        
        train_loss = np.mean(train_loss) 
        valid_loss = np.mean(valid_loss) 

        print('| [epoch{}] trian_loss={:.2f} valid_loss={:.2f}'.format(cur_epoch, train_loss, valid_loss))

        if train_loss < best_loss * 0.99:
            best_loss = train_loss
            torch.save(model.state_dict(), args.path_output + args.model_name + "_best_ckpt.pt")
            

def evaluate(args, eval_loader, idx2word):

    print(' | Evaluation')

    model = to_cuda(CODE_AE(args))
    best_checkpoint = torch.load(args.path_output + args.model_name + '_best_ckpt.pt')
    model.load_state_dict(best_checkpoint)
    model.eval()
    test_loss = []
    
    for batch_idx, (data, target) in enumerate(eval_loader):
        data, target    = Variable(data), Variable(target)
        data, target    = to_cuda(data), to_cuda(target)
        logits, rec_emb = model(data)
        test_loss.extend(np.linalg.norm((rec_emb - target).data.cpu(), axis=1).tolist())
        _, codes = logits.max(dim=-1).squeeze()
        # logits = torch.zeros_like(logits).scatter_(-1, codes.unsqueeze(2), 1.0) # k-dimensional vector

        if batch_idx > 0:
            glove2codes = torch.cat([glove2codes, codes], dim=0)
        else:
            glove2codes = codes
            
    test_loss = np.mean(test_loss)
    print("Evluation Loss (Euc) : {:.3f}".format(test_loss))        
    print("Evluation Loss (L2)  : {:.3f}".format(test_loss**2))        
    
    glove2codes = glove2codes.cpu().numpy()
    word2codes  = {}
    
    for i, word in enumerate(idx2word):
        word2codes[str(word)] = glove2codes[i]
    
    pdb.set_trace()

    with open(args.path_output + args.model_name + "_GloVe_Codes.pkl", 'wb') as f:
        pickle.dump(word2codes, f)
        f.close()
           
        
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train',           action='store_true', help='to train?')
    parser.add_argument('--evaluate',        action='store_true', help='to evaluate?')

    parser.add_argument('--batch_size',      type=int,   default=128,      help='batch size')
    parser.add_argument('--max_epoch',       type=int,   default=200,      help='number of epochs to train')
    parser.add_argument('--lr',              type=float, default=0.0001,   help='learning rate')
    parser.add_argument('--seed',            type=int,   default=42,       help='random seed')
    parser.add_argument('--hidden_dim',      type=int,   default=300,      help='number of dimensions of input size')
    parser.add_argument('--code_book_len',   type=int,   default=32,       help='number of codebooks')
    parser.add_argument('--cluster_num',     type=int,   default=16,       help='length of a codebook')

    parser.add_argument('--glove_file',      type=str,   default='',                    help='input data path')
    parser.add_argument('--path_glove',      type=str,   default='../data/',            help='input data path')
    parser.add_argument('--path_output',     type=str,   default='../output/',          help='path output codes')
    parser.add_argument('--model_name',      type=str,   default='glove.6B.300d.txt',   help='glove_file')
    
    
    args = parser.parse_args()

    if args.train:
        set_seed(args)
        train_loader, valid_loader, _ , _ = load(args)
        train(args, train_loader, valid_loader)

    elif args.evaluate:
        set_seed(args)
        _, _, eval_loader, idx2word = load(args)
        evaluate(args, eval_loader, idx2word)


if __name__ == '__main__':
    main()




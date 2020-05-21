#-*- coding: utf-8 -*-
import sys
import pickle
import argparse
import numpy as np


import torch
import pdb

from module import CODE_AE
from torch.autograd import Variable


def load_GloVe(args): # multiprocessing?
    
    print("| Loading Glove")

    GloVe_file = args.path_glove + 'glove.42B.300d.txt'

    if not args.is_processed:
        f = open(GloVe_file,'rb')

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
        torch.save(vectors, args.path_glove + "glove_embeddings.pt")

        with open(args.path_glove + "word2idx.pkl", 'wb') as f:
            pickle.dump(word2idx, f)
            f.close()

        with open(args.path_glove + "idx2word.pkl", 'wb') as f:
            pickle.dump(idx2word, f)
            f.close()

    else:
        vectors  = torch.load(args.path_glove  + "glove_embeddings.pt")

        with open(args.path_glove + "idx2word.pkl", 'rb') as f:
            idx2word = pickle.load(f)
            f.close()

        with open(args.path_glove + "word2idx.pkl", 'rb') as f:
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
    loss_ftn    = torch.nn.MSELoss(reduction="sum")
    best_loss   = float("Inf")
    train_steps = 0
    for cur_epoch in range(1, args.max_epoch):
        model.train()
        train_loss  = []
        for train_step, (data, target) in enumerate(train_loader):            
            model.zero_grad()
            data, target = to_cuda(data), to_cuda(target)
            data, target = Variable(data), Variable(target)
            predicted    = model(data)
            loss         = loss_ftn(predicted, target).div(len(data))
            train_loss.append(loss.data.clone().item())
            train_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            
            print('| training progress [{}/{}]'.format(train_steps, args.max_train_steps), end='\r', flush=True)
            
            # if train_steps > args.max_train_steps:
            #     return
    
        model.eval()
        valid_loss = []
        for valid_step, (data, target) in enumerate(valid_loader):
            model.zero_grad()
            data, target = to_cuda(data), to_cuda(target)
            data, target = Variable(data), Variable(target)
            predicted    = model(data)
            loss         = loss_ftn(predicted, target).div(len(data))
            valid_loss.append(loss.data.clone().item())
        
        train_loss = np.mean(train_loss) / 2
        valid_loss = np.mean(valid_loss) / 2

        print('| [epoch{}] trian_loss={:.2f} valid_loss={:.2f}'.format(cur_epoch, train_loss, valid_loss))

        if train_loss < best_loss * 0.99:
            best_loss = train_loss
            torch.save(model.state_dict(), args.path_output + args.model_name + "_best_ckpt.pt")
            

def evaluate(args, eval_loader, idx2word):

    print(' | Evaluation')

    model = CODE_AE(args).cuda() 
    best_checkpoint = torch.load(args.path_output + args.model_name + '_best_ckpt.pt')
    model.load_state_dict(best_checkpoint)
    model.eval()
    loss_ftn  = torch.nn.MSELoss(reduction="sum")

    test_loss = 0 
    for batch_idx, (data, target) in enumerate(eval_loader):
        data, target  = Variable(data), Variable(target)
        data, target  = to_cuda(data), to_cuda(target)
        mask, codes   = model.encoder(data, is_training=False) # B x M x K
        predicted     = model.decoder(mask)
        loss          = loss_ftn(predicted, target)
        test_loss    += loss.item()
         
        if batch_idx > 0:
            glove2codes = torch.cat([glove2codes, codes], dim=0)
        else:
            glove2codes = codes

    print("Test Loss {:.3f}".format(test_loss/len(eval_loader)))        
    
    glove2codes = glove2codes.cpu().numpy()
    word2codes  = {}
    
    for i, word in enumerate(idx2word):
        word2codes[word] = glove2codes[i]
    
    with open(args.path_output + "GloVe_Codes.pkl", 'wb') as f:
        pickle.dump(word2codes, f)
           
        
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',      type=int,   default=128,      help='batch size')
    parser.add_argument('--max_epoch',       type=int,   default=200,      help='number of epochs to train')
    parser.add_argument('--max_train_steps', type=int,   default=10000*1024, help='number of epochs to train')
    parser.add_argument('--lr',              type=float, default=0.0001,   help='learning rate')
    parser.add_argument('--seed',            type=int,   default=42,       help='random seed')
    parser.add_argument('--hidden_dim',      type=int,   default=300,      help='number of dimensions of input size')
    parser.add_argument('--code_book_len',   type=int,   default=32,       help='number of codebooks')
    parser.add_argument('--cluster_num',     type=int,   default=16,       help='length of a codebook')

    parser.add_argument('--path_glove',      type=str,   default='../data/',   help='input data path')
    parser.add_argument('--path_output',     type=str,   default='../output/', help='path output codes')
    parser.add_argument('--model_name',      type=str,   default=''          , help='model name')
    parser.add_argument('--train',           action='store_true'             , help='to train?')
    parser.add_argument('--evaluate',        action='store_true'             , help='to evaluate?')
    parser.add_argument('--is_processed',    action='store_true'             , help='is GloVe file processed?')

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




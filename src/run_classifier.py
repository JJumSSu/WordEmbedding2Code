import os
import csv
import sys
import nltk
import random
import pickle
import argparse
import numpy as np

import torch
import pdb

from module import Classifier
from torch.autograd import Variable
from keras.datasets import imdb

def load(args):
    
    print("| Processing")

    set_seed(args)

    # Load Glove Related Files

    GloVe_dir = args.path_glove + args.glove_file
    glove_vec = torch.load(GloVe_dir[:-4] + "_glove_embeddings.pt").numpy()
    codebook  = torch.load(args.path_output + args.glove_model_name + "_codebook.pt")

    with open(GloVe_dir[:-4] + "_idx2word.pkl", 'rb') as f:
        glove_idx2word = pickle.load(f)
        f.close()

    with open(GloVe_dir[:-4] + "_word2idx.pkl", 'rb') as f:
        glove_word2idx = pickle.load(f)
        f.close()

    with open(args.path_output + args.glove_model_name + "_glove2onehot.pkl", 'rb') as f: # V x M x K
        glove_onehot = pickle.load(f)
        f.close()

    glove_onehot = torch.from_numpy(glove_onehot)
    glove_recon  = torch.matmul(glove_onehot.reshape(len(glove_vec), -1), codebook.cpu())
    glove_recon  = glove_recon.numpy()
   
    # Load IMDB

    (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=400, index_from=3)
    imdb_word2idx  = imdb.get_word_index()
    imdb_word2idx  = {k:(v+3) for k,v in imdb_word2idx.items()}
    imdb_word2idx["<pad>"]    = 0
    imdb_word2idx["<start>"]  = 1
    imdb_word2idx["<unk>"]    = 2
    imdb_word2idx["<unused>"] = 3
    imdb_idx2word  = {v: k for k, v in imdb_word2idx.items()}   
    
    joint_embed    = []
    joint_embed_re = []
    joint_idx2word = []
    joint_word2idx = {}

    n = 0
    for word in imdb_word2idx.keys(): 
        if word in glove_word2idx:
            joint_embed.append(glove_vec[glove_word2idx[word]])
            joint_embed_re.append(glove_recon[glove_word2idx[word]])
            joint_word2idx[word] = n
            joint_idx2word.append(word)
            n += 1
        
    joint_embed.append(np.mean(glove_vec, axis=0))
    joint_embed_re.append(np.mean(glove_recon, axis=0))
    joint_word2idx['<unk>'] = len(joint_idx2word)
    joint_idx2word.append('<unk>')
    
    joint_embed.append(np.random.randn(len(glove_vec[0]),))
    joint_embed_re.append(np.random.randn(len(glove_vec[0]),))
    joint_word2idx['<pad>'] = len(joint_idx2word)
    joint_idx2word.append('<pad>')

    joint_embed.append(np.random.randn(len(glove_vec[0]),))
    joint_embed_re.append(np.random.randn(len(glove_vec[0]),))
    joint_word2idx['<start>'] = len(joint_idx2word)
    joint_idx2word.append('<start>')
    
    joint_embed    = np.array(joint_embed)
    joint_embed    = torch.from_numpy(joint_embed).type('torch.FloatTensor')
    joint_embed_re = np.array(joint_embed_re)
    joint_embed_re = torch.from_numpy(joint_embed_re).type('torch.FloatTensor')
        
   
    new_x_train = []
    for sent in x_train:
        new_sent = []
        len_pad  = 400 - len(sent)
        for token in sent:
            word = imdb_idx2word[token]
            if word in joint_word2idx:
                new_sent.append(joint_word2idx[word])
            else:
                new_sent.append(joint_word2idx['<unk>'])
        for _ in range(len_pad):
            new_sent.append(joint_word2idx['<pad>'])
        new_x_train.append(np.array(new_sent))
    new_x_train = np.array(new_x_train)

    new_x_test = []
    for sent in x_test:    
        new_sent = []
        len_pad  = 400 - len(sent) # max_len : 400
        for token in sent:
            word = imdb_idx2word[token]
            if word in joint_word2idx:
                new_sent.append(joint_word2idx[word])
            else:
                new_sent.append(joint_word2idx['<unk>'])
        for _ in range(len_pad):
            new_sent.append(joint_word2idx['<pad>'])
        new_x_test.append(np.array(new_sent))
    new_x_test = np.array(new_x_test)

    valid_ids   = np.random.choice(range(len(new_x_train)), args.batch_size*3, replace=False) 
    train_ids   = np.array(list(set(range(len(new_x_train))) - set(valid_ids)))
    new_x_train = torch.tensor(new_x_train).type('torch.LongTensor')
    new_x_test  = torch.tensor(new_x_test).type('torch.LongTensor')
    y_train     = torch.tensor(y_train)
    y_test      = torch.tensor(y_test)

    train = torch.utils.data.TensorDataset(new_x_train[train_ids], y_train[train_ids])
    valid = torch.utils.data.TensorDataset(new_x_train[valid_ids], y_train[valid_ids])
    test  = torch.utils.data.TensorDataset(new_x_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train, shuffle=True,  batch_size=args.batch_size, num_workers=5)
    valid_loader = torch.utils.data.DataLoader(valid, shuffle=False, batch_size=args.batch_size, num_workers=5)
    eval_loader  = torch.utils.data.DataLoader(test,  shuffle=False, batch_size=args.batch_size, num_workers=5)

    return train_loader, valid_loader, eval_loader, joint_embed, joint_embed_re

def to_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def run(args, glove_embedding, recon_embedding, train_loader, valid_loader, test_loader):

    print("| Training")

    set_seed(args)

    glove = recon_embedding if args.use_word2code else glove_embedding

    model       = to_cuda(Classifier(args, glove))
    optimizer   = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_ftn    = torch.nn.NLLLoss(reduction='sum')
    best_loss   = float('inf')
    train_steps = 0
    max_train_steps = args.max_epoch * len(train_loader)
    
    
    for cur_epoch in range(1, args.max_epoch):
        model.train()
        train_loss  = []
        for train_step, (data, label) in enumerate(train_loader):            
            model.zero_grad()
            data, label  = to_cuda(data), to_cuda(label)
            data, label  = Variable(data), Variable(label)
            logits       = model(data)
            loss         = loss_ftn(logits, label).div(len(data))
            train_loss.append(loss.data.clone().item())
            train_steps += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            
            print('| training progress [{}/{}]'.format(train_steps, max_train_steps), end='\r', flush=True)
                
        model.eval()
        valid_loss = []
        for valid_step, (data, label) in enumerate(valid_loader):
            model.zero_grad()
            data, label  = to_cuda(data), to_cuda(label)
            data, label  = Variable(data), Variable(label)
            logits       = model(data)
            loss         = loss_ftn(logits, label).div(len(data))
            valid_loss.append(loss.data.clone().item())
        
        train_loss = np.mean(train_loss) 
        valid_loss = np.mean(valid_loss) 
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), args.path_output + args.model_name + "_best_ckpt.pt")
            
        print('| [epoch{}] trian_loss : {:.2f} valid_loss : {:.2f} best_valid_loss : {:.2f}'.format(cur_epoch, train_loss, valid_loss, best_loss))
    
    print("| Testing")

    test_model      = to_cuda(Classifier(args, glove_embedding))
    best_checkpoint = torch.load(args.path_output + args.model_name + '_best_ckpt.pt')
    test_model.load_state_dict(best_checkpoint)

    correct  = 0
    num_test = 0 
    test_model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):            
        test_model.zero_grad()
        data, label = to_cuda(data), to_cuda(label)
        data, label = Variable(data), Variable(label)
        logits      = test_model(data)
        preds       = logits.data.max(1)[1]
        correct    += torch.sum(preds == label).item()
        num_test   += len(label)
    
    print("Test Accuracy: {:.3f}".format(correct/num_test))
            
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_word2code',   action='store_true', help='to use codes?')

    parser.add_argument('--batch_size',      type=int,   default=128,      help='batch size')
    parser.add_argument('--max_epoch',       type=int,   default=50,       help='number of epochs to train')
    parser.add_argument('--lr',              type=float, default=0.0001,   help='learning rate')
    parser.add_argument('--seed',            type=int,   default=1,        help='random seed')
    parser.add_argument('--hidden_size',     type=int,   default=150,      help='number of dimensions of lstm')
    
    parser.add_argument('--code_book_len',   type=int,   default=32,       help='number of codebooks')
    parser.add_argument('--cluster_num',     type=int,   default=16,       help='length of a codebook')

    parser.add_argument('--glove_file',       type=str,   default='glove.42B.300d.txt',       help='input data path')
    parser.add_argument('--imdb_file',        type=str,   default='../data/IMDB Dataset.csv', help='imdb data path')
    parser.add_argument('--path_glove',       type=str,   default='../data/',                 help='input data path')
    parser.add_argument('--path_output',      type=str,   default='../output/',               help='path output codes')
    parser.add_argument('--glove_model_name', type=str,   default='42B_FINAL',                help='model name')
    parser.add_argument('--model_name',       type=str,   default='baseline',                 help='model name')
    
    
    args = parser.parse_args()
    set_seed(args)
    train_laoder, valid_loader, test_loader, glove_embedding, recon_embedding = load(args)
    run(args, glove_embedding, recon_embedding, train_laoder, valid_loader, test_loader)

    


if __name__ == '__main__':
    main()


import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from glob import glob
from torchtext import data

import copy
import logging
import dataset
from models import attention as net


import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    


torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():

    parser = argparse.ArgumentParser(description="lstm_attention", add_help=False)

    # train path  
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--noise', type=bool, default=True,
                        help='{True,False}')
    parser.add_argument('--ntype', type=str, default='asymmetric',
                        help='{asymmetric,symmetric}')
    parser.add_argument('--nrate', type=str, default='0.2',
                        help='{0.05, 0.1, 0.2}')


    # train
    parser.add_argument('--epoch', type=int, default=6, help="Epoch")
    parser.add_argument('--batch_size', type=int,
                        default=64, help="Batch size")
    parser.add_argument('--dropout', type=float,
                        default=0.5, help="Use dropout")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--K_PGD', type=int, default=10, help="K_PGD")
    # model
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=200)

    parser_args, _ = parser.parse_known_args()
    target_parser = argparse.ArgumentParser(parents=[parser])
    args = target_parser.parse_args()
    return args

def clamp(x,eps):
    upper_limit = torch.ones_like(x).cuda()*eps
    lower_limit = torch.ones_like(x).cuda()*-eps
    return torch.max(torch.min(x, upper_limit), lower_limit)
    
def pgd_attacks(model, x, labels, eps=0.01, alpha=0.001, iters=20, rands=True):
    pgd_model = copy.deepcopy(model)
    pgd_model.train()
    emb = x.clone()
    criterion = nn.CrossEntropyLoss()
    if rands:
        delta = torch.Tensor(emb.shape).uniform_(-0.001,0.001).cuda()
    else:
        delta = torch.zeros_like(emb).cuda()
    delta.requires_grad = True
    for i in range(iters):
        output = pgd_model(emb + delta)
        loss = criterion(output, labels)
        grad = torch.autograd.grad(loss,emb)[0]
        delta.data = clamp(delta + alpha * torch.sign(grad),eps)
    delta = delta.detach()
    return delta

def compute_mean_logit(logit,label):
    logit = F.softmax(logit,dim=1)
    init_tensor = torch.tensor(0.0).cuda()
    y_onehot = F.one_hot(label, num_classes=2)
    p = torch.sum(torch.mul(logit,y_onehot),dim=1)
    mean_logit = torch.mean(p)
    tau = mean_logit+init_tensor
    return p,p-tau

def train(model_emb,model_lstm, iterator, optimizer, criterion,args):
    epoch_loss = 0
    epoch_acc = 0

    model_emb.train()
    model_lstm.train()

    for batch in iterator:
        batch.text = batch.text.permute(1, 0)
        emb = model_emb(batch.text) 
        pred = model_lstm(emb) 
        loss_forward = criterion(pred, batch.label)
        p,sub = compute_mean_logit(pred,batch.label)
        down_number_rate = 0.075
        down_number = round(batch.label.shape[0]*down_number_rate)
        up_sampleid = torch.topk(p, batch.label.shape[0]-down_number).indices
        down_sampleid = torch.topk(-p, down_number).indices    
        delta = pgd_attacks(model_lstm,emb, batch.label)
        delta[down_sampleid] = 0
        emb_add_delta = emb+delta
        pred = model_lstm(emb_add_delta) 
        y_onehot = F.one_hot(batch.label,num_classes=2)
        soft_logit = torch.tensor(F.softmax(pred[down_sampleid],dim=-1).detach().cpu().numpy()).cuda()
        pred[down_sampleid] = pred[down_sampleid] +0.075*(y_onehot[down_sampleid]-soft_logit)
        loss_adv = criterion(pred, batch.label)
        loss = 0.4*loss_forward + 0.6*loss_adv
        loss.backward()
       
        optimizer.step()
        optimizer.zero_grad()        

        acc = np.mean((torch.argmax(pred, 1) == batch.label).cpu().numpy())
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model_emb,model_lstm, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_loss_lst = []
    model_emb.eval()
    model_lstm.eval()

    with torch.no_grad():
        for batch in iterator:
            batch.text = batch.text.permute(1, 0)

            emb = model_emb(batch.text) 
            pred = model_lstm(emb) 

            loss = criterion(pred, batch.label)
            acc = np.mean((torch.argmax(pred, 1) == batch.label).cpu().numpy())

            epoch_loss_lst += [loss.item()]
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main(args):

    TEXT = data.Field(sequential=True, tokenize=dataset.text_token,
                      lower=True, fix_length=args.max_length)
    LABEL = data.Field(sequential=False, use_vocab=False)
    BIAS = data.Field(sequential=False, use_vocab=False)

    fields = [('label', LABEL), ('text', TEXT), ('bias', BIAS)]

    if args.noise:
        train_path = "dataset/" + args.dataset +"/train_noise_" + args.nrate + "_" + args.ntype + ".csv"
    else:
        train_path = "dataset/" + args.dataset +"/train_clean.csv" 
    valid_path = "dataset/" + args.dataset +"/dev_clean.csv" 
    test_path = "dataset/" + args.dataset +"/test_clean.csv" 
    train_iter, valid_iter,test_iter, embed_matrix = dataset.data_iter(train_path,
                                                                       valid_path,
                                                                       test_path,
                                                                       args.batch_size,
                                                                       device,
                                                                       fields,
                                                                       TEXT)

    logger.info("Model: LSTM with Attention")

    model_emb = net.EMBEDDING(embed_matrix, args)
    model_lstm = net.BILSTM( args)
    model_emb.to(device)
    model_lstm.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam([{'params': model_emb.parameters()},
                                {'params': model_lstm.parameters()}], lr=args.lr)
       
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9)


    logger.info("start traning...")
    #logger.info("\nTrue; lamda = 100; logit = normal; l2 ")
    best_acc = 0
    for epoch in range(args.epoch):
        train_loss, train_acc = train(model_emb,model_lstm, train_iter, optimizer, criterion,args)
        valid_loss, valid_acc = evaluate(model_emb,model_lstm, valid_iter, criterion)

        if best_acc < valid_acc:
            best_acc = valid_acc
            emb_pth = model_emb.state_dict()
            lstm_pth = model_lstm.state_dict()
            torch.save(emb_pth, "result/"+args.dataset+"emb_best.pth")
            torch.save(lstm_pth, "result/"+args.dataset+"lstm_best.pth")

        logger.info(
            f'Epoch: {epoch:02}, Train Acc: {train_acc * 100:.2f}%, Train Loss1: {train_loss:.2f},valid Acc: {valid_acc * 100:.2f}% , Best Acc: {best_acc * 100:.2f}%')
        scheduler.step(2)

    # load model
    model_emb = net.EMBEDDING(embed_matrix, args)
    

    test_model_emb = net.EMBEDDING(embed_matrix, args)
    test_model_lstm = net.BILSTM( args)
    test_model_emb.to(device)
    test_model_lstm.to(device)
    test_model_emb.load_state_dict(torch.load("result/"+args.dataset+"emb_best.pth"))
    test_model_lstm.load_state_dict(torch.load("result/"+args.dataset+"lstm_best.pth"))
    
    # test acc
    test_loss, test_acc = evaluate(test_model_emb,test_model_lstm, test_iter, criterion)
    logger.info(f'Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    args = parse_args()
    
    main(args)

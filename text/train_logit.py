
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from glob import glob
from torchtext import data

import logging
import dataset
from models import bilstm as net



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
    parser = argparse.ArgumentParser(
        description="lstm_attention", add_help=False)

    # train path
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--noise', type=bool, default=True,
                        help='{True,False}')
    parser.add_argument('--ntype', type=str, default='symmetric',
                        help='{asymmetric,symmetric}')
    parser.add_argument('--nrate', type=str, default='0.1',
                        help='{0.05, 0.1, 0.2}')

    parser.add_argument('--train_num', type=int,
                        default="20000")

    # train
    parser.add_argument('--epoch', type=int, default=6, help="Epoch")
    parser.add_argument('--batch_size', type=int,
                        default=64, help="Batch size")
    parser.add_argument('--dropout', type=float,
                        default=0.5, help="Use dropout")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")

    # model
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=200)

    parser_args, _ = parser.parse_known_args()
    target_parser = argparse.ArgumentParser(parents=[parser])
    args = target_parser.parse_args()
    return args

def my_loss(inputs,target):
    target = target.unsqueeze(-1)
    inp = torch.log(F.softmax(inputs))
    return torch.abs(torch.gather(inp,1,target))


def get_matrix(matirx,lst):
    for i in range(len(matirx)):
        if i in lst:
            matirx[i] = 1
    return matirx


def train(model, iterator, optimizer, criterion,epoch):
    #epoch_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:

        optimizer.zero_grad()
        batch.text = batch.text.permute(1, 0)

        pred, bais_l1 = model(batch.text, batch.bias, True)

        # ce loss
        loss1 = criterion(pred, batch.label)
        # l1 loss
        loss2 = 0.75*bais_l1
        loss = loss1 + loss2
        acc = np.mean((torch.argmax(pred, 1) == batch.label).cpu().numpy())

        loss.backward()
        optimizer.step()

        epoch_loss1 += loss1.item()
        epoch_loss2 += loss2.item()
        epoch_acc += acc.item()

    return epoch_loss1 / len(iterator), epoch_loss2 / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            batch.text = batch.text.permute(1, 0)
            pred, _ = model(batch.text, batch.bias, False)

            loss = criterion(pred, batch.label)
            acc = np.mean((torch.argmax(pred, 1) == batch.label).cpu().numpy())

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

        
    logger.info("Model: LSTM with Attension")

    model = net.BILSTM(embed_matrix, args, device)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)


    bias_params  = []
    other_params = []
    for pname, p in model.named_parameters():
        if pname == "bias.weight":
            bias_params += [p]   
        else:
            other_params += [p]

    optimizer = torch.optim.Adam([
                {'params': other_params}, 
                {'params': bias_params, 'lr': 400*args.lr}],
                lr=args.lr,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9)

    logger.info("start traning...")

    best_acc = 0
    
    for epoch in range(args.epoch):

        train_loss1, train_loss2, train_acc = train(model, train_iter, optimizer, criterion,epoch)
      
        bias_l1 = torch.sum(torch.abs(model.bias.weight),axis=1)
        
        valid_loss, valid_acc = evaluate(model, valid_iter, criterion)

        if best_acc < valid_acc:
            best_acc = valid_acc
            pth = model.state_dict()
            torch.save(pth, "result/"+args.dataset+"_best.pth")

        logger.info(
            f'Epoch: {epoch:02}, Train Acc: {train_acc * 100:.2f}%, Train Loss1: {train_loss1:.3f}, Train Loss2: {train_loss2:.3f}, valid Acc: {valid_acc * 100:.2f}% , Best Acc: {best_acc * 100:.2f}%')
        scheduler.step(2)

    # load model
    test_model = net.BILSTM(embed_matrix, args, device)
    test_model.to(device)
    test_model.load_state_dict(torch.load("result/"+args.dataset+"_best.pth"))

    # test acc
    test_loss, test_acc = evaluate(test_model, test_iter, criterion)
    logger.info(f'Test Acc: {test_acc:.4f}')


if __name__ == '__main__':        
    args = parse_args()
    print(args)
    main(args)

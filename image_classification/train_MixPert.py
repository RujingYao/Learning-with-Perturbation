import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import resnet
from dataset import train_loader,val_loader
from attack import pgd
import torch.nn.functional as F

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--class_num', default=10, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0


def main():
    
    starting_time = time.time()
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.OriResnet(args))
    #model = resnet.OriResnet(args)
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150, 225], last_epoch=args.start_epoch - 1)



    TrainAttack = pgd.IPGD(eps = 8/255.0, sigma = 2/255.0, nb_iter = 5, norm = np.inf,
                              mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std = torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))

    EvalAttack = pgd.IPGD(eps = 8/255.0, sigma = 2/255.0, nb_iter = 20, norm = np.inf,
                              mean=torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std=torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))
    
    if args.evaluate:
        validate(val_loader, model, criterion,EvalAttack)
        return

    best_cleanacc = 0
    best_cleanacc_epoch = 0
    best_cleanacc_epoch_vs_advacc=0

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch,TrainAttack)
        lr_scheduler.step()

        # evaluate on validation set

        prec1,clean_accuracy_val, adv_accuracy_val = validate(val_loader, model, criterion,EvalAttack)
    
        if best_cleanacc < clean_accuracy_val:
            best_cleanacc = clean_accuracy_val
            best_cleanacc_epoch = epoch
            best_cleanacc_epoch_vs_advacc = adv_accuracy_val

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print("best acc = "+str(best_prec1))

        print("best clean acc = "+str(best_cleanacc))
        print("best_cleanacc_epoch = "+str(best_cleanacc_epoch))
        print("best_cleanacc_epoch_vs_advacc = "+str(best_cleanacc_epoch_vs_advacc))

    ending_time = time.time()
    print("total cost time :", starting_time-ending_time)


def sign(pred,target):
    y_onehot = F.one_hot(target, num_classes=pred.shape[1])
    p = torch.sum(torch.mul(F.softmax(pred,dim=-1),y_onehot),dim=1)
    down_number_rate = 0.15 
    down_number = round(target.shape[0]*down_number_rate)
    up_sampleid = torch.topk(p, target.shape[0]-down_number).indices
    up_or_down = torch.tensor([0]*target.shape[0]).cuda()
    up_or_down[up_sampleid] = 1   
    up_or_down_trans = up_or_down.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return up_or_down_trans,down_number
    
def compute_grad(logit,label,num_classes):
    logit_softmax = F.softmax(logit,dim=-1)
    label_onehot = F.one_hot(label, num_classes=num_classes)
    grad = label_onehot-logit_softmax
    grad_new = torch.tensor(grad.detach().cpu().numpy()).cuda()
    return grad_new

def train(train_loader, model, criterion, optimizer, epoch,AttackMethod):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    

    end = time.time()
    for i, (input, target,_) in enumerate(train_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target


        with torch.no_grad():
            pred = model(input_var)
            up_or_down_trans,down_number = sign(pred,target)
            up_or_down_trans =up_or_down_trans.detach().cpu().numpy()


        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(model, input, target,up_or_down_trans=up_or_down_trans)
            optimizer.zero_grad()

        model.train()

        # measure data loading time
        data_time.update(time.time() - end)

        output_clean =  model(input_var)
        loss_clean = criterion(output_clean, target_var)
        # compute output
        output_adv = model(adv_inp.to(torch.float32))


        # add pert
        eta = 3
        grad_new = eta*compute_grad(output_adv,target_var,args.class_num) 
        mask = torch.from_numpy(1-up_or_down_trans).squeeze(-1).squeeze(-1).cuda()
        with torch.no_grad():
            mask_TF = torch.where(torch.tensor(up_or_down_trans.squeeze(-1).squeeze(-1).squeeze(-1))==0,True,False)
        grad_new = grad_new*mask
        output = output_adv + grad_new        
        
        loss_adv_cleandata = criterion(output_clean[~mask_TF],target_var[~mask_TF])
        loss_adv_advdata   = criterion(output[~mask_TF],target_var[~mask_TF])
        

        loss_adv_sumloss = (0.4*loss_adv_cleandata+0.6*loss_adv_advdata)*(target_var.shape[0]-down_number)

        loss_pert_sumloss = criterion(output[mask_TF],target_var[mask_TF])*down_number

        loss = (loss_adv_sumloss + loss_pert_sumloss)/target_var.shape[0]



        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, EvalAttack):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    clean_accuracy = AverageMeter()
    adv_accuracy = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    #with torch.no_grad():
    for i, (input, target,_) in enumerate(val_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()

        if args.half:
            input_var = input_var.half()

        with torch.no_grad():
            # compute output
            output = model(input_var)
            #clean data accuracy
            prec_cleanaccuracy = accuracy(output.data, target)[0]
            clean_accuracy.update(prec_cleanaccuracy.item(), input.size(0))

        if EvalAttack is not None:
            adv_inp = EvalAttack.attack(model, input, target)
            
            with torch.no_grad():
                output = model(adv_inp)
                # adv data accuracy
                prec_advaccuracy = accuracy(output.data, target)[0]
                adv_accuracy.update(prec_advaccuracy.item(), input.size(0))

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'clean_accuracy {clean_accuracy.val:.4f} ({clean_accuracy.avg:.4f})\t'
                    'adv_accuracy {adv_accuracy.val:.4f} ({adv_accuracy.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,clean_accuracy=clean_accuracy,adv_accuracy=adv_accuracy,
                        top1=top1))

    print(' * Prec@1 {top1.avg:.4f},* clean_accuracy {clean_accuracy.avg:.4f},* adv_accuracy {adv_accuracy.avg:.4f}'
          .format(top1=top1,clean_accuracy=clean_accuracy,adv_accuracy=adv_accuracy))

    return top1.avg, clean_accuracy.avg, adv_accuracy.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
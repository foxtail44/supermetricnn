import torch
import torch.nn as nn
from torch.autograd import Variable
from model import CNNSimple
import dataloader
import time
import os
import shutil

path_train = "./dataset/train/"
path_validate = "./dataset/validate/"

lr = 0.0001
bs = 10
epochs = 1000000

print_freq = 100

resume = None
start_epoch = 0

trainset = dataloader.MetricDataLoader(path_train, 0.25)
valset = dataloader.MetricDataLoader(path_validate, 0.25)


train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs)
val_loader = torch.utils.data.DataLoader(valset, batch_size=bs)


model = CNNSimple(10)

optimizer = torch.optim.SGD(model.parameters(), lr)

criterion = nn.CrossEntropyLoss()


def main():
    # Add code to resume from checkpoint here
    
    resume = None
    start_epoch = 0
    best_prec1 = 0

    # Load checkpoint if flag up
    if resume:
        print("**Using pretrained model**")
        if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("**Loaded checkpoint**")
        else:
            print("**No checkpoint found**")

    for epoch in range(start_epoch, epochs):
            adjust_learning_rate(lr, optimizer, epoch)
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'CNN1',
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        model.train()

        # measure data loading time
        data_time.update(time.time() - end)

        # Set things up
        input_var = Variable(input)
        target_var = Variable(target)

        output = model(input_var)

        loss = \
            criterion(output, target_var)

        prec1, prec5 = accuracy(output.data.cpu(), target_var.data.cpu(), topk=(1, 5))
        losses.update(loss.data[0], input_var.data.size(0))
        top1.update(prec1[0], input_var.data.size(0))
        top5.update(prec5[0], input_var.data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'BTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        model.train()

        # measure data loading time
        data_time.update(time.time() - end)

        # Set things up
        input_var = Variable(input)
        target_var = Variable(target)

        output = model(input_var)

        loss = \
            criterion(output, target_var)

        prec1, prec5 = accuracy(output.data.cpu(), target_var.data.cpu(), topk=(1, 5))
        losses.update(loss.data[0], input_var.data.size(0))
        top1.update(prec1[0], input_var.data.size(0))
        top5.update(prec5[0], input_var.data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'BTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))
    return top1.avg

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


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    main()

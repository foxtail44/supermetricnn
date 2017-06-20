import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import dataloader
import time

path = "./dataset/"
lr = 0.1
bs = 10

trainset = dataloader.MetricDataLoader(path, 0.25)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs)
print trainset[0]
model = model.CNNSimple(10, 20, 3)

optimizer = torch.optim.SGD(model.parameters(), lr)

criterion = nn.CrossEntropyLoss()

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'BTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))

train(train_loader, model, criterion, optimizer, 0)

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

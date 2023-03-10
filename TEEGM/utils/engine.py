import torch
from torch import nn
from timm.utils import AverageMeter
import matplotlib.pyplot as plt
import time
import copy
from .tools import get_grad_norm



class TrainPainter():
    def __init__(self):
        self.train_acc_his, self.train_loss_his = [], []
        self.val_acc_his, self.val_loss_his = [], []

    def update(self, train_loss, train_acc, val_loss, val_acc):
        self.train_loss_his.append(train_loss)
        self.train_acc_his.append(train_acc)
        self.val_loss_his.append(val_loss)
        self.val_acc_his.append(val_acc)

    def plot(self):
        plt.close()
        plt.subplot(121)
        plt.plot(self.train_acc_his, label='train acc')
        plt.plot(self.val_acc_his, label='test acc')
        plt.title("accuracy")
        plt.xlabel('epoch')
        plt.legend()
        plt.subplot(122)
        plt.plot(self.train_loss_his, label='train loss')
        plt.plot(self.val_loss_his, label='test loss')
        plt.title("loss")
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('training_curves.png')


def train(model: nn.Module, train_loader, val_loader, criterion, optimizer, lr_scheduler,
          epoch=1000, rta=None, plot=False, logger=None):

    best_train, best_val, best_val_loss, best_train_loss = 0, 0, float('inf'), float('inf')
    best_model = None
    best_optimizer = None
    if plot:
        painter = TrainPainter()

    print('------start training------')
    for i in range(epoch):
        start=time.time()

        train_loss, train_acc, grad_norm = train_one_epoch(model, train_loader, criterion, optimizer, lr_scheduler, i, rta)
        val_loss, val_acc = validate(model, val_loader,)

        print(f'Train:{i+1}|{epoch}\t'
              f'train loss {train_loss:.4f}\t'
              f'train acc {train_acc:.4f}\t'
              f'val loss {val_loss:.4f}\t'
              f'val acc {val_acc:.4f}\t'
              f'grad norm {grad_norm:.2f}')

        if plot:
            painter.update(train_loss, train_acc, val_loss, val_acc)
        if logger:
            logger.add_scalars('accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, global_step=i+1)
            logger.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, global_step=i+1)

        if val_loss < best_val_loss * 0.99:
            best_train, best_val, best_val_loss, best_train_loss = train_acc, val_acc, val_loss, train_loss
            best_model = copy.deepcopy(model)
            best_optimizer = copy.deepcopy(optimizer.state_dict())

        print(f'Epoch time: {time.time()-start:.2f} s')

    print("best_train_acc:{}  best_val_acc:{}".format(best_train, best_val))
    print('------------------------------')
    if plot:
        painter.plot()
    if logger:
        logger.close()
    return best_model, best_optimizer, best_train_loss, best_val_loss


def retrain(best_train_loss, model: nn.Module, trainval_loader, val_loader, criterion, optimizer, lr_scheduler,
          epoch=500, rta=None):

    print('------start retraining------')
    for i in range(epoch):
        start = time.time()

        train_loss, train_acc, grad_norm = train_one_epoch(model, trainval_loader, criterion, optimizer, lr_scheduler, i, rta)
        val_loss, val_acc = validate(model, val_loader,)

        print(f'Train:{i+1}|{epoch}\t'
              f'loss {train_loss:.4f}\t'
              f'acc {train_acc:.4f}\t'
              f'val loss {val_loss:.4f}\t'
              f'val acc {val_acc:.4f}\t'
              f'grad norm {grad_norm:.2f}')
        print(f'epoch time: {time.time() - start:.2f} s')

        if val_loss <= best_train_loss:
            best_model = copy.deepcopy(model)
            break
    else:
        best_model = copy.deepcopy(model)
    return best_model


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, rta=None):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    grad_meter = AverageMeter()

    optimizer.zero_grad()

    for idx, (data, label) in enumerate(train_loader):
        data = data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        if rta:
            data, label = rta(data, label)
        N, C, T = data.shape

        out = model(data)
        loss = criterion(out, label)
        loss.backward()
        # grad_norm = get_grad_norm(model.parameters())
        # grad_meter.update(grad_norm, label.shape[0])
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
        torch.cuda.synchronize()

        acc = (torch.sum(torch.argmax(out.detach(), 1) == label)/label.shape[0]).cpu()
        acc_meter.update(acc.item(), label.shape[0])
        loss_meter.update(loss.item(), label.shape[0])
    return loss_meter.avg, acc_meter.avg, grad_meter.avg


@torch.no_grad()
def validate(model, val_loader, criterion = nn.CrossEntropyLoss(reduction='mean')):
    model.eval()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    for data, label in val_loader:
        N, C, T = data.shape
        data = data.reshape(N, C, T)
        data = data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        out = model(data)
        loss = criterion(out, label)

        acc = (torch.sum(torch.argmax(out.detach(), 1) == label)/label.shape[0]).cpu()
        acc_meter.update(acc.item(), label.shape[0])
        loss_meter.update(loss.item(), label.shape[0])
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    corr = 0
    for data, labels in test_loader:
        pred = torch.argmax(model(data).detach(), dim=-1)
        corr += torch.sum(pred == labels).cpu().item()

    return corr / len(test_loader.dataset)
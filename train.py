import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import attention_net, list_loss, ranking_loss
from dataset import lmdbDataset
from logger import logger

cudnn.benchmark = True
log = logger('checkpoint')
steps = 0


class Params:
    # arch = 'inception_v3'
    num_classes = 1010
    workers = 6
    epochs = 100
    start_epoch = 0
    batch_size = 8  # might want to make smaller
    lr = 0.0045
    lr_decay = 0.94
    epoch_decay = 4
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 100
    proposal_num = 6
    input_size = (448, 448)
    cuda = torch.cuda.is_available()

    resume = ''  # set this to path of model to resume training

    # set evaluate to True to run the test set
    evaluate = False
    op_file_name = 'checkpoint/inat2018_test_preds.csv'  # submission file
    train_file = 'dataset/traindata'
    val_file = 'dataset/valdata'
    test_file = 'dataset/testdata'


def main():
    global args, steps
    args = Params()
    best_pred = 0

    model = attention_net(topN=args.proposal_num, classes=args.num_classes, input_size=args.input_size)
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()

        # define loss function (criterion) and optimizer
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = MultiStepLR(optimizer, milestones=[60, 100], gamma=.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_pred = checkpoint['best_pred']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # data loading code
    train_dataset = lmdbDataset(args.train_file, args.input_size)
    val_dataset = lmdbDataset(args.val_file, args.input_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        accuracy, preds, im_ids = validate(val_loader, model, criterion)
        # write predictions to file
        with open(args.op_file_name, 'w') as opfile:
            opfile.write('id,predicted\n')
            for ii in range(len(im_ids)):
                opfile.write(str(im_ids[ii]) + ',' + ' '.join(str(x) for x in preds[ii, :]) + '\n')
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        steps = epoch * len(train_loader)
        print('Begin epoch %d' % epoch)
        train(train_loader, model, criterion, optimizer)

        # evaluate on validation set
        print('Begin validating')
        pred, _, _ = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = pred > best_pred
        best_pred = max(pred, best_pred)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_pred': best_pred,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer):
    global steps, args
    # switch to train mode
    model.train()

    start = time.time()
    for i, (_, input_var, target_var) in enumerate(train_loader):
        # measure data loading time
        batch_size = input_var.size(0)
        steps += 1
        if args.cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        raw_logits, concat_logits, part_logits, _, top_n_prob = model(input_var)
        part_loss = list_loss(part_logits.view(batch_size * args.proposal_num, -1),
                                    target_var.unsqueeze(1).repeat(1, args.proposal_num).view(-1)) \
            .view(batch_size, args.proposal_num)
        raw_loss = criterion(raw_logits, target_var)
        concat_loss = criterion(concat_logits, target_var)
        rank_loss = ranking_loss(top_n_prob, part_loss)
        partcls_loss = criterion(part_logits.view(batch_size * args.proposal_num, -1),
                                 target_var.unsqueeze(1).repeat(1, args.proposal_num).view(-1))

        total_loss = raw_loss + concat_loss + rank_loss + partcls_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            log.scalar_summary('train/total_loss', total_loss.item(), steps)
            log.scalar_summary('train/speed', i / (time.time() - start), steps)


def validate(val_loader, model, criterion):
    global steps
    # switch to evaluate mode
    model.eval()

    correct, pred, im_ids, loss_avg = 0, [], [], 0
    for i, (im_id, input_var, target_var) in enumerate(val_loader):
        if args.cuda:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        _, concat_logits, _, _, _ = model(input_var)
        concat_loss = criterion(concat_logits, target_var)
        loss_avg += concat_loss.item()

        _, concat_predict = torch.max(concat_logits, 1)
        correct += torch.sum(concat_predict.data == target_var.data)
        pred.append(concat_predict.unsqueeze(1).cpu().numpy().astype(np.int))
        im_ids.append(im_id.cpu().numpy().astype(np.int))

    log.scalar_summary('val/concat_loss', loss_avg / len(val_loader), steps)
    log.scalar_summary('val/accuracy', correct / (len(val_loader) * args.batch_size), steps)

    return correct / (len(val_loader) * args.batch_size), np.vstack(pred), np.hstack(im_ids)


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, 'model_best.pth')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

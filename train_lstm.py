#!/usr/bin/env python3
# Andreas Goulas <goulasand@gmail.com> | first creation
# Nikolaos Gkalelis <gkalelis@iti.gr> | 23/4/2021 | minor changes (main(), print, etc.)

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, accuracy_score
import os
import sys

from fcvid import FCVID
from ylimed import YLIMED
from model import Classifier

def train(model, loader, crit, opt, sched, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(loader):
        feats, label = batch
        feats = feats.to(device)
        label = label.to(device)

        opt.zero_grad()
        out_data = model(feats)
        loss = crit(out_data, label)
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    sched.step()
    return epoch_loss / len(loader)

def test(model, loader, scores, device):
    gidx = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats, _ = batch
            feats = feats.to(device)
            out_data = model(feats)

            N = out_data.shape[0]
            scores[gidx:gidx+N, :] = out_data.cpu()
            gidx += N

parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('--dataset', default='fcvid', choices=['fcvid', 'ylimed'])
parser.add_argument('--feats_folder', default='feats', help='directory to load features')
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--gamma', type=float, default=1, help='learning rate decay rate')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loader; set always to zero!')
parser.add_argument('--eval_interval', type=int, default=1, help='interval for evaluating models (epochs)')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()

def main():

    train_feats = torch.load(os.path.join(args.feats_folder, 'feats-train.pt'))
    train_truth = torch.load(os.path.join(args.feats_folder, 'truth-train.pt'))
    test_feats = torch.load(os.path.join(args.feats_folder, 'feats-test.pt'))
    test_truth = torch.load(os.path.join(args.feats_folder, 'truth-test.pt'))

    if args.dataset == 'ylimed':
        train_truth = train_truth.long()
        test_truth = test_truth.long()

    train_dataset = TensorDataset(train_feats, train_truth)
    test_dataset = TensorDataset(test_feats, test_truth)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    test_truth = test_truth.numpy()

    device = torch.device('cuda:0')
    if args.verbose:
        print('running on %s' % device)
        print('num train samples=%d' % len(train_dataset))
        print('num test samples=%d' % len(test_dataset))

    if args.dataset == 'fcvid':
        crit = nn.BCEWithLogitsLoss()
        num_feats, num_class = FCVID.NUM_FEATS, FCVID.NUM_CLASS
    elif args.dataset == 'ylimed':
        crit = nn.CrossEntropyLoss()
        num_feats, num_class = YLIMED.NUM_FEATS, YLIMED.NUM_CLASS
    else:
        sys.exit("Unknown dataset!")

    start_epoch = 0
    model = Classifier(2 * num_feats, num_feats, num_class).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.ExponentialLR(opt, args.gamma)

    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.perf_counter()
        loss = train(model, train_loader, crit, opt, sched, device)
        t1 = time.perf_counter()

        sched.step()
        if args.verbose:
            print('[epoch %d] loss=%f dt=%.2fsec' % (epoch + 1, loss, t1 - t0))

        if (epoch + 1) % args.eval_interval == 0:
            num_test = len(test_dataset)
            scores = torch.zeros((num_test, num_class), dtype=torch.float32)
            test(model, test_loader, scores, device)
            scores = scores.numpy()

            if args.dataset == 'fcvid':
                ap = average_precision_score(test_truth, scores)
                print('mAP=%.2f%%' % (100 * ap))
            elif args.dataset == 'ylimed':
                pred = scores.argmax(axis=1)
                acc = accuracy_score(test_truth, pred)
                print('accuracy=%.2f%%' % (100 * acc))

                torch.save(pred, os.path.join(args.feats_folder, 'pred-test.pt'))

if __name__ == '__main__':
    main()

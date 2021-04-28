#!/usr/bin/env python3
# Andreas Goulas <goulasand@gmail.com> | first creation
# Nikolaos Gkalelis <gkalelis@iti.gr> | 23/4/2021 | minor changes (main(), print, etc.)

import argparse
import time
import os
import torch
from torch.utils.data import DataLoader

from fcvid import FCVID
from ylimed import YLIMED
from model import GraphModule

parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('model', nargs=1, help='trained model')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='fcvid', choices=['fcvid', 'ylimed'])
parser.add_argument('--dataset_root', default=r'D:\Users\gkalelis\PycharmProjects\FCVID', help='dataset root directory')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loader')
parser.add_argument('--save_folder', default='feats', help='directory to save features')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()


def main():
    if args.dataset == 'fcvid':
        train_dataset = FCVID(args.dataset_root, is_train=True)
        test_dataset = FCVID(args.dataset_root, is_train=False)
    elif args.dataset == 'ylimed':
        train_dataset = YLIMED(args.dataset_root, is_train=True)
        test_dataset = YLIMED(args.dataset_root, is_train=False)

    device = torch.device('cuda:0')
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.verbose:
        print("running on {}".format(device))
        print("num train samples={}".format(len(train_dataset)))
        print("num test samples={}".format(len(test_dataset)))
        print("missing videos={}".format(train_dataset.num_missing + test_dataset.num_missing))

    out_dim = 2 * train_dataset.NUM_FEATS
    model = GraphModule(args.gcn_layers, train_dataset.NUM_FEATS).to(device)
    data = torch.load(args.model[0])
    model.load_state_dict(data)

    dataset_list = [('train', train_dataset), ('test', test_dataset)]
    for phase, dataset in dataset_list:
        num_samples = len(dataset)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        out_feats = torch.zeros((num_samples, dataset.NUM_FRAMES, out_dim), dtype=torch.float32)

        t0 = time.perf_counter()
        gidx = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                feats, feat_global, _, _ = batch
                feats = feats.to(device)

                N, FR, B, NF = feats.shape
                feats = feats.view(N * FR, B, NF)
                out_data = model(feats, device).cpu()
                out_data = out_data.view(N, FR, -1)
                out_feat = torch.cat([out_data, feat_global], dim=-1)

                out_feats[gidx:gidx+N, :, :] = out_feat
                gidx += N

        t1 = time.perf_counter()

        truth = torch.from_numpy(dataset.labels)
        torch.save(out_feats, os.path.join(args.save_folder, 'feats-' + phase + '.pt'))
        torch.save(truth, os.path.join(args.save_folder, 'truth-' + phase + '.pt'))

        if args.verbose:
            print('phase {} dt={:.2f}sec'.format(phase, t1 - t0))


if __name__ == '__main__':
    main()

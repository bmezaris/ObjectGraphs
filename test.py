#!/usr/bin/env python3
# Andreas Goulas <goulasand@gmail.com> | first creation
# Nikolaos Gkalelis <gkalelis@iti.gr> | 23/4/2021 | minor changes (text processing, etc.)

import argparse
import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, accuracy_score

from fcvid import FCVID
from ylimed import YLIMED
from model import Model

parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('model', nargs=1, help='trained model')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='fcvid', choices=['fcvid', 'ylimed'])
parser.add_argument('--dataset_root', default=r'D:\Users\gkalelis\PycharmProjects\FCVID', help='dataset root directory')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--save_scores', action='store_true', help='save the output scores')
parser.add_argument('--save_path', default='scores.txt', help='output path')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()

def test(model, dataset, loader, scores, out_file, device):
    gidx = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats, feat_global, _, _ = batch
            feats = feats.to(device)
            feat_global = feat_global.to(device)
            out_data = model(feats, feat_global, device)

            N = out_data.shape[0]
            if out_file:
                for j in range(N):
                    video_name = dataset.videos[gidx + j]
                    #out_file.write('%s ' % video_name)
                    out_file.write("{}".format(video_name))
                    out_file.write(' '.join([str(x.item()) for x in out_data[j, :]]))
                    out_file.write('\n')
                
            scores[gidx:gidx+N, :] = out_data.cpu()
            gidx += N

def main():
    if args.dataset == 'fcvid':
        dataset = FCVID(args.dataset_root, is_train=False)
    elif args.dataset == 'ylimed':
        dataset = YLIMED(args.dataset_root, is_train=False)

    device = torch.device('cuda:0')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.verbose:
        print("running on {}".format(device))
        print("num samples={}".format(len(dataset)))
        print("missing videos={}".format(dataset.num_missing))

    model = Model(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS).to(device)
    data = torch.load(args.model[0])
    model.load_state_dict(data['model_state_dict'])
    #model.load_state_dict(data)

    out_file = None
    if args.save_scores:
        out_file = open(args.save_path, 'w')

    num_test = len(dataset)
    scores = torch.zeros((num_test, dataset.NUM_CLASS), dtype=torch.float32)

    t0 = time.perf_counter()
    test(model, dataset, loader, scores, out_file, device)
    t1 = time.perf_counter()

    scores = scores.numpy()
    if args.save_scores:
        out_file.close()

    if args.dataset == 'fcvid':
        ap = average_precision_score(dataset.labels, scores)
        print('mAP={:.2f}% dt={:.2f}sec'.format(100 * ap, t1 - t0))
    elif args.dataset == 'ylimed':
        pred = scores.argmax(axis=1)
        acc = accuracy_score(dataset.labels, pred)
        print('accuracy={:.2f}% dt={:.2f}sec'.format(100 * acc, t1 - t0))

if __name__ == '__main__':
    main()

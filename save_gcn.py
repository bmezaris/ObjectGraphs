#!/usr/bin/env python3
# Andreas Goulas <goulasand@gmail.com>
# Nikolaos Gkalelis <gkalelis@iti.gr> | 23/4/2021 | minor changes (main(), etc.)

import argparse
import torch

from fcvid import FCVID
from ylimed import YLIMED
from model import Model

parser = argparse.ArgumentParser(description='GCN Video Classification')
parser.add_argument('in_model', nargs=1, help='trained model')
parser.add_argument('out_model', nargs=1, help='gcn model path')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='fcvid', choices=['fcvid', 'ylimed'])
args = parser.parse_args()

def main():
    if args.dataset == 'fcvid':
        num_feats, num_class = FCVID.NUM_FEATS, FCVID.NUM_CLASS
    elif args.dataset == 'ylimed':
        num_feats, num_class = YLIMED.NUM_FEATS, YLIMED.NUM_CLASS

    model = Model(args.gcn_layers, num_feats, num_class)
    data = torch.load(args.in_model[0])
    model.load_state_dict(data['model_state_dict'])
    torch.save(model.graph.state_dict(), args.out_model[0])


if __name__ == '__main__':
    main()

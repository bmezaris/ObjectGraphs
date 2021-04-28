# Andreas Goulas <goulasand@gmail.com>
# Nikolaos Gkalelis <gkalelis@iti.gr> | 23/4/2021 | minor changes (add object label/class information, text processing, etc.)

import os
import numpy as np
from torch.utils.data import Dataset

class FCVID(Dataset):
    NUM_BOXES = 50
    NUM_FEATS = 2048
    NUM_CLASS = 239
    NUM_FRAMES = 9
    
    def __init__(self, root_dir, is_train):
        self.root_dir = root_dir 
        self.phase = 'train' if is_train else 'test'
        
        split_path = os.path.join(root_dir, 'materials', 'FCVID_VideoName_TrainTestSplit.txt')
        data_split = np.genfromtxt(split_path, dtype='str')

        label_path = os.path.join(root_dir, 'materials', 'FCVID_Label.txt')
        labels = np.genfromtxt(label_path, dtype=np.float32)
        
        self.num_missing = 0
        mask = np.zeros(data_split.shape[0], dtype=bool)
        for i, row in enumerate(data_split):
            if row[1] == self.phase:
                base, _ = os.path.splitext(os.path.normpath(row[0]))
                feats_path = os.path.join(root_dir, 'R152', base +'.npy')
                if os.path.exists(feats_path):
                    mask[i] = 1
                else:
                    self.num_missing += 1

        self.labels = labels[mask, :]
        self.videos = data_split[mask, 0]
        
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, 'R152', name +'.npy')
        global_path = os.path.join(self.root_dir, 'R152_global', name +'.npy')
        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = self.labels[idx, :]

        return (feats, feat_global, label, name)


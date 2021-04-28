# Andreas Goulas <goulasand@gmail.com>
# Nikolaos Gkalelis <gkalelis@iti.gr> | 23/4/2021 | minor changes (add object label/class information, text processing, etc.)

import os
import numpy as np
from torch.utils.data import Dataset

class YLIMED(Dataset):
    NUM_BOXES = 50
    NUM_FEATS = 2048
    NUM_CLASS = 10
    NUM_FRAMES = 9
    
    def __init__(self, root_dir, is_train):
        self.root_dir = root_dir 
        self.phase = 'Training' if is_train else 'Test'

        split_path = os.path.join(root_dir, 'YLI-MED_Corpus_v.1.4.txt')
        data_split = np.genfromtxt(split_path, dtype='str', skip_header=1)
        
        self.num_missing = 0
        mask = np.zeros(data_split.shape[0], dtype=bool)
        for i, row in enumerate(data_split):
            if row[7] == 'Ev100':
                continue

            if row[13] == self.phase:
                feats_path = os.path.join(root_dir, 'R152', row[0] + '.npy')
                if os.path.exists(feats_path):
                    mask[i] = 1
                else:
                    self.num_missing += 1

        self.videos = data_split[mask, 0]
        labels = [int(x[3:])-1 for x in data_split[mask, 7]]
        self.labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        name = self.videos[idx]
        name, _ = os.path.splitext(name)

        feats_path = os.path.join(self.root_dir, 'R152', name + '.npy')
        global_path = os.path.join(self.root_dir, 'R152_global', name + '.npy')

        feats = np.load(feats_path)
        feat_global = np.load(global_path)
        label = np.int64(self.labels[idx])

        return (feats, feat_global, label, name)


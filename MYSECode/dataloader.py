import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import soundfile

class WavDataset(Dataset):
    def __init__(self, addr):
        self.addr = addr # noisy addr
        self.clean_addr = addr.replace('noisy', 'clean') # clean addr
        self.noisy_dirs = []
        self.clean_dirs = []
        # NO Exceptations detection
        for dir in os.listdir(self.addr):
            self.noisy_dirs.append(os.path.join(self.addr, dir))
        for dir in os.listdir(self.clean_addr):
            self.clean_dirs.append(os.path.join(self.clean_addr, dir))

    def __len__(self):
        return len(self.noisy_dirs)
    
    def __getitem__(self, index):
        noisy_dir = self.noisy_dirs[index]
        clean_dir = self.clean_dirs[index]
        inputs, fs = soundfile.read(noisy_dir)
        targets, fs = soundfile.read(clean_dir)
        # preprocess inputs and targets to 16sec
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)
        
        return inputs, targets


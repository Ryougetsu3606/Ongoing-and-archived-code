import torch
import torch.nn as nn
import os
import sys
from show import show_params, show_model
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT 
from asteroid.models import BaseModel
import random
import numpy as np
from complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm
from dataloader import WavDataset
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from pesq import pesq
from pystoi import stoi
import pysepm
import tqdm
def cal_stoi_batch(clean_wavs, dirty_wavs, FS=16000):
    stoi_score = Parallel(n_jobs=-1)(delayed(stoi)(c, n, FS, extended=False) for c, n in zip(clean_wavs, dirty_wavs))
    stoi_score = np.array(stoi_score)
    return np.mean(stoi_score)

def cal_pesq(clean_wav, dirty_wav, FS=16000):
    try:
        pesq_score = pesq(FS, clean_wav, dirty_wav, "wb")
    except:
        print(' No utterances error')
        pesq_score = -1
    return pesq_score

def cal_pesq_batch(clean_wavs, dirty_wavs, FS=16000):
    pesq_score = Parallel(n_jobs=-1)(delayed(cal_pesq)(c, n, FS=FS) for c, n in zip(clean_wavs, dirty_wavs))
    pesq_score = np.array(pesq_score)
    return np.mean(pesq_score)

model = BaseModel.from_pretrained("dccrnp.bin")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = model.to(DEVICE)
print(model)
# print(model)
test_set = WavDataset('./1/noisy_testset_wav')
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
model.eval()
cln_all = []
enh_all = []
noi_all = []
with torch.no_grad():
    for inputs, targets in tqdm.tqdm(test_loader):

        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        outputs = model(inputs)
        outputs=outputs.squeeze(1)
        #print(f"input_shape:{inputs.shape}, output_shape:{outputs.shape}")
        
        noisy_wavs = inputs.cpu().detach().numpy()[:,:outputs.size(1)]
        clean_wavs = targets.cpu().detach().numpy()[:,:outputs.size(1)]
        enhanced_wavs = outputs.cpu().detach().numpy()
        cln_all.extend(clean_wavs)
        enh_all.extend(enhanced_wavs)
        noi_all.extend(noisy_wavs)
        del inputs
        del targets
        del outputs

avg_stoi = cal_stoi_batch(cln_all, enh_all)
avg_stoi_noi = cal_stoi_batch(cln_all, noi_all)
print(f"noise:{avg_stoi_noi}, enhance:{avg_stoi}")

avg_pesq = cal_pesq_batch(cln_all, enh_all)
avg_pesq_noi = cal_pesq_batch(cln_all, noi_all)
print(f"noise:{avg_pesq_noi}, enhance:{avg_pesq}")

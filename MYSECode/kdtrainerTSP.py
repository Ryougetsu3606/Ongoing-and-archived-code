import torch
import torch.nn as nn
import time
import random
import numpy as np
from dataloader import WavDataset
from conv_stft import ConvSTFT
from torch.utils.data import DataLoader
import tqdm
import copy
import argparse
import os
import itertools
from dc_crn import DCCRN
from muon import MuonWithAuxAdam, SingleDeviceMuon
from myalgo import validatorT, vanilla_kdT, bilevel_kd_tspT, bilevel_kd_penaltyT
from myalgody import tsp_dy_bilevel_kd
from func import collate_fn_pad
from demucs import Demucs
def main(args):
    DEVICE = 'cuda'
    EPOCHS = 100
    LR = args.xlr
    ZLR = args.zlr
    META_LR = args.alr
    LEN = 512 # fft_len
    BATCH_SIZE = 16

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(f'mission:KD-decums, lr: {LR}, zlr: {ZLR}, metalr:{META_LR}, batch_size: {BATCH_SIZE}, student_size: {args.student_size}, algorithm: {args.algorithm}, opt: {args.opt}')

    # load teacher
    teacher_model = Demucs(depth=5, resample=4, hidden=48).to(DEVICE)
    teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=DEVICE))
    teacher_model = teacher_model.to(DEVICE)
    teacher_model.eval()

    # saving
    save_path = f'./checkpoints/KD-decums-dykd/{LR}_{args.student_size}_{args.algorithm}_{args.opt}'
    os.makedirs(save_path, exist_ok=True)

    # load student
    if args.student_size == 'small':
        student_model = Demucs(depth=5, resample=4, hidden=24).to(DEVICE)
    elif args.student_size == 'tiny':
        student_model = Demucs(depth=5, resample=4, hidden=12).to(DEVICE)
    student_model = student_model.to(DEVICE)
    temp_model = copy.deepcopy(student_model)
    if args.frompre:
        student_model.load_state_dict(torch.load(args.frompre))

    # load data
    train_set = WavDataset('./1/noisy_trainset_28spk_wav')
    test_set = WavDataset('./1/noisy_testset_wav')
    num_total = len(train_set)
    num_val = int(num_total * 0.2)
    num_train = num_total - num_val
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [num_train, num_val], generator=generator)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE // 4, shuffle=True, collate_fn=collate_fn_pad)
    print(len(train_loader), len(val_loader))
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    # load muon
    class AlphaScheduler(nn.Module):
        def __init__(self, embedding_dim=8, hidden_dim=16):
            super().__init__()
            

            self.embedding = nn.Parameter(torch.randn(1, embedding_dim))
            
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )     

        def forward(self):
            logits = self.mlp(self.embedding)
            return torch.sigmoid(logits).squeeze()
    a_sche = AlphaScheduler()
    a_sche.to(DEVICE)
    last = {}
    optimizer = [torch.optim.AdamW(a_sche.parameters(), lr=META_LR), torch.optim.AdamW(student_model.parameters(), lr=LR), torch.optim.AdamW(temp_model.parameters(), lr=ZLR)]
    for epoch in range(EPOCHS):
        st = time.time()
        train_loss, train_lower_loss, train_upper_loss, last = tsp_dy_bilevel_kd(a_sche, student_model, temp_model, teacher_model, train_loader, val_loader, optimizer, epoch, DEVICE, last, args)

        val_loss, val_lower_loss, val_upper_loss, pesq, stoi = validatorT(student_model, teacher_model, test_loader, epoch, DEVICE)

        # scheduler.step(val_loss)
        print(f'epoch: {epoch}, a: {a_sche():.3f}, T-GT: {(train_upper_loss):.3f}, T-KD: {(train_lower_loss):.3f}, V-GT: {(val_upper_loss):.3f}, V-KD: {(val_lower_loss):.3f}, PESQ: {pesq:.3f}, STOI: {100 * stoi:.2f}, time: {(time.time()-st):.1f}')
        if epoch % 5 == 0:
            torch.save(student_model.state_dict(), f'{save_path}/epoch_{META_LR}_{epoch}.pth')
    print('done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_size', type=str, default='tiny')
    parser.add_argument('--teacher_path', type=str, default='./checkpoints/scartch_deno/0.0003_32/epoch_32.pth')
    parser.add_argument('--algorithm', type=str, default='tsp')
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--xlr', type=float, default=3e-4)
    parser.add_argument('--alr', type=float, default=3e-5)
    parser.add_argument('--zlr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--delta', type=float, default=1e-1)
    parser.add_argument('--tau', type=float, default=1e-2)
    parser.add_argument('--frompre', type=str, default=None)
    parser.add_argument('--a', type=float, default=0.5)


    args = parser.parse_args()
    print('algorithm:', args.algorithm)
    print('optimizer:', args.opt)
    main(args)
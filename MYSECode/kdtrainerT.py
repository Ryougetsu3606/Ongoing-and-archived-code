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
from func import collate_fn_pad
from demucs import Demucs
def main(args):
    DEVICE = 'cuda'
    EPOCHS = 100
    LR = args.xlr
    INNER_LR = args.ylr
    TEMP_LR = args.zlr
    LEN = 512 # fft_len
    BATCH_SIZE = 16

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(f'mission:KD-decums, lr: {LR}, glr:{INNER_LR},zlr:{TEMP_LR}, batch_size: {BATCH_SIZE}, student_size: {args.student_size}, algorithm: {args.algorithm}, opt: {args.opt}, penalty-gamma: {args.gamma}, penalty-delta: {args.delta}, inner-tau: {args.tau}')

    # load teacher
    teacher_model = Demucs(depth=5, resample=4, hidden=48).to(DEVICE)
    teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=DEVICE))
    teacher_model = teacher_model.to(DEVICE)
    teacher_model.eval()

    # saving
    if args.algorithm == 'vanilla':
        save_path = f'./checkpoints/KD-decums-sac/{LR}_{args.student_size}_{args.algorithm}_{args.opt}'
    else:
        save_path = f'./checkpoints/KD-decums-sac/{LR}_{args.student_size}_{args.algorithm}_inner{INNER_LR}_temp{TEMP_LR}_{args.opt}_gamma{args.gamma}'
    os.makedirs(save_path, exist_ok=True)

    # load student
    if args.student_size == 'small':
        student_model = Demucs(depth=5, resample=4, hidden=24).to(DEVICE)
    elif args.student_size == 'tiny':
        student_model = Demucs(depth=5, resample=4, hidden=12).to(DEVICE)
    student_model = student_model.to(DEVICE)
    if args.frompre:
        student_model.load_state_dict(torch.load(args.frompre))

    # load data
    train_set = WavDataset('./1/noisy_trainset_28spk_wav')
    val_set = WavDataset('./1/noisy_testset_wav')
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

    # load muon


    if args.algorithm == 'vanilla': # a=0: from scartch, a=0.3, normal
        trainer = vanilla_kdT
        if args.opt == 'AdamW':
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=LR)
            schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
            
        else: # SGD proven bad, using muon
            # optimizer = torch.optim.SGD(student_model.parameters(), lr=LR)
            pass
    elif args.algorithm == 'tsp':
        trainer = bilevel_kd_tspT
        temp_model = copy.deepcopy(student_model)
        temp_model = temp_model.to(DEVICE)
        if args.opt == 'AdamW':
            optimizer = [torch.optim.AdamW(itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters()), lr=LR), torch.optim.AdamW(student_model.encoder.parameters(), lr=INNER_LR), torch.optim.Adam(temp_model.encoder.parameters(), lr=TEMP_LR)]
            schedulers = []
            for opt in optimizer:
                 schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3))
        else:
            optimizer = []
            hidden_weights =  [p for p in itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()) if p.ndim >= 2]
            hidden_gain_biases = [p for p in itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()) if p.ndim < 1]
            param_groups = [
                dict(params=hidden_weights, use_muon=True, lr=50*LR, weight_decay=0.01),
                dict(params=hidden_gain_biases, use_muon=False, lr=LR, betas=(0.9, 0.95), weight_decay=0.01),
            ]
            optim = SingleDeviceMuon(param_groups)
            optimizer.append(optim)
            optimizer.append(torch.optim.Adam(student_model.encoder.parameters(), lr=INNER_LR))
            optimizer.append(torch.optim.Adam(temp_model.encoder.parameters(), lr=TEMP_LR))
            schedulers = []
            for opt in optimizer:
                 schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3))
    elif args.algorithm == 'penalty':
        trainer = bilevel_kd_penaltyT
        optimizer = [torch.optim.AdamW(student_model.parameters(), lr=LR), torch.optim.Adam(student_model.encoder.parameters(), lr=INNER_LR)]
        schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], mode='max', factor=0.5, patience=3), torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[1], mode='max', factor=0.5, patience=3)]
    else:
        raise NotImplementedError

    # optimizer = torch.optim.Adam(student_model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    last = {}
    for epoch in range(EPOCHS):
        st = time.time()
        if args.algorithm == 'tsp':
            train_loss, train_lower_loss, train_upper_loss, last = trainer(student_model, teacher_model, temp_model, train_loader, optimizer, epoch, DEVICE, last, args)
        else:
            train_loss, train_lower_loss, train_upper_loss = trainer(student_model, teacher_model, train_loader, optimizer, epoch, DEVICE)

        val_loss, val_lower_loss, val_upper_loss, pesq, stoi = validatorT(student_model, teacher_model, val_loader, epoch, DEVICE)
        if schedulers:
            if isinstance(optimizer, list):
                for i, scheduler in enumerate(schedulers):
                    scheduler.step(pesq)
                    print(f'optimizer {i} lr: {scheduler.get_last_lr()}')
            else:
                schedulers.step(pesq)
                print(f'optimizer lr: {schedulers.get_last_lr()}')
        # scheduler.step(val_loss)
        print(f'epoch: {epoch}, T-UL: {(train_upper_loss):.3f}, T-LL: {(train_lower_loss):.3f}, V-UL: {(val_upper_loss):.3f}, V-LL: {(val_lower_loss):.3f}, PESQ: {pesq:.3f}, STOI: {100 * stoi:.2f}, time: {(time.time()-st):.1f}')
        if args.algorithm == 'tsp':
            torch.save(student_model.state_dict(), f'{save_path}/epoch_tsp_{epoch}.pth')
        else:
            if epoch % 10 == 0:
                torch.save(student_model.state_dict(), f'{save_path}/epoch_{epoch}.pth')
    print('done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_size', type=str, default='tiny')
    parser.add_argument('--teacher_path', type=str, default='./checkpoints/scartch_deno/0.0003_32/epoch_32.pth')
    parser.add_argument('--algorithm', type=str, default='tsp')
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--xlr', type=float, default=3e-4)
    parser.add_argument('--ylr', type=float, default=3e-3)
    parser.add_argument('--zlr', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--delta', type=float, default=1e-1)
    parser.add_argument('--tau', type=float, default=1e-2)
    parser.add_argument('--frompre', type=str, default=None)


    args = parser.parse_args()
    print('algorithm:', args.algorithm)
    print('optimizer:', args.opt)
    main(args)
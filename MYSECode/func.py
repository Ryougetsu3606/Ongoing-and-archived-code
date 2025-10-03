import torch
from joblib import Parallel, delayed
from pesq import pesq
from pystoi import stoi
import numpy as np
from conv_stft import ConvSTFT
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stft = ConvSTFT(win_len=320, win_inc=160, fft_len=512, feature_type="complex")
stft = stft.to(DEVICE)
mse = torch.nn.functional.mse_loss
def stft2(x, fft_size, hop_size, win_length):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, torch.hann_window(win_length, device=x.device), return_complex=True)
    real = x_stft.real
    imag = x_stft.imag

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)

def demucs_stft_loss(y1,y2):
    # y1:pure, y2:enhance
    Lsc=torch.norm(y1-y2,p="fro")/torch.norm(y1,p="fro")
    Lmag=torch.nn.functional.l1_loss(torch.log(y1), torch.log(y2))
    return Lsc+Lmag

def demucs_loss(y1,y2):
    fft_size=[1024, 2048, 512]
    hop_size=[120, 240, 50]
    win_length=[600, 1200, 240]
    loss=0
    for i in range(3):
        fs = fft_size[i]
        ss = hop_size[i]
        wl = win_length[i]
        s1=stft2(y1,fs,ss,wl)
        s2=stft2(y2,fs,ss,wl)
        loss+=demucs_stft_loss(s1,s2)
    return loss/3
def pmsqe(s1, s2, lamda=10):
    dim = 512//2+1
    s1_real = s1[:, :dim, :]
    s1_imag = s1[:, dim:, :]
    s2_real = s2[:, :dim, :]
    s2_imag = s2[:, dim:, :]
    s1_mags = torch.sqrt(s1_real**2+s1_imag**2)
    s2_mags = torch.sqrt(s2_real**2+s2_imag**2)
    condition = s1_mags > s2_mags # s1=enhance, s2=clean(gt)
    weight = torch.where(condition, lamda, 1)
    se = (s1_mags - s2_mags) ** 2
    weight_se = se * weight
    return torch.mean(weight_se)

def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def latent_loss_cal(la1, la2, eps=1e-8):
    # assume the la1, la2 has the shape of [B, C, T, D], where only C is different
    la1 = la1.permute(2, 3, 0, 1)
    la2 = la2.permute(2, 3, 0, 1)
    la1_feature = torch.matmul(la1, la1.transpose(-1, -2))
    la2_feature = torch.matmul(la2, la2.transpose(-1, -2))
    if la1_feature.shape[-1] == 1:
        la1_feature = la1_feature.squeeze(-2, -1)
        la2_feature = la2_feature.squeeze(-2, -1)
    la1_feature = la1_feature / (torch.norm(la1_feature, p=2, dim=-1, keepdim=True) + eps)
    la2_feature = la2_feature / (torch.norm(la2_feature, p=2, dim=-1, keepdim=True) + eps)
    loss = mse(la1_feature, la2_feature)
    return loss

def latent_loss_calT(la1, la2, eps=1e-8):
    # assume the la1, la2 has the shape of [B, C, T, D], where only C is different
    la1 = la1.permute(2, 0, 1)
    la2 = la2.permute(2, 0, 1)
    la1_feature = torch.matmul(la1, la1.transpose(-1, -2))
    la2_feature = torch.matmul(la2, la2.transpose(-1, -2))
    if la1_feature.shape[-1] == 1:
        la1_feature = la1_feature.squeeze(-2, -1)
        la2_feature = la2_feature.squeeze(-2, -1)
    loss = 1-torch.mean(torch.cosine_similarity(la1_feature, la2_feature, dim=0))
    return loss

def kd_loss_cal(student_model, teacher_model, inputs):
    with torch.no_grad():
        teacher_outputs = teacher_model(inputs)
    student_outputs = student_model(inputs)
    kd_loss = 0
    for sls, tls in zip(student_outputs[2], teacher_outputs[2]):
        kd_loss += latent_loss_cal(sls, tls)
    return kd_loss

def kd_loss_calT(student_model, teacher_model, inputs):
    with torch.no_grad():
        teacher_outputs = teacher_model(inputs)
    student_outputs = student_model(inputs)
    kd_loss = 0
    for sls, tls in zip(student_outputs[1], teacher_outputs[1]):
        kd_loss += latent_loss_calT(sls, tls)
    return kd_loss

def gt_loss_cal(s1, s2):
    # s1_stft = stft(s1)
    # s2_stft = stft(s2)
    # loss1 = pmsqe(s1_stft, s2_stft)
    loss1 = -(si_snr(s1, s2))
    return loss1
def gt_loss_calT(s1, s2):
    # s1_stft = stft(s1)
    # s2_stft = stft(s2)
    # loss1 = pmsqe(s1_stft, s2_stft)
    loss = torch.nn.functional.l1_loss(s1,s2)
    loss1 = demucs_loss(s2, s1)
    return loss1+loss
def cal_stoi_batch(clean_wavs, dirty_wavs, FS=16000):
    stoi_score = Parallel(n_jobs=-1)(delayed(stoi)(c, n, FS, extended=False) for c, n in zip(clean_wavs, dirty_wavs))
    stoi_score = np.array(stoi_score)
    return np.mean(stoi_score)

def cal_pesq(clean_wav, dirty_wav, FS=16000):
    try:
        pesq_score = pesq(FS, clean_wav, dirty_wav, "wb")
    except:
        # print(' No utterances error')
        pesq_score = -1
    return pesq_score

def cal_pesq_batch(clean_wavs, dirty_wavs, FS=16000):
    pesq_score = Parallel(n_jobs=-1)(delayed(cal_pesq)(c, n, FS=FS) for c, n in zip(clean_wavs, dirty_wavs))
    pesq_score = np.array(pesq_score)
    return np.mean(pesq_score)

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)
def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm

def collate_fn_pad(batch):
    inputs, targets = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)
    if inputs.size(1) > 64000:
        inputs = inputs[:, :64000]
        targets = targets[:, :64000]
    return inputs, targets

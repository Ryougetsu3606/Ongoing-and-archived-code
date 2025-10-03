import torch
import itertools
import tqdm
from func import kd_loss_cal, gt_loss_cal, cal_stoi_batch, cal_pesq_batch, kd_loss_calT, gt_loss_calT
from tempcode import suband_loss_cal

def validator(student_model, teacher_model, val_loader, EPOCH, DEVICE):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    teacher_model.eval()
    student_model.eval()

    for inputs, targets in val_loader:
        batch_num += 1
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        with torch.no_grad():
            student_outputs = student_model(inputs)
        student_wav = student_outputs[1]
        
        sub_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_cal(student_wav, targets[:,:student_wav.shape[1]])
        sub_loss = 0.3 * sub_kd_loss + 0.7 * sub_gt_loss

        clean_wavs = targets.cpu().detach().numpy()[:,:student_wav.size(1)]
        enhance_wavs = student_wav.cpu().detach().numpy()
        bacth_pesq = cal_pesq_batch(clean_wavs, enhance_wavs)
        batch_stoi = cal_stoi_batch(clean_wavs, enhance_wavs)
        avg_pesq += bacth_pesq
        avg_stoi += batch_stoi

        loss += sub_loss.item()
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num, avg_pesq/batch_num, avg_stoi/batch_num

def validatorT(student_model, teacher_model, val_loader, EPOCH, DEVICE):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    teacher_model.eval()
    student_model.eval()

    for inputs, targets in val_loader:
        batch_num += 1
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        with torch.no_grad():
            student_outputs = student_model(inputs)
        student_wav = student_outputs[0]
        student_wav = student_wav.squeeze(1)
        sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_calT(student_wav, targets[:,:student_wav.shape[1]])
        sub_loss = 0.5 * sub_kd_loss + 0.5 * sub_gt_loss

        clean_wavs = targets.cpu().detach().numpy()[:,:student_wav.size(1)]
        enhance_wavs = student_wav.cpu().detach().numpy()
        bacth_pesq = cal_pesq_batch(clean_wavs, enhance_wavs)
        batch_stoi = cal_stoi_batch(clean_wavs, enhance_wavs)
        avg_pesq += bacth_pesq
        avg_stoi += batch_stoi

        loss += sub_loss.item()
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num, avg_pesq/batch_num, avg_stoi/batch_num

def vanilla_kd(student_model, teacher_model, train_loader, optimizer, EPOCH, DEVICE):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    batch_num = 0
    a = 0.5
    if EPOCH == 0:
        print(f"a={a}")

    teacher_model.eval()
    student_model.train()

    for inputs, targets in train_loader:
        batch_num += 1
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        student_outputs = student_model(inputs)
        student_wav = student_outputs[1]

        sub_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_cal(student_wav, targets[:,:student_wav.shape[1]])
        sub_loss = a * sub_kd_loss + (1 - a) * sub_gt_loss # hard-coding

        optimizer.zero_grad()
        sub_loss.backward()
        optimizer.step()

        loss += sub_loss.item()
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()

    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num

def vanilla_kdT(student_model, teacher_model, train_loader, optimizer, EPOCH, DEVICE):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    batch_num = 0
    a = 0.5
    if EPOCH == 0:
        print(f"a={a}")

    teacher_model.eval()
    student_model.train()

    for inputs, targets in train_loader:
        batch_num += 1
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        student_outputs = student_model(inputs)
        student_wav = student_outputs[0]
        student_wav = student_wav.squeeze(1)
        sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_calT(student_wav, targets[:,:student_wav.shape[1]])
        # print(sub_kd_loss.item())
        # print(sub_kd_loss.item(), sub_gt_loss.item())
        sub_loss = a * sub_kd_loss + (1 - a) * sub_gt_loss # hard-coding

        optimizer.zero_grad()
        sub_loss.backward()
        optimizer.step()

        loss += sub_loss.item()
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()

    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num

def suband(student_model, teacher_model, train_loader, optimizer, EPOCH, DEVICE):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    batch_num = 0
    a = 0.9
    if EPOCH == 0:
        print(f"a={a}")

    teacher_model.eval()
    student_model.train()

    for inputs, targets in train_loader:
        batch_num += 1
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        student_outputs = student_model(inputs)
        student_wav = student_outputs[1]

        sub_kd_loss = suband_loss_cal(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_cal(student_wav, targets[:,:student_wav.shape[1]])
        sub_loss = a * sub_kd_loss + (1 - a) * sub_gt_loss # hard-coding

        optimizer.zero_grad()
        sub_loss.backward()
        optimizer.step()

        loss += sub_loss.item()
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()

    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num

def bilevel_kd_bome(student_model, teacher_model, train_loader, optimizer, EPOCH, DEVICE):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    batch_num = 0
    eta = 0.5
    K = 3
    alpha = 1e-3

    teacher_model.eval()
    student_model.train()

    opt_full = optimizer[0]
    opt_y = optimizer[1]

    for inputs, targets in train_loader:
        batch_num += 1
        sub_kd_loss = 0
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        student_outputs = student_model(inputs)
        student_wav = student_outputs[1]

        sub_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_cal(student_wav, targets[:,:student_wav.shape[1]])
        
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()

        #======strat-update======
        grad_f = torch.autograd.grad(sub_gt_loss, student_model.parameters(), retain_graph=True)
        grad_g = torch.autograd.grad(sub_kd_loss, student_model.parameters(), retain_graph=True)
        #======inner-loop========
        for i in range(K):
            if i == 0:
                update_kd_loss = sub_kd_loss
            else:
                update_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)

            in_grad_y_g = torch.autograd.grad(update_kd_loss, student_model.encoder.parameters(), retain_graph=True)
            # with torch.no_grad():
            opt_y.zero_grad()
            for y, dy in zip(student_model.encoder.parameters(), in_grad_y_g):
                # y.sub_(dy, alpha=alpha)
                if dy is not None:
                    y.grad = dy
            opt_y.step()
            
        #======outer-loop========
        update_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
        grad_g_rest = torch.autograd.grad(update_kd_loss, student_model.parameters())
        grad_q = []
        # grad tuple to list then edit
        for i in range(len(grad_g)):
            grad_q.append(grad_g[i] - grad_g_rest[i])
        
        fq, qq = 0.0, 0.0
        for i in range(len(grad_q)):
            fq += torch.mul(grad_f[i], grad_q[i]).sum()
            qq += torch.mul(grad_q[i], grad_q[i]).sum()
        labda = max(0, eta - fq/qq)

        # with torch.no_grad():
        opt_full.zero_grad()
        for i, param in enumerate(student_model.parameters()):
            d = grad_f[i] + labda * grad_q[i]
            if d is not None:
                param.grad = d
            # param.sub_(d, alpha=alpha)
        opt_full.step()
        # print(f'\nUL:{sub_gt_loss:.6f}, LL:{sub_kd_loss:.6f}')
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num

def bilevel_kd_penaltyT(student_model, teacher_model, train_loader, optimizer, EPOCH, DEVICE):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    batch_num = 0
    K = 3
    gamma = min(0.2, 0.005 * EPOCH)
    teacher_model.eval()
    student_model.train()

    opt_full = optimizer[0]
    opt_y = optimizer[1]

    for inputs, targets in train_loader:
        batch_num += 1
        sub_kd_loss = 0
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        student_outputs = student_model(inputs)
        student_wav = student_outputs[0]
        student_wav = student_wav.squeeze(1)

        sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_calT(student_wav, targets[:,:student_wav.shape[1]])
        if batch_num % 100 == 0:
            print(f'f:{sub_gt_loss.item():.3f}, g:{sub_kd_loss.item():.3f}')
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()
        # ======start-update======
        grad_x_f = torch.autograd.grad(sub_gt_loss, itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters()), retain_graph=True)
        grad_y_f = torch.autograd.grad(sub_gt_loss, student_model.encoder.parameters(), retain_graph=True)
        grad_x_g_origin = torch.autograd.grad(sub_kd_loss, itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters()), retain_graph=True)
        grad_y_g = torch.autograd.grad(sub_kd_loss, student_model.encoder.parameters())
        # ======inner-loop========
        for i in range(K):
            if i == 0:
                in_grad_y = grad_y_g
            else:
                sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs)
                in_grad_y = torch.autograd.grad(sub_kd_loss, student_model.encoder.parameters())
            # with torch.no_grad():
            #     for y, dy in zip(student_model.encoder.parameters(), in_grad_y):
            #         y.sub_(dy, alpha=beta)
            opt_y.zero_grad()
            for y, dy in zip(student_model.encoder.parameters(), in_grad_y):
                if dy is not None:
                    y.grad = dy
            opt_y.step()

        # ======outer-loop========
        sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs)
        grad_x_g_new = torch.autograd.grad(sub_kd_loss, itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters()))
        opt_full.zero_grad()
        for i, x in enumerate(itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters())):
            grad_x = grad_x_f[i] + gamma * (grad_x_g_origin[i] - grad_x_g_new[i])
            # x.sub_(grad_x, alpha=alpha)
            x.grad = grad_x
        for j, y in enumerate(student_model.encoder.parameters()):
            grad_y = grad_y_f[j] + gamma * grad_y_g[j]
            y.grad = grad_y
            # y.sub_(grad_y, alpha=alpha)
        opt_full.step()
        # ======end-update======
        # print(f'\nUL:{sub_gt_loss:.6f}, LL:{sub_kd_loss:.6f}')
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num

def bilevel_kd_penalty(student_model, teacher_model, train_loader, optimizer, EPOCH, DEVICE):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    batch_num = 0
    K = 3
    alpha = 1e-3
    beta = 5e-4
    gamma = min(0.2, 0.005 * EPOCH)
    teacher_model.eval()
    student_model.train()

    opt_full = optimizer[0]
    opt_y = optimizer[1]

    for inputs, targets in train_loader:
        batch_num += 1
        sub_kd_loss = 0
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        student_outputs = student_model(inputs)
        student_wav = student_outputs[1]

        sub_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_cal(student_wav, targets[:,:student_wav.shape[1]])
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()
        # ======start-update======
        grad_x_f = torch.autograd.grad(sub_gt_loss, itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()), retain_graph=True)
        grad_y_f = torch.autograd.grad(sub_gt_loss, student_model.encoder.parameters(), retain_graph=True)
        grad_x_g_origin = torch.autograd.grad(sub_kd_loss, itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()), retain_graph=True)
        grad_y_g = torch.autograd.grad(sub_kd_loss, student_model.encoder.parameters())
        # ======inner-loop========
        for i in range(K):
            if i == 0:
                in_grad_y = grad_y_g
            else:
                sub_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
                in_grad_y = torch.autograd.grad(sub_kd_loss, student_model.encoder.parameters())
            # with torch.no_grad():
            #     for y, dy in zip(student_model.encoder.parameters(), in_grad_y):
            #         y.sub_(dy, alpha=beta)
            opt_y.zero_grad()
            for y, dy in zip(student_model.encoder.parameters(), in_grad_y):
                if dy is not None:
                    y.grad = dy
            opt_y.step()

        # ======outer-loop========
        sub_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
        grad_x_g_new = torch.autograd.grad(sub_kd_loss, itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()))
        # step_x = grad_x_f + gamma * (nabla_x_g_origin - nabla_x_g_new)
        # step_y = grad_y_f + gamma * nabla_y_g  
        # with torch.no_grad():
        #     for i, x in enumerate(itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters())):
        #         grad_x = grad_x_f[i] + gamma * (grad_x_g_origin[i] - grad_x_g_new[i])
        #         x.sub_(grad_x, alpha=alpha)
        #     for j, y in enumerate(student_model.encoder.parameters()):
        #         grad_y = grad_y_f[j] + gamma * grad_y_g[j]
        #         y.sub_(grad_y, alpha=alpha)
        opt_full.zero_grad()
        for i, x in enumerate(itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters())):
            grad_x = grad_x_f[i] + gamma * (grad_x_g_origin[i] - grad_x_g_new[i])
            # x.sub_(grad_x, alpha=alpha)
            x.grad = grad_x
        for j, y in enumerate(student_model.encoder.parameters()):
            grad_y = grad_y_f[j] + gamma * grad_y_g[j]
            y.grad = grad_y
            # y.sub_(grad_y, alpha=alpha)
        opt_full.step()
        # ======end-update======
        # print(f'\nUL:{sub_gt_loss:.6f}, LL:{sub_kd_loss:.6f}')
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num


def bilevel_kd_penalty_free(student_model, teacher_model, train_loader, optimizer, EPOCH, DEVICE):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    batch_num = 0
    alpha = 1e-3
    beta = 5e-4
    gamma = min(0.2, 0.02 * EPOCH)
    teacher_model.eval()
    student_model.train()

    opt_x = optimizer[0]
    opt_y = optimizer[1]

    for inputs, targets in train_loader:
        batch_num += 1
        sub_kd_loss = 0
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        student_outputs = student_model(inputs)
        student_wav = student_outputs[1]

        sub_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_cal(student_wav, targets[:,:student_wav.shape[1]])
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()
        # ======start-update======
        grad_y_f = torch.autograd.grad(sub_gt_loss, student_model.encoder.parameters(), retain_graph=True)
        grad_y_g = torch.autograd.grad(sub_kd_loss, student_model.encoder.parameters(), retain_graph=True)

        step_y = [gamma * dyf + dyg for dyf, dyg in zip(grad_y_f, grad_y_g)]

        # ===to-be-integrated-in-existed-optimizer===
        # with torch.no_grad():
        #     for y, dy in zip(student_model.encoder.parameters(), step_y):
        #         y.sub_(dy, alpha=alpha)
        for y, dy in zip(student_model.encoder.parameters(), step_y):
            if dy is not None:
                y.grad = dy

        student_wav = student_model(inputs)[1]
        sub_gt_loss = gt_loss_cal(student_wav, targets[:,:student_wav.shape[1]])
        grad_x = torch.autograd.grad(sub_gt_loss, itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()))
        # with torch.no_grad():
        #     for x, dx in zip(itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()), grad_x):
        #         x.sub_(dx, alpha=alpha)
        opt_x.zero_grad()
        for x, dx in zip(itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()), grad_x):
            if dx is not None:
                x.grad = dx
        opt_x.step()
        # print(f'\nUL:{sub_gt_loss:.6f}, LL:{sub_kd_loss:.6f}')
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num

def bilevel_kd_tsp(student_model, teacher_model, temp_model, train_loader, optimizer, EPOCH, DEVICE, last):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    tau = mu = theta = 1e-2
    eta = alpha = beta = 1e-3
    gamma = omega = 0.1
    p = 1
    delta = 1e-4
    if EPOCH != 0:
        h, x_hat, y_hat, lamda, lamda_p = last['h'], last['x_hat'], last['y_hat'], last['lamda'], last['lamda_p']
    else:
        h = 0
        x_hat = list(itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()))
        y_hat = list(student_model.encoder.parameters())
        lamda = 0
        lamda_p = 0
        # z = student_model.encoder.state_dict()
    batch_num = 0
    opt_x = optimizer[0]
    opt_y = optimizer[1]
    opt_z = optimizer[2]

    teacher_model.eval()
    student_model.train()

    for inputs, targets in train_loader:
        batch_num += 1
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        for x1, x2 in zip(itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()), itertools.chain(temp_model.enhance.parameters(), temp_model.decoder.parameters())):
            x2.data.copy_(x1.data)
        # ======start-update======
        # temp_model = copy.deepcopy(student_model)
        # temp_dict = temp_model.state_dict()
        # for k, v in z.items():
        #     k_real = 'encoder.' + k
        #     temp_dict[k_real] = v
        # temp_model.load_state_dict(temp_dict)
        # ===========================================update-h
        sub_kd_loss_y = kd_loss_cal(student_model, teacher_model, inputs)
        sub_kd_loss_z = kd_loss_cal(temp_model, teacher_model, inputs)
        kd_loss += sub_kd_loss_y.item()
        with torch.no_grad():
            y_flat = torch.cat([y.flatten() for y in student_model.encoder.parameters()])
            z_flat = torch.cat([z.flatten() for z in temp_model.encoder.parameters()])
            yznorm = torch.norm(y_flat - z_flat) ** 2
            h = (1 - theta) * h + theta * (sub_kd_loss_y - sub_kd_loss_z - 0.5 * yznorm / gamma - delta)
        # ==========================================update-lambda
        lamda_p = max(0, lamda + tau * h)
        lamda = (1 - mu) * lamda + mu * lamda_p
        # ===========================================update-z
        grad_z = torch.autograd.grad(sub_kd_loss_z, temp_model.encoder.parameters(), retain_graph=True)
        grad_x_g_z = torch.autograd.grad(sub_kd_loss_z, itertools.chain(temp_model.enhance.parameters(), temp_model.decoder.parameters()))
        # with torch.no_grad():
        #     for pz, y, dz in zip(temp_model.encoder.parameters(), student_model.encoder.parameters(), grad_z):
        #         step = dz + (pz - y) / gamma
        #         pz.sub_(step, alpha=eta)
        #     z = temp_model.encoder.state_dict()
        opt_z.zero_grad()
        for pz, y, dz in zip(temp_model.encoder.parameters(), student_model.encoder.parameters(), grad_z):
            if dz is not None:
                step = dz + (pz - y) / gamma
                pz.grad = step
        opt_z.step()
        # ============================================update-x,y
        student_wav = student_model(inputs)[1]
        sub_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_cal(student_wav, targets[:,:student_wav.shape[1]])
        gt_loss += sub_gt_loss.item()
        grad_f_y = torch.autograd.grad(sub_gt_loss, student_model.encoder.parameters(), retain_graph=True)
        grad_g_y = torch.autograd.grad(sub_kd_loss, student_model.encoder.parameters())
        # with torch.no_grad():
        #     for i, (name, pay2) in enumerate(student_model.encoder.named_parameters()):
        #         step1 = grad_f_y[i] + lamda * (grad_g_y[i] + (z[name] - pay2) / gamma) + p * (pay2 - y_hat[name]) # state_dict -> name, grad_list -> index
        #         pay2.sub_(step1, alpha=beta)
        #         y_hat_up = pay2 - y_hat[name]
        #         y_hat[name].add_(y_hat_up, alpha=omega)
        opt_y.zero_grad()
        i = 0
        for z, pay2 in zip(temp_model.encoder.parameters(), student_model.encoder.parameters()):
            if grad_f_y[i] is not None:
                step1 = grad_f_y[i] + lamda * (grad_g_y[i] + (z - pay2) / gamma) + p * (pay2 - y_hat[i]) # state_dict -> name, grad_list -> index
                pay2.grad = step1
                y_hat_up = pay2 - y_hat[i]
                with torch.no_grad():
                    y_hat[i].add_(y_hat_up, alpha=omega)
            i += 1
        opt_y.step()
        
        student_wav = student_model(inputs)[1]
        sub_kd_loss = kd_loss_cal(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_cal(student_wav, targets[:,:student_wav.shape[1]])
        grad_f_x = torch.autograd.grad(sub_gt_loss, itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()), retain_graph=True)
        grad_g_x = torch.autograd.grad(sub_kd_loss, itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters()))
        # with torch.no_grad():
        #     for i, (name, pax1) in enumerate(itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters())):
        #         step2 = grad_f_x[i] + lamda * (grad_g_x[i] - grad_x_g_z[i]) + p * (pax1 - x_hat[name])
        #         pax1.sub_(step2, alpha=alpha)
        #         x_hat_up = pax1 - x_hat[name]
        #         x_hat[name].add_(x_hat_up, alpha=omega)
        opt_x.zero_grad()
        for i, pax1 in enumerate(itertools.chain(student_model.enhance.parameters(), student_model.decoder.parameters())):
            if grad_f_x[i] is not None:
                step2 = grad_f_x[i] + lamda * (grad_g_x[i] - grad_x_g_z[i]) + p * (pax1 - x_hat[i])
                pax1.grad = step2
                x_hat_up = pax1 - x_hat[i]
                with torch.no_grad():
                    x_hat[i].add_(x_hat_up, alpha=omega)
        opt_x.step()
        # ============================================end update
        # print(f'\nUL:{sub_gt_loss:.6f}, LL:{sub_kd_loss:.6f}')
    last = {'h': h, 'x_hat': x_hat, 'y_hat': y_hat, 'lamda': lamda, 'lamda_p': lamda_p}
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num, last

def bilevel_kd_tspT(student_model, teacher_model, temp_model, train_loader, optimizer, EPOCH, DEVICE, last, args):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    tau = mu = theta = args.tau
    gamma = args.gamma
    omega = 0.1
    p = 1
    delta = args.delta
    if EPOCH != 0:
        h, x_hat, y_hat, lamda, lamda_p = last['h'], last['x_hat'], last['y_hat'], last['lamda'], last['lamda_p']
        print(f'last-lamda:{lamda:.3f}, last-lamda_p:{lamda_p:.3f}, last-h:{h:.3f}')
    else:
        h = 0
        x_hat = list(itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters()))
        y_hat = list(student_model.encoder.parameters())
        lamda = 0
        lamda_p = 0
        # z = student_model.encoder.state_dict()
    batch_num = 0
    opt_x = optimizer[0]
    opt_y = optimizer[1]
    opt_z = optimizer[2]

    teacher_model.eval()
    student_model.train()

    for inputs, targets in train_loader:
        batch_num += 1
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        for x1, x2 in zip(itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters()), itertools.chain(temp_model.lstm.parameters(), temp_model.decoder.parameters())):
            x2.data.copy_(x1.data)
        # ======start-update======
        # temp_model = copy.deepcopy(student_model)
        # temp_dict = temp_model.state_dict()
        # for k, v in z.items():
        #     k_real = 'encoder.' + k
        #     temp_dict[k_real] = v
        # temp_model.load_state_dict(temp_dict)
        # ===========================================update-h
        sub_kd_loss_y = kd_loss_calT(student_model, teacher_model, inputs)
        sub_kd_loss_z = kd_loss_calT(temp_model, teacher_model, inputs)
        kd_loss += sub_kd_loss_y.item()
        with torch.no_grad():
            y_flat = torch.cat([y.flatten() for y in student_model.encoder.parameters()])
            z_flat = torch.cat([z.flatten() for z in temp_model.encoder.parameters()])
            yznorm = torch.norm(y_flat - z_flat) ** 2
            h = (1 - theta) * h + theta * (sub_kd_loss_y - sub_kd_loss_z - 0.5 * yznorm / gamma - delta)
        # ==========================================update-lambda
        lamda_p = max(0, lamda + tau * h)
        lamda = (1 - mu) * lamda + mu * lamda_p
        # ===========================================update-z
        grad_z = torch.autograd.grad(sub_kd_loss_z, temp_model.encoder.parameters(), retain_graph=True)
        grad_x_g_z = torch.autograd.grad(sub_kd_loss_z, itertools.chain(temp_model.lstm.parameters(), temp_model.decoder.parameters()))
        # with torch.no_grad():
        #     for pz, y, dz in zip(temp_model.encoder.parameters(), student_model.encoder.parameters(), grad_z):
        #         step = dz + (pz - y) / gamma
        #         pz.sub_(step, alpha=eta)
        #     z = temp_model.encoder.state_dict()
        opt_z.zero_grad()
        for pz, y, dz in zip(temp_model.encoder.parameters(), student_model.encoder.parameters(), grad_z):
            if dz is not None:
                step = dz + (pz - y) / gamma
                pz.grad = step
        opt_z.step()
        # ============================================update-x,y
        student_wav = student_model(inputs)[0]
        student_wav = student_wav.squeeze(1)
        sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_calT(student_wav, targets[:,:student_wav.shape[1]])
        if batch_num % 50 == 0:
            print(f'f:{sub_gt_loss.item():.3f}, g:{sub_kd_loss.item():.3f}, y-z-norm:{torch.norm(torch.cat([y.flatten() for y in student_model.encoder.parameters()]) - torch.cat([z.flatten() for z in temp_model.encoder.parameters()])):.3f}, lamda:{lamda:.3f}, h:{h:.3f}')
        gt_loss += sub_gt_loss.item()
        grad_f_y = torch.autograd.grad(sub_gt_loss, student_model.encoder.parameters(), retain_graph=True)
        grad_g_y = torch.autograd.grad(sub_kd_loss, student_model.encoder.parameters())
        # with torch.no_grad():
        #     for i, (name, pay2) in enumerate(student_model.encoder.named_parameters()):
        #         step1 = grad_f_y[i] + lamda * (grad_g_y[i] + (z[name] - pay2) / gamma) + p * (pay2 - y_hat[name]) # state_dict -> name, grad_list -> index
        #         pay2.sub_(step1, alpha=beta)
        #         y_hat_up = pay2 - y_hat[name]
        #         y_hat[name].add_(y_hat_up, alpha=omega)
        opt_y.zero_grad()
        i = 0
        for z, pay2 in zip(temp_model.encoder.parameters(), student_model.encoder.parameters()):
            if grad_f_y[i] is not None:
                step1 = grad_f_y[i] + lamda * (grad_g_y[i] + (z - pay2) / gamma) + p * (pay2 - y_hat[i]) # state_dict -> name, grad_list -> index
                pay2.grad = step1
            i += 1
        opt_y.step()
        with torch.no_grad():
            for i, pay2 in enumerate(student_model.encoder.parameters()):
                y_hat_up = pay2 - y_hat[i]
                y_hat[i].add_(y_hat_up, alpha=omega)
        
        student_wav = student_model(inputs)[0]
        student_wav = student_wav.squeeze(1)
        sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs)
        sub_gt_loss = gt_loss_calT(student_wav, targets[:,:student_wav.shape[1]])
        grad_f_x = torch.autograd.grad(sub_gt_loss, itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters()), retain_graph=True)
        grad_g_x = torch.autograd.grad(sub_kd_loss, itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters()))

        opt_x.zero_grad()
        for i, pax1 in enumerate(itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters())):
            if grad_f_x[i] is not None:
                step2 = grad_f_x[i] + lamda * (grad_g_x[i] - grad_x_g_z[i]) + p * (pax1 - x_hat[i])
                pax1.grad = step2
        opt_x.step()
        with torch.no_grad():
            for i, pax1 in enumerate(itertools.chain(student_model.lstm.parameters(), student_model.decoder.parameters())):
                x_hat_up = pax1 - x_hat[i]
                x_hat[i].add_(x_hat_up, alpha=omega)
        # ============================================end update
        # print(f'\nUL:{sub_gt_loss:.6f}, LL:{sub_kd_loss:.6f}')
    last = {'h': h, 'x_hat': x_hat, 'y_hat': y_hat, 'lamda': lamda, 'lamda_p': lamda_p}
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num, last
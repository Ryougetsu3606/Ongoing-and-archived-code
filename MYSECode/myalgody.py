import torch
import itertools
import tqdm
from func import kd_loss_calT, gt_loss_calT
from tempcode import suband_loss_cal
from collections import OrderedDict
import copy
def dy_bilevel_kd(a_sche, student_model, teacher_model, train_loader, val_loader, optimizer, EPOCH, DEVICE, args):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    val_loss = 0
    batch_num = 0

    gamma = 0.1
    teacher_model.eval()
    student_model.train()
    a_sche.train()

    opt_a = optimizer[0]
    opt_th = optimizer[1]

    for t_inputs, v_inputs in zip(train_loader, val_loader):
        batch_num += 1
        # train_stage / lower level
        inputs_t = t_inputs[0].float().to(DEVICE)
        targets_t = t_inputs[1].float().to(DEVICE)
        student_outputs_t = student_model(inputs_t)
        student_wav_t = student_outputs_t[0]
        student_wav_t = student_wav_t.squeeze(1)

        inputs_v = v_inputs[0].float().to(DEVICE)
        targets_v = v_inputs[1].float().to(DEVICE)
        student_outputs_v = student_model(inputs_v)
        student_wav_v = student_outputs_v[0]
        student_wav_v = student_wav_v.squeeze(1)

        sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs_t)
        sub_gt_loss = gt_loss_calT(student_wav_t, targets_t[:,:student_wav_t.shape[1]])
        sub_val_loss = gt_loss_calT(student_wav_v, targets_v[:,:student_wav_v.shape[1]])
        # if batch_num % 10 == 0:
        #     print(a_sche())
        if batch_num % 100 == 0:
            print(f'gt:{sub_gt_loss.item():.3f}, kd:{sub_kd_loss.item():.3f}, val:{sub_val_loss.item():.3f}, a:{a_sche()}')
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()
        loss_g = a_sche() * sub_kd_loss + (1 - a_sche()) * sub_gt_loss
        loss_f = sub_val_loss

        # ======start-update======
        grads_theta_g = torch.autograd.grad(loss_g, student_model.parameters(), create_graph=True, retain_graph=True)
        grads_theta_f = torch.autograd.grad(loss_f, student_model.parameters(), retain_graph=True)
        grads_a_g_origin = torch.autograd.grad(loss_g, a_sche.parameters(), retain_graph=True)
        # grads_a_f = 0, since f() did not contain a
        
        # ======inner-loop========
        opt_th.zero_grad()
        for p, grad in zip(student_model.parameters(), grads_theta_g):
            p.grad = grad
        opt_th.step()
        # update g^
        student_outputs_t = student_model(inputs_t)
        student_wav_t = student_outputs_t[0]
        student_wav_t = student_wav_t.squeeze(1)

        sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs_t)
        sub_gt_loss = gt_loss_calT(student_wav_t, targets_t[:,:student_wav_t.shape[1]])
        loss_g = a_sche() * sub_kd_loss + (1 - a_sche()) * sub_gt_loss

        grads_a_g_new = torch.autograd.grad(loss_g, a_sche.parameters(), retain_graph=True)

        opt_a.zero_grad()
        for i, x in enumerate(a_sche.parameters()):
            grad_a = gamma * (grads_a_g_origin[i] - grads_a_g_new[i])
            x.grad = grad_a
        opt_a.step()
        opt_th.zero_grad()
        for i, x in enumerate(student_model.parameters()):
            grad_theta = (args.alr / args.xlr) * (grads_theta_f[i] + gamma * grads_theta_g[i]) # upper-lr = 3e-5
            x.grad = grad_theta
        opt_th.step()
        
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num

def tsp_dy_bilevel_kd(a_sche, student_model, temp_model, teacher_model, train_loader, val_loader, optimizer, EPOCH, DEVICE, last, args):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    val_loss = 0
    tau = mu = theta = 1e-3
    gamma = 1
    delta = 1e-1
    if EPOCH != 0:
        h, lamda, lamda_p = last['h'], last['lamda'], last['lamda_p']
    else:
        h = 0
        lamda = 0
        lamda_p = 0
    batch_num = 0
    opt_a = optimizer[0]
    opt_th = optimizer[1]
    opt_z = optimizer[2]

    teacher_model.eval()
    student_model.train()
    a_sche.train()
    temp_model.train()

    for t_inputs, v_inputs in zip(train_loader, val_loader):
        batch_num += 1
        # ===update dual variables
        # h
        inputs_t = t_inputs[0].float().to(DEVICE)
        targets_t = t_inputs[1].float().to(DEVICE)
        with torch.no_grad():
            student_outputs_t_y = student_model(inputs_t)
            student_wav_t_y = student_outputs_t_y[0]
            student_wav_t_y = student_wav_t_y.squeeze(1)
            sub_kd_loss_y = kd_loss_calT(student_model, teacher_model, inputs_t)
            sub_gt_loss_y = gt_loss_calT(student_wav_t_y, targets_t[:,:student_wav_t_y.shape[1]])
            sub_loss_y = a_sche() * sub_kd_loss_y + (1 - a_sche()) * sub_gt_loss_y

            student_outputs_t_z = temp_model(inputs_t)
            student_wav_t_z = student_outputs_t_z[0]
            student_wav_t_z = student_wav_t_z.squeeze(1)
            sub_kd_loss_z = kd_loss_calT(temp_model, teacher_model, inputs_t)
            sub_gt_loss_z = gt_loss_calT(student_wav_t_z, targets_t[:,:student_wav_t_z.shape[1]])
            sub_loss_z = a_sche() * sub_kd_loss_z + (1 - a_sche()) * sub_gt_loss_z

            h = (1 - theta) * h + theta * (sub_loss_y - sub_loss_z - 1 / (2 * gamma) * torch.norm(torch.cat([y.flatten() for y in student_model.parameters()]) - torch.cat([z.flatten() for z in temp_model.parameters()]), p=2) ** 2 - delta)
            # lamda
            lamda_p = max(0, lamda_p + tau * h)
            lamda = (1 - mu) * lamda + mu * lamda_p

        # ===update x,y,z
        student_outputs_t_y = student_model(inputs_t)
        student_wav_t_y = student_outputs_t_y[0]
        student_wav_t_y = student_wav_t_y.squeeze(1)
        sub_kd_loss_y = kd_loss_calT(student_model, teacher_model, inputs_t)
        sub_gt_loss_y = gt_loss_calT(student_wav_t_y, targets_t[:,:student_wav_t_y.shape[1]])
        sub_loss_y = a_sche() * sub_kd_loss_y + (1 - a_sche()) * sub_gt_loss_y

        student_outputs_t_z = temp_model(inputs_t)
        student_wav_t_z = student_outputs_t_z[0]
        student_wav_t_z = student_wav_t_z.squeeze(1)
        sub_kd_loss_z = kd_loss_calT(temp_model, teacher_model, inputs_t)
        sub_gt_loss_z = gt_loss_calT(student_wav_t_z, targets_t[:,:student_wav_t_z.shape[1]])
        sub_loss_z = a_sche() * sub_kd_loss_z + (1 - a_sche()) * sub_gt_loss_z
        # update z
        grads_g_z = torch.autograd.grad(sub_loss_z, temp_model.parameters(), retain_graph=True)
        grads_x_g_z = torch.autograd.grad(sub_loss_z, a_sche.parameters())

        opt_z.zero_grad()
        for pz, py, dz in zip(temp_model.parameters(), student_model.parameters(), grads_g_z):
            if dz is not None:
                step = dz + (pz - py) / gamma
                pz.grad = step
        opt_z.step()
        # update y
        grads_g_y = torch.autograd.grad(sub_loss_y, student_model.parameters())

        inputs_v = v_inputs[0].float().to(DEVICE)
        targets_v = v_inputs[1].float().to(DEVICE)
        students_ouputs_v = student_model(inputs_v)
        student_wav_v = students_ouputs_v[0]
        student_wav_v = student_wav_v.squeeze(1)
        sub_val_loss = gt_loss_calT(student_wav_v, targets_v[:,:student_wav_v.shape[1]])

        grads_f_y = torch.autograd.grad(sub_val_loss, student_model.parameters())

        opt_th.zero_grad()
        i = 0
        for pz, py in zip(temp_model.parameters(), student_model.parameters()):
            if grads_f_y[i] is not None:
                step = grads_f_y[i] + lamda * (grads_g_y[i] + (pz - py) / gamma)
                # if i == 0:
                #     print(step)
                py.grad = step
            i += 1
        opt_th.step()

        students_ouputs_t_y = student_model(inputs_t)
        student_wav_t_y = students_ouputs_t_y[0]
        student_wav_t_y = student_wav_t_y.squeeze(1)
        sub_gt_loss_y = gt_loss_calT(student_wav_t_y, targets_t[:,:student_wav_t_y.shape[1]])
        sub_kd_loss_y = kd_loss_calT(student_model, teacher_model, inputs_t)
        sub_loss_y = a_sche() * sub_kd_loss_y + (1 - a_sche()) * sub_gt_loss_y

        grads_gy_x = torch.autograd.grad(sub_loss_y, a_sche.parameters())
        i = 0
        opt_a.zero_grad()
        for i, pa in enumerate(a_sche.parameters()):
            if grads_gy_x[i] is not None and grads_x_g_z[i] is not None:
                step = lamda * (grads_gy_x[i] - grads_x_g_z[i])
                pa.grad = step
        opt_a.step()
        kd_loss += sub_kd_loss_y.item()
        gt_loss += sub_gt_loss_y.item()
        val_loss += sub_val_loss.item()
        if batch_num % 100 == 0:
            print(f'gt:{sub_gt_loss_y.item():.3f}, kd:{sub_kd_loss_y.item():.3f}, val:{sub_val_loss.item():.3f}, a: {a_sche()}, h: {h:.3f}, lamda: {lamda:.5f}')
        # torch.cuda.empty_cache()
    last = {'h':h, 'lamda':lamda, 'lamda_p':lamda_p}
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num, last

def fixed_kd(student_model, teacher_model, train_loader, optimizer, EPOCH, DEVICE, args):
    loss = 0
    kd_loss = 0
    gt_loss = 0
    val_loss = 0
    batch_num = 0
    a = args.a
    if EPOCH == 0:
        print(f"a={a}")
    teacher_model.eval()
    student_model.train()

    for t_inputs in train_loader:
        batch_num += 1
        # train_stage / lower level
        inputs_t = t_inputs[0].float().to(DEVICE)
        targets_t = t_inputs[1].float().to(DEVICE)
        student_outputs_t = student_model(inputs_t)
        student_wav_t = student_outputs_t[0]
        student_wav_t = student_wav_t.squeeze(1)

        sub_kd_loss = kd_loss_calT(student_model, teacher_model, inputs_t)
        sub_gt_loss = gt_loss_calT(student_wav_t, targets_t[:,:student_wav_t.shape[1]])
        # if batch_num % 10 == 0:
        #     print(a_sche())
        if batch_num % 100 == 0:
            print(f'gt:{sub_gt_loss.item():.3f}, kd:{sub_kd_loss.item():.3f}')
        kd_loss += sub_kd_loss.item()
        gt_loss += sub_gt_loss.item()
        loss = a * sub_kd_loss + (1 - a) * sub_gt_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # grads_a_f = 0, since f() did not contain a
        
        
    return loss/batch_num, kd_loss/batch_num, gt_loss/batch_num
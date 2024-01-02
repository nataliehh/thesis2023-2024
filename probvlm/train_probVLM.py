import os
import numpy as np
import torch
import torch.nn as nn

import clip
from ds import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders
from tqdm import tqdm
from losses import *
from utils import *

# From: https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/4
def load_checkpoint(model, optimizer, scheduler, resume_path='../ckpt/ProbVLM_Net_last.pth', log = True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(resume_path):
        if log:
            print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if log:
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))

    return model, optimizer, scheduler, start_epoch

def save_checkpoint(path, model, optimizer, scheduler, epoch, loss):
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(), 'loss': loss}, path)

def train_ProbVLM(CLIP_Net, BayesCap_Net, train_loader, eval_loader, Cri = TempCombLoss(), device='cuda', dtype=torch.float,
                  init_lr=1e-4, num_epochs=100, eval_every=1, ckpt_path='../ckpt/ProbVLM', cross_modal_lambda=1e-4,
                  T1=1e0, T2=5e-2, resume_path = '', log = True):
    CLIP_Net.to(device)
    BayesCap_Net.to(device)
    
    optimizer = torch.optim.Adam(list(BayesCap_Net.img_BayesCap.parameters())+list(BayesCap_Net.txt_BayesCap.parameters()), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    model, optimizer, scheduler, start_epoch = load_checkpoint(BayesCap_Net, optimizer, scheduler, resume_path = resume_path,
                                                              log = log)
    
    CLIP_Net.eval()
    ##
    BayesCap_Net.img_BayesCap.train()
    BayesCap_Net.txt_BayesCap.train()
    ##
    
    score = 1e8 # worst score 'so far'
    all_loss = []
    for eph in range(start_epoch, num_epochs):
        eph_loss = 0
        BayesCap_Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for (idx, batch) in enumerate(tepoch):
                if idx>2000:
                    break
                if log:
                    tepoch.set_description('Epoch {}'.format(eph))
                ##
                xI, xT  = batch[0].to(device), batch[1].to(device)
                with torch.no_grad():
                    outputs = CLIP_Net(xI, xT)
                    if type(outputs) == dict:
                        outputs = (outputs['image_features'], outputs['text_features'])
                    xfI, xfT = outputs

                # pass them through the network
                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)
                
                optimizer.zero_grad()
                loss_i = Cri(img_mu, img_1alpha, img_beta, xfI, T1=T1, T2=T2)
                loss_t = Cri(txt_mu, txt_1alpha, txt_beta, xfT, T1=T1, T2=T2)
                #cross modal terms
                loss_i4t = Cri(img_mu, img_1alpha, img_beta, xfT, T1=T1, T2=T2)
                loss_t4i = Cri(txt_mu, txt_1alpha, txt_beta, xfI, T1=T1, T2=T2)
                loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i)

                loss.backward()
                optimizer.step()
                ##
                eph_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            all_loss.append(eph_loss)
            if log:
                print('Avg. loss: {}'.format(eph_loss))
                with open('eval.txt', 'a') as f:
                    f.write(f'Epoch: {eph}\tavg. loss: {eph_loss}\n')
        # evaluate and save the models
        if eph % 5 == 0:
            save_checkpoint(ckpt_path+f'_epoch_{eph}.pth', BayesCap_Net, optimizer, scheduler, eph, eph_loss)
        save_checkpoint(ckpt_path+f'_last.pth', BayesCap_Net, optimizer, scheduler, eph, eph_loss)   
        # torch.save(BayesCap_Net.state_dict(), ckpt_path+'_last.pth')
        if eph % eval_every == 0 or eph == num_epochs - 1:
            curr_score = eval_ProbVLM(CLIP_Net, BayesCap_Net, eval_loader, device=device, dtype=dtype, log = log)
            
            if log:
                print('current score: {} | Last best score: {}'.format(curr_score, score))
                with open('eval.txt', 'a') as f:
                    f.write(f'Epoch: {eph}\tcurrent score: {curr_score}\tbest score: {score}\n')
                
            if curr_score <= score:
                score = curr_score
                save_checkpoint(ckpt_path+'_best.pth', BayesCap_Net, optimizer, scheduler, eph, eph_loss)
                # torch.save(BayesCap_Net.state_dict(), ckpt_path+'_best.pth')
    scheduler.step()
    
@torch.no_grad()    
def eval_ProbVLM(CLIP_Net, BayesCap_Net, eval_loader, device='cuda', dtype=torch.float, log = True):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    BayesCap_Net.to(device)
    BayesCap_Net.eval()

    mean_mse, mean_mae, num_imgs  = 0, 0, 0
    list_error, list_var = [], []

    with tqdm(eval_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
            if log: tepoch.set_description('Validating ...')
            ##
            xI, xT  = batch[0].to(device), batch[1].to(device)
            
            # pass them through the network
            with torch.no_grad():
                outputs = CLIP_Net(xI, xT)
                if type(outputs) is dict:
                    outputs = (outputs['image_features'], outputs['text_features'])
            xfI, xfT = outputs
            (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)
                
            n_batch = img_mu.shape[0]
            for j in range(n_batch):
                num_imgs += 1
                mean_mse += emb_mse(img_mu[j], xfI[j]) + emb_mse(txt_mu[j], xfT[j])
                mean_mae += emb_mae(img_mu[j], xfI[j]) + emb_mae(txt_mu[j], xfT[j])
            ##
        mean_mse /= num_imgs
        mean_mae /= num_imgs
        if log:
            print('Avg. MSE: {} | Avg. MAE: {}'.format(mean_mse, mean_mae))
            with open('eval.txt', 'a') as f:
                    f.write(f'Avg. MSE: {mean_mse}\tAvg. MAE: {mean_mae}\n')
    return mean_mae

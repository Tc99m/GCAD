import argparse
import os
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from copy import deepcopy

from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def simple_moving_average(arr, label, window_size):
    '''window_size should be an odd int'''
    if len(arr)!=len(label):
        print("len(score) != len(label)")
        return None
    moving_averages = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i + window_size]
        average = sum(window) / window_size
        moving_averages.append(average)
    n = int((window_size-1)/2)
    
    return [moving_averages, label[n:len(arr)-n]]


def get_err_norm_parms(model,save_path,dataloader,device,parms_path,sample_p=0.1):

    saved_model = torch.load(save_path)
    model.load_state_dict(saved_model)

    # start testing
    model.eval()
    
    test_mloss = torch.zeros(1, device=device)
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    all_loss = []

    # with torch.no_grad():
    first_tag = 0

    for i, (batch_x, batch_y) in pbar:    #, batch_labels

        '''batch_x:(batch, win_len, seq_num)
            batch_y:(batch, pre_len, seq_num)
            batch_labels:(batch, win_len)
            outputs:(batch, pre_len, seq_num)
        '''
        sample = random.random()
        if (sample <= sample_p) or (first_tag==0):


            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_x.requires_grad = True
            outputs = model(batch_x).float().to(device) 

            loss = criterion(outputs, batch_y)
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)

            Grad = torch.autograd.grad(loss, batch_x, allow_unused=True, create_graph=False, retain_graph=False)     #Grad[0].shape: (batch, window_len, n_sensors)

            # current_gpu_index = torch.cuda.current_device()
            # print(torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3))
            
            Grad = Grad[0]
            Grad = torch.abs(Grad)
            '''Grad.shape: (batch_size, input_len, num_features)'''

            if first_tag==0:
                first_tag = 1
                input_grad = Grad
            else:
                input_grad = torch.cat((input_grad,Grad), dim=0)   #shape: (n*batch_size, pred_len, num_features)

    temp_shape = input_grad.shape
    input_grad_reshape = torch.reshape(input_grad,((-1,temp_shape[-1])))     #(-1,num_features)
    err_norm_mean = torch.mean(input_grad_reshape,dim=0,keepdim=False)    #(,num_features)
    err_norm_std = torch.std(input_grad_reshape,dim=0,keepdim=False)      #(,num_features)
    
    err_norm_mean = err_norm_mean.data.cpu().numpy()      #(num_features,)
    err_norm_std = err_norm_std.data.cpu().numpy()
    
    df = pd.DataFrame({
        'mean':err_norm_mean,
        'std':err_norm_std
    })
    df.to_csv(parms_path)   


def save_train_mean_causal(model,save_path,dataloader,device,parms_path,sparse_th,sample_p=0.01):

    saved_model = torch.load(save_path)
    model.load_state_dict(saved_model)

    # start testing
    model.eval()
    
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    first_tag = 0
    
    print("sampling causal matrix on train set with rate", sample_p)

    for i, (batch_x, batch_y) in pbar:    #, batch_labels

        '''batch_x:(batch, win_len, seq_num)
            batch_y:(batch, pre_len, seq_num)
            batch_labels:(batch, win_len)
            outputs:(batch, pre_len, seq_num)
        '''

        
        sample = random.random()
        if (sample <= sample_p) or (first_tag==0):
            
            model.zero_grad()


            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_x.requires_grad = True
            outputs = model(batch_x).float().to(device) 
            
            for features in range(outputs.shape[-1]):
                

                model.zero_grad()
                
                loss_i = criterion(outputs[:,:,features],batch_y[:,:,features])
                loss_i.requires_grad_(True)
                loss_i.backward(retain_graph=True)

                Grad_i = batch_x.grad

                Grad_i = torch.abs(Grad_i)
                
                batch_x.grad = None  
                
                
                
                if features==0:
                    grad_causal_mat = torch.unsqueeze(Grad_i,dim=3)
                else:
                    #grad_causal_mat.shape: (batch_size,input_win,num_features(input),features(output))
                    grad_causal_mat = torch.cat([grad_causal_mat,torch.unsqueeze(Grad_i,dim=3)],dim=3)      
                
            if first_tag==0:
                input_grad_causal_map = grad_causal_mat
                first_tag = 1
            else:
                #(n*batch_size,features(input),input_win,num_features(output))
                input_grad_causal_map = torch.cat([input_grad_causal_map,grad_causal_mat],dim=0)
            

        # current_gpu_index = torch.cuda.current_device()
        # print(torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3))
            
    #(n*batch_size,features(input),num_features(output))
    input_grad_causal_mat = torch.mean(input_grad_causal_map,dim=1)
    

    upper_triangle = torch.triu(input_grad_causal_mat, diagonal=0)  
    lower_triangle_transposed = torch.tril(input_grad_causal_mat, diagonal=-1).transpose(1,2) 
    result = torch.triu(upper_triangle - lower_triangle_transposed, diagonal=0) 

    result_upper = torch.where(result < 0, torch.zeros_like(result).to(device), result)
    result_lower = torch.where(result < 0, torch.abs(result).to(device), torch.zeros_like(result).to(device)).transpose(1,2)
    input_grad_causal_mat = result_upper + result_lower
    
    

    zero = torch.zeros_like(input_grad_causal_mat).to(device)
    input_grad_causal_mat = torch.where(input_grad_causal_mat<sparse_th, zero, input_grad_causal_mat)
    

    
    #(features(input),num_features(output))
    input_grad_causal_map = torch.mean(input_grad_causal_mat,dim=0)
    
 
    
    
    input_grad_causal_map = input_grad_causal_map.data.cpu().numpy()
    df = pd.DataFrame(input_grad_causal_map)
    df.to_csv(parms_path,index=False,header=False)


def test(model,save_path,dataloader,device,parms_path,sparse_th):

    saved_model = torch.load(save_path)
    model.load_state_dict(saved_model)

    # start testing
    model.eval()
    
    test_mloss = torch.zeros(1, device=device)
    criterion = torch.nn.MSELoss(reduction='sum').to(device)

    print(('\n' + '%-10s' * 1) % ('Test loss'))
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    all_loss = []

    # with torch.no_grad():

    for i, (batch_x, batch_y, batch_labels) in pbar:    #, batch_labels

        '''batch_x:(batch, win_len, seq_num)
            batch_y:(batch, pre_len, seq_num)
            batch_labels:(batch, win_len)
            outputs:(batch, pre_len, seq_num)
        '''
        
        # batch_size = batch_x.shape[0]

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        batch_labels = batch_labels.to(device)
        label,_ = torch.max(batch_labels, dim=1)

        batch_x.requires_grad = True
        outputs = model(batch_x).float().to(device)      #(batch,pre_len,num_features)
        
        for features in range(outputs.shape[-1]):

            model.zero_grad()

            Grad_i = torch.zeros_like(batch_x)
                
            loss_i = criterion(outputs[:,:,features],batch_y[:,:,features])
            loss_i.requires_grad_(True)
            loss_i.backward(retain_graph=True)
            #Grad_i.shape: (batch, window_len, n_sensors)
            Grad_i = batch_x.grad
            #Grad_i = torch.autograd.grad(loss, batch_x, allow_unused=True, create_graph=False, retain_graph=False)[0]
            Grad_i = torch.abs(Grad_i)
            
            # Grad_i = torch.div(Grad_i,torch.sqrt(loss_i))
            
            batch_x.grad = None 
            
            
            if features==0:
                grad_causal_mat = torch.unsqueeze(Grad_i,dim=3)
            else:
                #grad_causal_mat.shape: (batch_size,input_win,num_features(input),features(output))
                grad_causal_mat = torch.cat([grad_causal_mat,torch.unsqueeze(Grad_i,dim=3)],dim=3)      
        # #       (batch_size,features(output),input_win,num_features(input))  
        # grad_causal_mat = torch.transpose(grad_causal_mat,dim0=0,dim1=1)
        # #       (batch_size,features(input),input_win,num_features(output)) 
        # grad_causal_mat = torch.transpose(grad_causal_mat,dim0=1,dim1=3)
                
        if i==0:
            input_grad_causal_mat = grad_causal_mat
            test_labels = label
        else:
            #(n*batch_size,features(input),input_win,num_features(output))
            input_grad_causal_mat = torch.cat([input_grad_causal_mat,grad_causal_mat],dim=0)
            test_labels = torch.cat((test_labels,label), dim=0)   #shape: (n*batch_size,)
            
        # current_gpu_index = torch.cuda.current_device()
        # print(torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3))
            
    #(n*batch_size,features(input),num_features(output))
    input_grad_causal_mat = torch.mean(input_grad_causal_mat,dim=1)
    
    upper_triangle = torch.triu(input_grad_causal_mat, diagonal=0) 
    lower_triangle_transposed = torch.tril(input_grad_causal_mat, diagonal=-1).transpose(1,2) 
    result = torch.triu(upper_triangle - lower_triangle_transposed, diagonal=0)

    result_upper = torch.where(result < 0, torch.zeros_like(result).to(device), result)
    result_lower = torch.where(result < 0, torch.abs(result).to(device), torch.zeros_like(result).to(device)).transpose(1,2)
    input_grad_causal_mat = result_upper + result_lower


    zero = torch.zeros_like(input_grad_causal_mat).to(device)
    input_grad_causal_mat = torch.where(input_grad_causal_mat<sparse_th, zero, input_grad_causal_mat)

                
    standard_causal_mat_df = pd.read_csv(parms_path,index_col=None,header=None)
    #shape(features(input),num_features(output))
    standard_causal_mat = torch.tensor(np.array(standard_causal_mat_df)).to(device)
    criterion = torch.nn.MSELoss().to(device)


    standard_causal_mat = standard_causal_mat + 1e-4
    for n_samples in range(input_grad_causal_mat.shape[0]):

        temp_error = torch.abs(input_grad_causal_mat[n_samples,:,:] - standard_causal_mat)
        temp_error = torch.div(temp_error,standard_causal_mat)
        temp_error = torch.mean(temp_error)

        if n_samples==0:
            err_score = torch.tensor([temp_error]).to(device)
        else:
            err_score = torch.cat([err_score,temp_error.unsqueeze(0)],dim=0)
    
    
    # for n_samples in range(input_grad_causal_mat.shape[0]):
    #     if n_samples==0:
    #         err_score = torch.tensor([criterion(input_grad_causal_mat[n_samples,:,:],standard_causal_mat)])
    #     else:
    #         err_score = torch.cat([err_score,torch.tensor([criterion(input_grad_causal_mat[n_samples,:,:],standard_causal_mat)])],dim=0)
    
    err_score = err_score.data.cpu().numpy()
    test_labels = test_labels.data.cpu().numpy()
    
    # SMA
    smoothed = simple_moving_average(err_score,test_labels,window_size=3)
    err_score = smoothed[0]
    test_labels = smoothed[1]
    
    

    auc_score = roc_auc_score(test_labels, err_score)
    print("ROC_score:",auc_score)
    precision, recall, thresholds = precision_recall_curve(test_labels, err_score)
    auc_precision_recall = auc(recall, precision)
    print("PRC_score:",auc_precision_recall)
    
    f1_scores = 2*recall*precision/(recall+precision+1e-10)
    
    f1 = np.max(f1_scores)
    pre = precision[np.argmax(f1_scores)]
    rec = recall[np.argmax(f1_scores)]
    print('Best F1-Score: ', f1)
    print('Precision: ', pre)
    print('Recall: ', rec)
    
    # point adjustment
    
    th_pa = thresholds[np.argmax(f1_scores)]
    pred = np.where(err_score >= th_pa, 1, 0).astype(int)
    gt = test_labels.astype(int)
    
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    # print("pred: ", pred.shape)
    # print("gt:   ", gt.shape)
    
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    
    accuracy_pa = accuracy_score(gt, pred)
    precision_pa, recall_pa, f1_pa, support = precision_recall_fscore_support(gt, pred,
                                                                            average='binary')
    
    eva_list = [auc_score,auc_precision_recall,f1,pre,rec,f1_pa]
            
    #[auc_roc,auc_precision_recall,f1,pre,rec,f1_pa]
    return eva_list 
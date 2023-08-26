import torch
import pandas as pd
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
import scipy
import torch.nn.functional as F
import torchvision

from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve, auc,confusion_matrix
from sklearn.feature_selection import r_regression

import cv2
import os
blue = lambda x: '\033[94m' + x + '\033[0m'
red = lambda x: '\033[31m' + x + '\033[0m'
green = lambda x: '\033[32m' + x + '\033[0m'
yellow = lambda x: '\033[33m' + x + '\033[0m'
greenline = lambda x: '\033[42m' + x + '\033[0m'
yellowline = lambda x: '\033[43m' + x + '\033[0m'
def record(values,epoch,writer,phase="Train"):
    """ tfboard write """
    for key,value in values.items():
        writer.add_scalar(key+"/"+phase,value,epoch)           
def calculate(y_hat,y_true,y_hat_logit):
    """ calculate five metrics using y_hat, y_true, y_hat_logit """
    train_acc=(np.array(y_hat) == np.array(y_true)).sum()/len(y_true) 
    recall=recall_score(y_true, y_hat,zero_division=0)
    precision=precision_score(y_true, y_hat,zero_division=0)
    Fscore=f1_score(y_true, y_hat,zero_division=0)
    roc=roc_auc_score(y_true, scipy.special.softmax(np.array(y_hat_logit),axis=1)[:,1])
    return train_acc,recall,precision,Fscore,roc
def calculate_map(y_hat,y_true,y_hat_logit,attmap_bool_selected_l=None,targets_att_bool_l=None):
    train_acc,recall,precision,Fscore,roc=calculate(y_hat,y_true,y_hat_logit)
    if attmap_bool_selected_l ==None:
        return train_acc,recall,precision,Fscore,roc
    else:
        iou = compute_iou(attmap_bool_selected_l,targets_att_bool_l)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        TPR=recall
        FPR=fp/(fp+tn)
        # TSS = TPR + TNR - 1 = TPR - FPR
        tss=TPR - FPR
        return train_acc,recall,precision,Fscore,roc,iou,tss

def print_1(epoch,phase,loss,acc,recall,precision,Fscore,roc,iou=None,tss=None,color=None):
    """ print epoch info"""
    if iou is None:
        if color is not None:
            print(color( f"epoch[{epoch:d}] {phase} loss: {loss:.3f} acc: {acc:.3f}  recall: {recall:.3f} precision: {precision:.3f} Fscore: {Fscore:.3f} roc: {roc:.3f}" ))
        else:
            print(( f"epoch[{epoch:d}] {phase} loss: {loss:.3f} acc: {acc:.3f}  recall: {recall:.3f} precision: {precision:.3f} Fscore: {Fscore:.3f} roc: {roc:.3f}" ))
    else:
        if color is not None:
            print(color( f"epoch[{epoch:d}] {phase} loss: {loss:.3f} acc: {acc:.3f}  recall: {recall:.3f} precision: {precision:.3f} Fscore: {Fscore:.3f} roc: {roc:.3f} tss: {tss:.3f} iou: {iou:.3f}" ))
        else:
            print(( f"epoch[{epoch:d}] {phase} loss: {loss:.3f} acc: {acc:.3f}  recall: {recall:.3f} precision: {precision:.3f} Fscore: {Fscore:.3f} roc: {roc:.3f} tss: {tss:.3f} iou: {iou:.3f}" ))
    
def compute_iou(x, y):
    if type(x) is torch.Tensor:
        x=x.numpy()
        y=y.numpy()
    intersection = np.bitwise_and(x, y)
    union = np.bitwise_or(x, y)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_exp_score(x, y):
    N =  np.sum(y!=0)
    epsilon = 1e-6
    tp = np.sum( x * (y>0))
    tn = np.sum((1-x) * (y<0))
    fp = np.sum( x * (y<0))
    fn = np.sum((1-x) * (y>0))

    exp_precision = tp / (tp + fp + epsilon)
    exp_recall = tp / (tp + fn + epsilon)
    exp_f1 = 2 * (exp_precision * exp_recall) / (exp_precision + exp_recall + epsilon)
    return exp_precision, exp_recall, exp_f1                                    

def normalize_image(cam):
    if type(cam) is np.ndarray:
        min_val = np.min(cam, axis=(1, 2), keepdims=True)
        max_val = np.max(cam, axis=(1, 2), keepdims=True)
        diff = max_val - min_val
        diff[diff == 0] = np.inf        
    elif type(cam) is torch.Tensor:
        min_val = torch.min(torch.min(cam, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        max_val = torch.max(torch.max(cam, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]        
        diff = max_val - min_val
        diff[diff == 0] = torch.inf
    cam = (cam - min_val) / diff #01 norm 
    return cam
def normalize_image_wholetime(cam):
    if type(cam) is np.ndarray:
        min_val = np.min(cam, axis=(0, -2,-1), keepdims=True)
        max_val = np.max(cam, axis=(0, -2,-1), keepdims=True)
        diff = max_val - min_val
        diff[diff == 0] = np.inf        
    cam = (cam - min_val) / diff #01 norm 
    return cam

def show_cam_on_image(img,mask,filenames,path):
    # time_length=img.shape[0]
    # for i in range(time_length):
    #     save_path = os.path.join(path, filenames[-1]+filenames[i])
    #     heatmap = cv2.applyColorMap(np.uint8(255 * mask[i]), cv2.COLORMAP_JET) 
    #     heatmap = np.float32(heatmap) / 255
    #     if img[i].shape[0]==1:
    #         cam = heatmap + img[i][0][:,:,None]
    #         cam = cam / np.max(cam)
    #         cv2.imwrite(save_path, np.uint8(255 * cam))
    #     elif img[i].shape[0]==4:
    #         tempimage=normalize_image(img[i])
    #         for j in range(4):
    #             tempimage_j=tempimage.transpose((1,2,0))[:,:,j][:,:,None]
    #             cam = heatmap + tempimage_j
    #             cam = cam / np.max(cam)
    #             cv2.imwrite(save_path+"_"+str(j)+".png", np.uint8(255 * cam))
    time_length=img.shape[0]
    for i in range(time_length):
        save_path = os.path.join(path, "origial_"+filenames[i])
        tempimage=normalize_image(img[i])
        for j in range(4):
            tempimage_j=tempimage.transpose((1,2,0))[:,:,j][:,:,None]
            cam = tempimage_j
            cv2.imwrite(save_path+"_"+str(j)+".png", np.uint8(255 * cam))

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, feature_module, target_layers):
        self.model = feature_module 
        self.target_layers = target_layers
        self.gradients = []
    def save_gradient(self, grad):
        self.gradients.append(grad)
    def __call__(self, x):
        features = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                features += [x]
        return features, x
class ModelOutputs():
    """ Class for forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
    def get_gradients(self):
        return self.feature_extractor.gradients
    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "imp" in name.lower(): # annotation imputation
                continue
            else:
                x = module(x)
        return target_activations, x

def last_only_mask(t_length,batch_size):
    mask=torch.zeros((t_length,batch_size))
    mask[-1]=1
    mask=mask.flatten()
    return mask

def percent_mask(t_length,batch_size,anno_percent):
    mask=torch.zeros((t_length,batch_size))
    idx = torch.randperm(batch_size)[:int(batch_size*anno_percent)]
    mask[-1,idx]=1
    idx2=torch.rand((t_length-1,batch_size))
    mask[:-1][idx2<=anno_percent]=1
    mask=mask.flatten()
    return mask
    
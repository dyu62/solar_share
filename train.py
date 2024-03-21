import argparse
import os
import random
import torch
import pandas as pd
import numpy as np
import time
import torch.optim as optim
import scipy

from matplotlib import cm
import matplotlib.pyplot as plt
import json
from model import resnet18,MultiLayerFeedForwardNN,CNNRNNinterpolation
import torch.nn.functional as F
from torch.nn.functional import softmax
import copy

torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import explained_variance_score,mean_squared_error,mean_absolute_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve, auc
from sklearn.feature_selection import r_regression
import pickle
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import dataset,util
from dataset import DatasetFolderWithPaths,cvsLoader,pil_loader, get_solar_dataset

import cv2

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models
import math
import shutil
import time
from datetime import date, timedelta,datetime

blue = lambda x: '\033[94m' + x + '\033[0m'
red = lambda x: '\033[31m' + x + '\033[0m'
green = lambda x: '\033[32m' + x + '\033[0m'
yellow = lambda x: '\033[33m' + x + '\033[0m'
greenline = lambda x: '\033[42m' + x + '\033[0m'
yellowline = lambda x: '\033[43m' + x + '\033[0m'

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', default='data/allsolar_png512', type=str)
    parser.add_argument('--data_dir', default='data/allsolar_full_png512', type=str)
    
    parser.add_argument('--model',default="our", type=str)
    parser.add_argument('--train_batch', default=16, type=int)
    parser.add_argument('--test_batch', default=16, type=int)
    
    parser.add_argument('--time_length', default=4, type=int)
    parser.add_argument('--h_ch', default=32, type=int)
    parser.add_argument('--eta', default=0.1, type=float) #tolerant
    parser.add_argument('--att_coef', default=1, type=float)
    parser.add_argument('--rec_coef', default=1, type=float)
    parser.add_argument('--anno_p', default=1, type=float)

    parser.add_argument('--dataset', type=str, default='solar')
    parser.add_argument('--log', type=str, default="True")
    parser.add_argument('--loadmodel', type=str, default="False")     
    parser.add_argument('--test_per_round', type=int, default=10)
    parser.add_argument('--patience', type=int, default=30)  #scheduler
    parser.add_argument('--nepoch', type=int, default=201)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--manualSeed', type=str, default="False")
    parser.add_argument('--man_seed', type=int, default=12345)
    args = parser.parse_args()
    args.log=True if args.log=="True" else False
    args.loadmodel=True if args.loadmodel=="True" else False    
    args.save_dir=os.path.join('./save/',args.dataset)
    args.manualSeed=True if args.manualSeed=="True" else False
    return args

def main(args,train_Loader,val_Loader,test_Loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    measure_Pearsonr=r_regression
    criterion_l1 = torch.nn.L1Loss(reduction='none') #reduction='sum'
    criterion_l2 = torch.nn.MSELoss()
    criterion=nn.CrossEntropyLoss()
    BCE_criterion = nn.BCELoss(reduction='none')
    gaussianblur=torchvision.transforms.GaussianBlur((3,3))
    last_only_mask=util.last_only_mask(args.time_length,args.train_batch)
    percent_mask=util.percent_mask(args.time_length,args.train_batch,args.anno_p)
    if args.dataset in ["solar"]:
        model = resnet18(input_ch=1)
    model=nn.Sequential(*list(model.children())[:-1])
    tmodel=torch.nn.GRU(512,args.h_ch,1) #input hidden layer
    finalmlpmodel=MultiLayerFeedForwardNN(args.h_ch,2,num_hidden_layers=0,hidden_dim=args.h_ch)
    model.to(device), tmodel.to(device), finalmlpmodel.to(device)
    # t_decay=[torch.tensor(0,dtype=torch.float32).requires_grad_()]
    opt_list=list(model.parameters())+list(tmodel.parameters())+list(finalmlpmodel.parameters())
    if args.model in ["our","our-t"]:
        interpolation_model=CNNRNNinterpolation()
        interpolation_model.to(device)
        opt_list=opt_list+list(interpolation_model.parameters())
    if args.model in ["our","our-p"]:
        decay_mlp=MultiLayerFeedForwardNN(args.time_length,1,num_hidden_layers=2,hidden_dim=args.h_ch)
        decay_mlp.to(device)
        opt_list=opt_list+list(decay_mlp.parameters())
    optimizer = torch.optim.Adam( opt_list, lr=args.lr)    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, min_lr=1e-8)   
    best_model=None
    class GradCam:
        def __init__(self, model, feature_module, target_layer_names):
            self.model = model 
            self.feature_module = feature_module
            self.extractor = util.ModelOutputs(self.model, self.feature_module, target_layer_names)
        def forward(self, input_data):
            return self.model(input_data)
        def get_attention_map(self, X,batchsize,tmodel,finalmlpmodel, index=None,is_testing=False ): 
            featuremaps, avgpool_outputs = self.extractor(X)
            
            t_X=avgpool_outputs.view(args.time_length,batchsize,-1)
            t_output,_=tmodel(t_X) 
            outputs=finalmlpmodel(t_output[-1])
                        
            one_hot = np.zeros((outputs.shape[0], outputs.size()[-1]), dtype=np.float32) 
            if is_testing:
                index = np.argmax(outputs.cpu().data.numpy(),1)
                one_hot[:,index] = 1
            else:
                one_hot[:,index.cpu()] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.to(device) * outputs)
            
            self.feature_module.zero_grad()
            self.model.zero_grad()
            if is_testing:
                one_hot.backward(retain_graph=False)
            else:
                one_hot.backward(retain_graph=True)

            grads_val = self.extractor.get_gradients()[-1] 
            
            target = featuremaps[-1]
            weights = torch.mean(grads_val, axis=(2, 3),keepdims=True) 
            cam = torch.zeros([target.shape[0],target.shape[2],target.shape[3]]).cuda()

            cam = F.relu((weights * target).mean(dim=1),inplace=True) 
            if is_testing:
                cam=torchvision.transforms.Resize(list(X.shape[2:]),antialias=True)(cam)
                cam=cam.detach()
                outputs=outputs.detach()
            # else:
            #     cam=util.normalize_image(cam)
            return cam, outputs    

        
    def train_map(model,tmodel,finalmlpmodel):

        epochloss=0
        y_hat, y_true,y_hat_logit,attmap_bool_selected_l,targets_att_bool_l = [], [], [], [],[]
        optimizer.zero_grad()
        model.train()
        tmodel.train()
        finalmlpmodel.train()
        torch.cuda.empty_cache()
        for batch_idx, (X, targets,att_targets, missflag_samples, missflag_masks, paths) in enumerate(train_Loader):
            batchsize=X.shape[0] if type(X)==torch.Tensor else X[0].shape[0]
            pmask_index= percent_mask if batchsize==args.train_batch else util.percent_mask(args.time_length,batchsize,args.anno_p)
            last_mask_index=last_only_mask if batchsize==args.train_batch else util.last_only_mask(args.time_length,batchsize)
                           
            targets_extended=targets.repeat(args.time_length)
            paths_cat=[]

            t_decay_l=torch.ones(args.time_length,1) 

            
            if args.dataset in ["solar"]:
                for i in paths: paths_cat+=i
                X,att_targets,missflag_masks = torch.cat(X).float(),torch.cat(att_targets).float(),(missflag_masks).t().flatten() 

                att_targets=att_targets[:,None,:,:]
            att_targets=torchvision.transforms.Resize((64,64),antialias=True)(att_targets)
            X, targets,att_targets = X.to(device), targets.to(device),att_targets.to(device)
            rec_loss=0
            masked_index=(int(1)-missflag_masks)*pmask_index 
            att_targets*=masked_index[:,None,None,None].to(device) 
            att_targets_full=att_targets.clone()
            if args.model in ["our","our-t"]: 
                """ annotation interpolation  """
                att_targets_recon=interpolation_model(att_targets,args.time_length,batchsize)
                att_targets_recon=util.normalize_image(att_targets_recon)
                rec_loss =torch.sum( torch.sum(criterion_l1(att_targets_recon,att_targets),(1,2,3))* masked_index.to(device) * (targets_extended).to(device) )

                att_targets_full[(1-masked_index).to(torch.bool)]=att_targets_recon[(1-masked_index).to(torch.bool)]
            att_maps, outputs = grad_cam.get_attention_map(X,batchsize,tmodel,finalmlpmodel, targets)
            if args.model in ["our","our-p"]:
                att_maps_orig=att_maps.reshape(args.time_length,-1,*att_maps.shape[-2:]).detach().cpu().clone()
                att_maps_orig=util.normalize_image_wholetime(att_maps_orig.numpy())
                decay_input=att_maps_orig.max(axis=(1,2,3))
                decay_lambda=decay_mlp(torch.tensor(decay_input[None,:],dtype=torch.float).to(device))
                for i in range(args.time_length):
                    t_decay_l[i,0]=torch.exp(-decay_lambda*(args.time_length-(i+1))) 
                t_decay_l=t_decay_l.repeat(1,batchsize).to(device)                
            att_maps=util.normalize_image(att_maps)
            
            task_loss = criterion(outputs, targets)

            att_maps_large=torchvision.transforms.Resize(list(att_targets.shape[-2:]),antialias=True)(att_maps)
            att_maps_bool = (att_maps_large > 0.5).cpu()
            targets_att_bool = (att_targets > 0)
            targets_att_bool=targets_att_bool.cpu()

            attmap_bool_selected_l.append(att_maps_bool[(masked_index==1 )& (targets_extended==1)])
            targets_att_bool_l.append(targets_att_bool[(masked_index==1 )& (targets_extended==1)])


            att_targets=torchvision.transforms.Resize(list(att_maps.shape[1:]),antialias=True)(att_targets)
            att_targets_full=torchvision.transforms.Resize(list(att_maps.shape[1:]),antialias=True)(att_targets_full)


                            
            if args.model in ["our","our-t"]:
                att_maps_binary = att_maps            
                selector=(targets_extended).to(device) 
                if args.model in ["our-t"]:
                    att_loss=criterion_l1(att_maps_binary,att_targets_full.squeeze_()) * selector[:,None,None]
                elif args.model=="our": 
                    att_loss=criterion_l1(att_maps_binary,att_targets_full.squeeze_()) * selector[:,None,None] * t_decay_l.flatten()[:,None,None]
                att_loss=torch.sum(att_loss,[1,2])
                att_loss = torch.sum(torch.relu( att_loss- args.eta))
            elif args.model=="our-p":
                selector=(targets_extended*masked_index).to(device)
                att_loss=criterion_l1(att_maps,att_targets_full.squeeze_()) * selector[:,None,None] * t_decay_l.flatten()[:,None,None]
                att_loss=torch.sum(att_loss,[1,2])
                att_loss = torch.sum(torch.relu( att_loss- args.eta))                 

            loss = task_loss + args.att_coef*att_loss + args.rec_coef*rec_loss
            if batch_idx==1:
                print(yellow(f"task_loss: {task_loss:.3f}, att_loss: {att_loss.item():.3f}, rec_loss: {rec_loss:.3f}"))

            optimizer.zero_grad()
            loss.backward()
            epochloss+=loss.detach()
            optimizer.step()
                        
            _, pred = outputs.topk(1, dim=1, largest=True, sorted=True)
            pred,targets,outputs=pred.cpu(),targets.cpu(),outputs.cpu()
            y_hat += list(pred.detach().numpy().reshape(-1))
            y_true += list(targets.detach().numpy().reshape(-1))
            y_hat_logit+=list(outputs.detach().numpy())

        attmap_bool_selected_l=torch.vstack(attmap_bool_selected_l)
        targets_att_bool_l=torch.vstack(targets_att_bool_l)        
        return epochloss.item()/len(train_Loader),y_hat, y_true,y_hat_logit,attmap_bool_selected_l,targets_att_bool_l
    def test_map(loader,model,tmodel,finalmlpmodel):

        epochloss=0
        y_hat, y_true,y_hat_logit,attmap_bool_selected_l,targets_att_bool_l = [], [], [], [],[]
        model.eval()
        # tmodel.eval()
        finalmlpmodel.eval()
        grad_cam = GradCam(model=model, feature_module=model[7], target_layer_names=["1"])
        for batch_idx, (X, targets,att_targets, missflag_samples, missflag_masks, paths) in enumerate(loader):
            batchsize=X.shape[0] if type(X)==torch.Tensor else X[0].shape[0]
            targets_extended=targets.repeat(args.time_length)
            paths_cat=[]
            if args.dataset in ["solar"]:
                for i in paths: paths_cat+=i
                X,att_targets,missflag_masks = torch.cat(X).float(),torch.cat(att_targets).float(),(missflag_masks).t().flatten() 
            X, targets,att_targets = X.to(device), targets.to(device),att_targets.to(device)
            
            att_maps, outputs = grad_cam.get_attention_map(X,batchsize,tmodel,finalmlpmodel, index=targets,is_testing=True)
            att_maps=att_maps.detach().cpu()       
            
            
            if args.loadmodel==True:
                attention_path=os.path.join("attention",args.dataset,args.model)
                if not os.path.exists(attention_path):
                    os.makedirs(attention_path,exist_ok=True)
                X_orig=X.reshape(args.time_length,-1,*X.shape[1:])
                att_maps_orig=att_maps.reshape(args.time_length,-1,*X.shape[2:]).clone()
                att_maps_orig=util.normalize_image_wholetime(att_maps_orig.numpy())
                for i in range(batchsize):
                    if args.dataset=="solar":
                        temppaths=[paths[j][i] for j in range(args.time_length)] 
                    if targets[i]==1:
                        util.show_cam_on_image(X_orig[:,i].cpu().detach().numpy(), att_maps_orig[:,i], temppaths,attention_path)
            
            task_loss = criterion(outputs, targets)
            epochloss+=task_loss.detach()
            
            att_maps=util.normalize_image(att_maps)
            att_maps_bool = (att_maps > 0.5)
            targets_att_bool = (att_targets > 0)
            targets_att_bool=targets_att_bool.cpu()
            attmap_bool_selected_l.append(att_maps_bool[(missflag_masks==0 )& (targets_extended==1)])
            targets_att_bool_l.append(targets_att_bool[(missflag_masks==0 )& (targets_extended==1)])
            
            
            _, pred = outputs.topk(1, dim=1, largest=True, sorted=True)
            pred,targets,outputs=pred.cpu(),targets.cpu(),outputs.cpu()
            y_hat += list(pred.detach().numpy().reshape(-1))
            y_true += list(targets.detach().numpy().reshape(-1))
            y_hat_logit+=list(outputs.detach().numpy())
        attmap_bool_selected_l=torch.vstack(attmap_bool_selected_l)
        targets_att_bool_l=torch.vstack(targets_att_bool_l)
        return epochloss.item()/len(loader),y_hat, y_true,y_hat_logit,attmap_bool_selected_l,targets_att_bool_l

    if args.loadmodel:
        try:
            suffix='Aug25-06:46:06'
            model.load_state_dict(torch.load(os.path.join("save",args.dataset,'model','best_model_'+suffix+'.pth')),strict=True)
            tmodel.load_state_dict(torch.load(os.path.join("save",args.dataset,'model','best_tmodel_'+suffix+'.pth')),strict=True)
            finalmlpmodel.load_state_dict(torch.load(os.path.join("save",args.dataset,'model','best_finalmlpmodel_'+suffix+'.pth')),strict=True)
        except OSError:
            pass
        test_loss, yhat_test, ytrue_test, yhatlogit_test,attmap_bool_selected_l_test,targets_att_bool_l_test = test_map(test_Loader,model,tmodel,finalmlpmodel)
        test_acc,test_recall,test_precision,test_Fscore,test_roc,test_iou,test_tss=util.calculate_map(yhat_test,ytrue_test,yhatlogit_test,attmap_bool_selected_l_test,targets_att_bool_l_test)
        util.print_1(0,'Test',test_loss,test_acc,test_recall,test_precision,test_Fscore,test_roc,iou=test_iou,tss=test_tss,color=blue)
                        
    else:
        """ 
        training start 
        """
        best_val_trigger = 1
        old_lr=1e3
        suffix="{}{}-{}:{}:{}".format(datetime.now().strftime("%h"),
                                        datetime.now().strftime("%d"),
                                        datetime.now().strftime("%H"),
                                        datetime.now().strftime("%M"),
                                        datetime.now().strftime("%S"))        
        if args.log: writer = SummaryWriter(os.path.join(tensorboard_dir,suffix))
        grad_cam = GradCam(model=model, feature_module=model[7], target_layer_names=["1"])
        for epoch in range(args.nepoch):
            train_loss,y_hat, y_true,y_hat_logit,attmap_bool_selected_l,targets_att_bool_l=train_map(model,tmodel,finalmlpmodel)
            
            train_acc,recall,precision,Fscore,roc,iou,tss=util.calculate_map(y_hat,y_true,y_hat_logit,attmap_bool_selected_l,targets_att_bool_l)
            try:util.record({"loss":train_loss,"acc":train_acc,"recall":recall,"precision":precision,"fscore":Fscore,"roc":roc,"tss":tss,"iou":iou},epoch,writer,"Train") 
            except: pass
            util.print_1(epoch,'Train',train_loss,train_acc,recall,precision,Fscore,roc,iou,tss)
            if epoch % args.test_per_round == 0:
                val_loss, yhat_val, ytrue_val, yhatlogit_val,attmap_bool_selected_l_val,targets_att_bool_l_val = test_map(val_Loader,model,tmodel,finalmlpmodel)
                test_loss, yhat_test, ytrue_test, yhatlogit_test,attmap_bool_selected_l_test,targets_att_bool_l_test = test_map(test_Loader,model,tmodel,finalmlpmodel)
                
                val_acc,val_recall,val_precision,val_Fscore,val_roc,val_iou,val_tss=util.calculate_map(yhat_val,ytrue_val,yhatlogit_val,attmap_bool_selected_l_val,targets_att_bool_l_val)
                try:util.record({"loss":val_loss,"acc":val_acc,"recall":val_recall,"precision":val_precision,"fscore":val_Fscore,"roc":val_roc,"iou":val_iou,"tss": val_tss},epoch,writer,"Val")
                except: pass
                util.print_1(epoch,'Val',val_loss,val_acc,val_recall,val_precision,val_Fscore,val_roc,iou=val_iou,tss=val_tss,color=blue) 
                test_acc,test_recall,test_precision,test_Fscore,test_roc,test_iou,test_tss=util.calculate_map(yhat_test,ytrue_test,yhatlogit_test,attmap_bool_selected_l_test,targets_att_bool_l_test)
                try:util.record({"loss":test_loss,"acc":test_acc,"recall":test_recall,"precision":test_precision,"fscore":test_Fscore,"roc":test_roc,"iou":test_iou,"tss":test_tss},epoch,writer,"Test")            
                except: pass
                util.print_1(epoch,'Test',test_loss,test_acc,test_recall,test_precision,test_Fscore,test_roc,iou=test_iou,tss=test_tss,color=blue)
                val_trigger=-val_Fscore
                if val_trigger < best_val_trigger:
                    best_val_trigger = val_trigger
                    best_model = copy.deepcopy(model)
                    best_tmodel=copy.deepcopy(tmodel)
                    best_finalmlpmodel=copy.deepcopy(finalmlpmodel)
                    best_info=[epoch,val_trigger]
            """ 
            update lr when epoch≥30
            """
            if epoch >= 30:
                lr = scheduler.optimizer.param_groups[0]['lr']
                if old_lr!=lr:
                    print(red('lr'), epoch, (lr), sep=', ')
                    old_lr=lr
                scheduler.step(val_trigger)        
        """
        use best model to get best model result 
        """
        val_loss, yhat_val, ytrue_val, yhat_logit_val,attmap_bool_selected_l_val,targets_att_bool_l_val  = test_map(val_Loader,best_model,best_tmodel,best_finalmlpmodel)
        test_loss, yhat_test, ytrue_test, yhat_logit_test,attmap_bool_selected_l_test,targets_att_bool_l_test = test_map(test_Loader,best_model,best_tmodel,best_finalmlpmodel)

        val_acc,val_recall,val_precision,val_Fscore,val_roc,val_iou,val_tss=util.calculate_map(yhat_val,ytrue_val,yhat_logit_val,attmap_bool_selected_l_val,targets_att_bool_l_val)
        util.print_1(best_info[0],'BestVal',val_loss,val_acc,val_recall,val_precision,val_Fscore,val_roc,iou=val_iou,tss=val_tss,color=blue)
        test_acc,test_recall,test_precision,test_Fscore,test_roc,test_iou,test_tss=util.calculate_map(yhat_test,ytrue_test,yhat_logit_test,attmap_bool_selected_l_test,targets_att_bool_l_test)
        util.print_1(best_info[0],'BestTest',test_loss,test_acc,test_recall,test_precision,test_Fscore,test_roc,iou=test_iou,tss=test_tss,color=blue)
                                                            
    if not args.loadmodel:
        """
        save training info and best result 
        """
        result_file=os.path.join(info_dir, suffix)
        with open(result_file, 'w') as f:
            print("Random Seed: ", Seed,file=f)
            print(f"acc  val : {val_acc:.3f}, Test : {test_acc:.3f}", file=f)
            print(f"recall  val : {val_recall:.3f}, Test : {test_recall:.3f}", file=f)
            print(f"precision  val : {val_precision:.3f}, Test : {test_precision:.3f}", file=f)
            print(f"Fscore  val : {val_Fscore:.3f}, Test : {test_Fscore:.3f}", file=f)
            print(f"roc  val : {val_roc:.3f}, Test : {test_roc:.3f}", file=f)
            print(f"iou  val : {val_iou:.3f}, Test : {test_iou:.3f}", file=f)
            print(f"tss  val : {val_tss:.3f}, Test : {test_tss:.3f}", file=f) 
            print(f"Best info: {best_info}", file=f)
            for i in [[a,getattr(args, a)] for a in args.__dict__]:
                print(i,sep='\n',file=f)
        torch.save(best_model.state_dict(), os.path.join(model_dir,'best_model'+"_"+suffix+'.pth') )
        torch.save(best_tmodel.state_dict(), os.path.join(model_dir,'best_tmodel'+"_"+suffix+'.pth') )
        torch.save(best_finalmlpmodel.state_dict(), os.path.join(model_dir,'best_finalmlpmodel'+"_"+suffix+'.pth') )
    print("done")

if __name__ == '__main__':
    args = get_args()
    """
    build dir 
    """
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir,exist_ok=True)
    tensorboard_dir=os.path.join(args.save_dir,'log')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir,exist_ok=True)
    model_dir=os.path.join(args.save_dir,'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir,exist_ok=True)    
    info_dir=os.path.join(args.save_dir,'info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir,exist_ok=True)      

    Seed = 0
    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed)     
    test_ratio=0.2
    print("data splitting Random Seed: ", Seed)
    if args.dataset in ['solar']:
        train_ds,val_ds,test_ds=get_solar_dataset(args.data_dir,args.time_length,Seed,test_ratio=test_ratio)         
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=args.train_batch, shuffle=False,pin_memory=True) 
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.test_batch, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds,batch_size=args.test_batch, shuffle=False,pin_memory=True)
    """
    set model seed 
    """
    Seed = args.man_seed if args.manualSeed else random.randint(1, 10000)
    # Seed=3407
    print("Random Seed: ", Seed)
    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed)    
    main(args,train_loader,val_loader,test_loader)
    
'''
Training code for MRBrainS18 datasets segmentation
Written by Whalechen
'''

from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset 
from datasets.cervix import Cervix30Dataset
from evaluate_during_training import Evaluator
from model import generate_model
from training_evaluation import Training_Evaluator
from early_stopping import Early_Stopping

import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
import os

#from medcam import medcam
    

def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_name, sets):

    evaluator = Evaluator(sets)
    early_stopping = Early_Stopping()
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    print(f'Current setting is: {sets}\n\n')
    # ignore_index (int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient.
    # When size_average is True, the loss is averaged over non-ignored targets.
    # Note that ignore_index is only applicable when the target contains class indices.
    '''
    class_weights = torch.tensor([0.66, 0.34])
    loss_class = nn.CrossEntropyLoss(weight=class_weights)
    '''
    loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    loss_class = nn.CrossEntropyLoss()
   
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()
        loss_class = loss_class.cuda()
        
    # Eval first for Baseline comparision
    model.eval()
    df_results = evaluator.evaluate_and_store(model, sets, save_name, 0)
    
    status_saved = False
    model.train()
    train_time_sp = time.time()
    
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))
        
        for batch_id, batch_data in enumerate(data_loader):
            
            #model.classifier.weights.requires_grad_()
            
            batch_id_sp = epoch * batches_per_epoch
            
            volumes, labels = batch_data
            #print(f"Labels: {labels}")

            if not sets.no_cuda: 
                volumes = volumes.cuda()

            optimizer.zero_grad()
            
            if sets.new_layer_names[0] == 'classifier':

                labels = torch.stack(labels, axis=1)
                out_labels = model(volumes)

                if not sets.no_cuda:
                    out_labels = out_labels.cuda()
                    labels = labels.cuda()
                
                #print(f"Labels: {labels}, Out Labels: {out_labels}")
                loss_value = loss_class(out_labels, labels)
                loss = loss_value
            
            elif sets.new_layer_names[0] == 'conv_seg':

                out_masks = model(volumes)
                [n, _, d, h, w] = out_masks.shape
                new_label_masks = np.zeros([n, d, h, w])
                for label_id in range(n):
                    
                    label_mask = labels[label_id]
                    [ori_c, ori_d, ori_h, ori_w] = label_mask.shape 
                    label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                    scale = [d*1.0/ori_d, h*1.0/ori_h, w*1.0/ori_w]
                    label_mask = ndimage.zoom(label_mask, scale, order=0)
                    new_label_masks[label_id] = label_mask

                new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
                if not sets.no_cuda:
                    new_label_masks = new_label_masks.cuda()

                # calculating loss
                loss_value_seg = loss_seg(out_masks, new_label_masks)
                loss = loss_value_seg

            loss.backward()
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info('Batch: {}-{} ({}), loss = {:.3f}, avg_batch_time = {:.3f}'\
                            .format(epoch, batch_id, batch_id_sp, loss.item(), avg_batch_time))
        
        # Evaluate model
        model.eval()
        df_results = evaluator.evaluate_and_store(model, sets, save_name, epoch)
        
        save_epoch = early_stopping.early_stopping(df_results, epoch)
        
        if save_epoch == True:
            model_save_path = '{}/epoch_{}_k_{}.pth.tar'.format(sets.trails_folder + '/' + save_name, epoch, k)
            model_save_dir = os.path.dirname(model_save_path)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            previous_files = [f for f in os.listdir(model_save_dir) if f.endswith(str(k) + '.pth.tar')]
            if len(previous_files) > 0:
                for file in previous_files:
                    os.remove(model_save_dir + '/' + file)
            log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
            torch.save({'ecpoch': epoch,'batch_id': batch_id,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, model_save_path)
            
            evaluator.generate_attention_map(model, sets, epoch)
            status_saved = True
            
        model.train()
        scheduler.step()
    
    if status_saved == False:
        model_save_path = '{}/epoch_{}_k_{}.pth.tar'.format(sets.trails_folder + '/' + save_name, epoch, k)
        model_save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        previous_files = [f for f in os.listdir(model_save_dir) if f.endswith(str(k) + '.pth.tar')]
        if len(previous_files) > 0:
            for file in previous_files:
                os.remove(model_save_dir + '/' + file)
        torch.save({'epoch': epoch,'batch_id': batch_id,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, model_save_path)
        evaluator.generate_attention_map(model, sets, epoch)
        
    print('Finished training')
    # make evaluation of training
    
    if sets.ci_test:
        print("Test only")
        exit()

if __name__ == '__main__':
    sets = parse_opts() 
    
    if sets.ci_test:
        sets.img_list = './toy_data/test_ci.txt' 
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 14
        sets.input_H = 28
        sets.input_W = 28
        sets.new_layer_names = classifier
    
    '''
    pretrained_name = sets.pretrain_path.split('_')
    pretrained_name = pretrained_name[1].split('.')
    model_depth = int(pretrained_name[0])
    sets.model_depth = model_depth
    '''
    
    if (sets.model_depth == 18 | sets.model_depth == 34):
        sets.resnet_shortcut = 'A'
    else:
        sets.resnet_shortcut = 'B'
        
        pretrain_path = sets.pretrain_path
    
    # 5-fold-Crossvalidation
    for k in range(5):
        # getting model
        sets.k = "K_" + str(k)
        print(f"K: {sets.k}")
        
        if sets.no_cuda:
            sets.pin_memory = False
        else:
            sets.pin_memory = True
        
        if (sets.double_pretrained == True):
            files = os.listdir(pretrain_path)
            path = [file for file in files if 'k_' + str(k) in file]
            print(path)
            sets.pretrain_path = pretrain_path + '/' + path[0]
        
        img_train_list = sets.split_root + sets.image_clip_type + '/' + sets.aug_type + '/' + sets.k + '/train.txt'
        training_dataset = Cervix30Dataset(img_train_list, sets)
        print(f"training_dataset: {training_dataset}")
        data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=sets.pin_memory)
        
        torch.manual_seed(sets.manual_seed)
        model, parameters = generate_model(sets) 
        print (model)
        # optimizer
        if sets.ci_test:
            params = [{'params': parameters, 'lr': sets.learning_rate}]
        elif sets.pretrain_path == '':
            params = [{'params': parameters, 'lr': sets.learning_rate}]
        else:
            # backbone parameter get adjusted only slightly and can be completly fixed via the settings
            params = [{ 'params': parameters['base_parameters'], 'lr': sets.learning_rate/100 }, 
                    { 'params': parameters['new_parameters'], 'lr': sets.learning_rate}]

        if sets.opimization == 'Adam':
            optimizer = torch.optim.Adam(params, lr=sets.learning_rate, weight_decay=sets.weight_decay)
            log.info('Using Adam Optimizer')
        else:
            optimizer = torch.optim.SGD(params, momentum=0.025, weight_decay=sets.weight_decay)

        # gamma – Multiplicative factor of learning rate decay
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, verbose=True)

        # train from resume
        if sets.resume_path:
            if os.path.isfile(sets.resume_path):
                print("=> loading checkpoint '{}'".format(sets.resume_path))
                checkpoint = torch.load(sets.resume_path)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                  .format(sets.resume_path, checkpoint['epoch']))

        save_name = sets.image_clip_type + '_md' + str(sets.model_depth) + '_bs' + str(sets.batch_size) + '_fl' + str(sets.fixed_layers) + '_lr' + str(sets.learning_rate) + '_o' + sets.opimization + '_d' + str(sets.input_D) + '_wh' + str(sets.input_W) + '_wd' + str(sets.weight_decay) + '_dr' + str(sets.dropout_rate) + '_a' + sets.aug_type
        if not os.path.exists(sets.save_folder + '/' + save_name):
            os.makedirs(sets.save_folder + '/' + save_name)
        # training
        train(data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_name=save_name, sets=sets)
    
    # Learning Evaluation
    train_evaluator = Training_Evaluator(sets.save_folder + '/' + save_name + '/', sets.name)
    train_evaluator.cross_validation_evaluation()
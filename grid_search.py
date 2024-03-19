from setting import parse_opts 
from evaluate_during_training import Evaluator
from model import generate_model
from training_evaluation import Training_Evaluator
from datasets.cervix import Cervix30Dataset


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

def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_name, sets):

    evaluator = Evaluator(sets)
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    print(f'Current setting is: {sets}\n\n')
    
    '''
    class_weights = torch.tensor([0.66, 0.34])
    loss_class = nn.CrossEntropyLoss(weight=class_weights)
    '''
    loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    loss_class = nn.CrossEntropyLoss()
   
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()
        loss_class = loss_class.cuda()
    
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
            
            

            labels = torch.stack(labels, axis=1)
            out_labels = model(volumes)

            if not sets.no_cuda:
                out_labels = out_labels.cuda()
                labels = labels.cuda()

            #print(f"Labels: {labels}, Out Labels: {out_labels}")
            loss_value = loss_class(out_labels, labels)
            loss = loss_value

            loss.backward()
            optimizer.step()
        
        # Evaluate model
        model.eval()
        df_results = evaluator.evaluate_and_store(model, sets, save_name, epoch)
            
        model.train()
        scheduler.step()
    
    print('Finished training')
    # make evaluation of training

if __name__ == '__main__':

    learning_rates = [0.001, 0.003, 0.005]
    weight_decays = [0.001, 0.003, 0.005]
    batch_sizes = [4, 5, 6, ]
    model_depths = [200]
    pretrain_paths = ['resnet_200.pth']


    # Grid search loop
    
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:


                sets = parse_opts()
                sets.save_folder = './gridsearch'
                sets.n_epochs = 50
                sets.learning_rate = learning_rate
                sets.batch_size = batch_size
                sets.model_depth = model_depths[0]
                sets.pretrain_path = 'pretrain/' + pretrain_paths[0]
                sets.weight_decay = weight_decay
                sets.image_clip_type = '2_filtTrue_distorted_ahe_org_randFalse'
                sets.gpu_id = [0]

                if (sets.model_depth == 18 | sets.model_depth == 34):
                    sets.resnet_shortcut = 'A'
                else:
                    sets.resnet_shortcut = 'B'

                sets.name = "Gridsearch 200"

                # 5-fold-Crossvalidation
                for k in range(2):
                    # getting model
                    sets.k = "K_" + str(k)
                    print(f"K: {sets.k}")

                    if sets.no_cuda:
                        sets.pin_memory = False
                    else:
                        sets.pin_memory = True

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

                    save_name = sets.image_clip_type + '_md' + str(sets.model_depth) + '_bs' + str(sets.batch_size) + '_fl' + str(sets.fixed_layers) + '_lr' + str(sets.learning_rate) + '_wd' + str(sets.weight_decay)

                    if not os.path.exists(sets.save_folder + '/' + save_name):
                        os.makedirs(sets.save_folder + '/' + save_name)
                    # training
                    train(data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_name=save_name, sets=sets)
from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset
from datasets.cervix import Cervix30Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from scipy import ndimage
import nibabel as nib
import sys
import os
from utils.file_process import load_lines
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
from medcam import medcam
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import fbeta_score

class Evaluator():
    
    def __init__(self, sets):
        self.eval_results = []
        self.train_results = []
        self.eval_probs = []
        self.train_probs = []
        
        img_train_list = sets.split_root + sets.image_clip_type + '/' + sets.aug_type + '/' + sets.k  + '/train.txt'
        img_test_list = sets.split_root + sets.image_clip_type + '/' + sets.aug_type + '/' + sets.k  + '/test.txt'

        self.train_dataset = Cervix30Dataset(img_train_list, sets)
        self.eval_dataset = Cervix30Dataset(img_test_list, sets)
        
    def store_loss(self, loss):
        self.losses.append(loss)       

    def test_class (self, data_loader, model, sets, mode):
        '''
        class_weights = torch.tensor([0.66, 0.34])
        loss_class = nn.CrossEntropyLoss(weight=class_weights)
        '''
        loss_class = nn.CrossEntropyLoss()
        m = nn.Softmax(dim=1)

        new_labels = []
        l = []
        p = []
        p_tensors = []
        for batch_id, batch_data in enumerate(data_loader):
            volumes, labels = batch_data
            labels = torch.stack(labels, axis=1)

            if not sets.no_cuda:
                volumes = volumes.cuda()
                labels = labels.cuda()
                loss_class = loss_class.cuda()
                
            with torch.no_grad():
                probs = model(volumes)
                #Softmax
                probs_soft = m(probs)

            # nimm das 1.Element d.h. W für 0
            probs_float = probs_soft[0,0].item()
            p.append(1 - round(probs_float, 3))
            if probs_float > 0.5:
                new_labels.append(0.0)
            else:
                new_labels.append(1.0)
            #print(f"Test Class Out: {new_labels}")
            p_tensors.append(probs_soft[0])
            l.append(labels[0])
            
        l = torch.stack(l)
        p_tensors = torch.stack(p_tensors)
        loss = loss_class(p_tensors, l)
        print(f"Loss: {loss}")
       
        return new_labels, p, loss.item()

    def probs_to_dataframe(self, probs):

        probs = np.vstack(probs)    

        # Get the shape of the array
        rows, columns, depth = probs.shape

        # Create a list of column names
        col_names = ['Label', 'Predicted Probability']

        # Create an empty DataFrame with the column names
        df = pd.DataFrame(columns=col_names)

        for j in range(probs.shape[0]):
            for i in range(probs.shape[2]):
                row = probs[j, :, i].flatten()
                df.loc[(depth*j)+i] = row
        return df

    def wrap_results(self):
        
        columns = ['Accuracy', 'Balanced Accuracy', 'DOR', 'Sensitivity', 'Specificity', 'AUC', 'Loss', 'Std']
        '''
        df_results = pd.DataFrame({f'Validation {col}': [x[i] for x in self.eval_results] for i, col in enumerate(columns)} +
                            {f'Train {col}': [x[i] for x in self.train_results] for i, col in enumerate(columns, start=8)})
        '''
        
        df_results = pd.DataFrame({'Validation Accuracy': [x[0] for x in self.eval_results],
                            'Validation Balanced Accuracy': [x[1] for x in self.eval_results],
                            'Validation DOR' :[x[2] for x in self.eval_results],
                            'Validation Sensitivity': [x[3] for x in self.eval_results],
                            'Validation Specificity': [x[4] for x in self.eval_results],
                            'Validation AUC': [x[5] for x in self.eval_results],
                            'Validation Loss': [x[6] for x in self.eval_results],
                            'Validation Std': [x[7] for x in self.eval_results],
                            'Train Accuracy': [x[0] for x in self.train_results],
                            'Train Balanced Accuracy': [x[1] for x in self.train_results],
                            'Train DOR': [x[2] for x in self.train_results],
                            'Train Sensitivity': [x[3] for x in self.train_results],
                            'Train Specificity': [x[4] for x in self.train_results],
                            'Train AUC': [x[5] for x in self.train_results],
                            'Train Loss': [x[6] for x in self.train_results],
                            'Train Std': [x[7] for x in self.train_results]})
        
        return df_results

    def evaluate_and_store(self, model, sets, save_name, epoch):
        
        save_folder = sets.save_folder + '/' + save_name + '/'

        scores, probs = self.evaluate(model, sets, 'eval train')
        self.train_results.append(scores)
        self.train_probs.append([probs])
        scores, probs = self.evaluate(model, sets, 'eval test')
        self.eval_results.append(scores)
        self.eval_probs.append([probs])

        df_results = self.wrap_results()
        df_probs_eval = self.probs_to_dataframe(self.eval_probs)
        df_probs_train = self.probs_to_dataframe(self.train_probs)
        df_results.to_csv(save_folder + 'Acc' + save_name + sets.k + '.csv', index=False)
        df_probs_eval.to_csv(save_folder + 'ProbsEval' + save_name + sets.k + '.csv', index= False)
        df_probs_train.to_csv(save_folder +'ProbsTrain' + save_name + sets.k + '.csv', index= False)
        
        return df_results

    def evaluate(self, model, sets, mode):  

        if sets.dataset == 'Cervix':
            if mode == 'eval train':
                print('Train')
                img_list = sets.split_root + sets.image_clip_type + '/' + sets.aug_type + '/' + sets.k + '/train.txt'
                testing_data = self.train_dataset
            else:
                print('Test')
                img_list = sets.split_root + sets.image_clip_type + '/' + sets.aug_type + '/' + sets.k  + '/test.txt'
                testing_data = self.eval_dataset
            #testing_data = Cervix30Dataset(img_list, sets)
            labels = [info.split("\t")[1] for info in load_lines(img_list)]

            # Batch_size for evaluation is always 1, shuffel needs to be false
            data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

            # label=lambda x: 0.5 < x
            # model.module.layer1[0].conv1
            #model = medcam.inject(model, output_dir="attention_maps/" + sets.name + "/" + sets.k , save_maps=True, backend='gcam', layer='auto', label='best')
            #model.disable_medcam()
            
            pred, probs, loss = self.test_class(data_loader, model, sets, mode)

            #print(f"Probabilities: {probs}")
            #print(f"Prediction: {pred}")
            #print(f"Labels of all: {labels}")

            tp, tn, fp, fn = 0, 0, 0, 0

            for i in range (len(pred)):
                if (pred[i] == 1.0):
                    if (labels[i] == '1.0'):
                        tp = tp + 1
                    else:
                        fp = fp + 1
                else:
                    if (labels[i] == '1.0'):
                        fn = fn + 1
                    else:
                        tn = tn + 1
            accuracy = (tp + tn) / len(labels)
            labels = list(map(float, labels))
            b_acc = balanced_accuracy_score(labels, pred)
            
            if (fp != 0) & (fn != 0):
                dor = (tp * tn) / (fp * fn)
            else:
                dor = (tp * tn) / 1e-10
            
            if (tp != 0 ):
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                sensitivity = tp / (tp + fn)
            else:
                sensitivity = 0
            if (tn != 0):
                specificity = tn / (fp + tn)
            else:
                specificity = 0
            
            labels = np.array(labels)
            probs = np.array(probs)
            probs_result = np.vstack((labels, probs))
            
            # AUC berechnen
            auc = roc_auc_score(labels, probs)
            
            std_dev = np.std(probs)
            print(std_dev)
            
            print(f'Accuracy: {accuracy} Balanced Accuracy: {b_acc} Diagnostic Ods Ratio: {dor} Specificity: {specificity} Sensitivity: {sensitivity} AUC: {auc}')

            
            return [accuracy, b_acc, dor, sensitivity, specificity, auc, loss, std_dev], probs_result

        # TODO: Muss angegepasst werden
        elif sets.dataset == 'Brain':
            # data tensor
            testing_data = BrainS18Dataset(sets.data_root, 'data/Brain/test.txt', sets)
            data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

            # testing
            img_names = [info.split(" ")[0] for info in load_lines(sets.img_test_list)]
            masks = test_seg(data_loader, model, img_names, sets)

            # evaluation: calculate dice 
            label_names = [info.split(" ")[1] for info in load_lines(sets.img_test_list)]
            Nimg = len(label_names)
            dices = np.zeros([Nimg, sets.n_seg_classes])
            for idx in range(Nimg):
                label = nib.load(os.path.join(sets.data_root, label_names[idx]))
                # von get_data()
                label = label.get_fdata()
                dices[idx, :] = seg_eval(masks[idx], label, range(sets.n_seg_classes))

            # print result
            for idx in range(1, sets.n_seg_classes):
                mean_dice_per_task = np.mean(dices[:, idx])
                print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))

    def seg_eval(pred, label, clss):
        """
        calculate the dice between prediction and ground truth
        input:
            pred: predicted mask
            label: groud truth
            clss: eg. [0, 1] for binary class
        """
        Ncls = len(clss)
        dices = np.zeros(Ncls)
        [depth, height, width] = pred.shape
        for idx, cls in enumerate(clss):
            # binary map
            pred_cls = np.zeros([depth, height, width])
            pred_cls[np.where(pred == cls)] = 1
            label_cls = np.zeros([depth, height, width])
            label_cls[np.where(label == cls)] = 1

            # cal the inter & conv
            s = pred_cls + label_cls
            inter = len(np.where(s >= 2)[0])
            conv = len(np.where(s >= 1)[0]) + inter
            try:
                dice = 2.0 * inter / conv
            except:
                print("conv is zeros when dice = 2.0 * inter / conv")
                dice = -1

            dices[idx] = dice

        return dices

    # Segmentierung: Gibt Vorhersage-Maske zurück
    def test_seg(data_loader, model, img_names, sets):
        masks = []
        model.eval()
        for batch_id, batch_data in enumerate(data_loader):
            volume = batch_data
            if not sets.no_cuda:
                volume = volume.cuda()
            with torch.no_grad():
                probs = model(volume)
                probs = F.softmax(probs, dim=1)

            # resize mask to original size
            [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
            data = nib.load(os.path.join(sets.data_root, img_names[batch_id]))
            data = data.get_fdata()
            [depth, height, width] = data.shape
            mask = probs[0]
            scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]

            mask = ndimage.zoom(mask.cpu(), scale, order=1)

            mask = np.argmax(mask, axis=0)

            masks.append(mask)

        return masks

    def generate_attention_map(self, model, sets, epoch):
        
        # label=lambda x: 0.5 < x
        #model = medcam.inject(model, output_dir="gradcam/" + sets.name + "/" + sets.k, save_maps=True, backend='gcam', layer='auto', label='best', data_shape=[20, 160, 160])
        #scores, probs = self.evaluate(model, sets, 'eval test')
        #model.disable_medcam()

        model2 = medcam.inject(model, output_dir="gbp/" + sets.name + "/" + sets.k, save_maps=True, backend='gbp', layer='auto', label='best')
        scores, probs = self.evaluate(model2, sets, 'eval test')
        model2.disable_medcam()
        
        model3 = medcam.inject(model, output_dir="ggcampp/" + sets.name + "/" + sets.k, save_maps=True, backend='gcampp', layer='auto', label='best')
        scores, probs = self.evaluate(model3, sets, 'eval test')
        model3.disable_medcam()
        
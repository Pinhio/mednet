from setting import parse_opts 
from data_loading.cervix import Cervix30Dataset
from model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import sys
import os
from utils.file_process import load_lines
import numpy as np


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

def test(data_loader, model, img_names, sets):
    masks = []
    model.eval() # for testing 
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            probs = model(volume, 'conv_seg')
            probs = F.softmax(probs, dim=1)
            print("Probs erhalten")
        # resize mask to original size
        [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
        print("load images")
        data = nib.load(img_names[batch_id])
        data = data.get_fdata()
        [depth, height, width] = data.shape
        mask = probs[0]
        scale = [1, depth*1.0/mask_d, height*1.0/mask_h, width*1.0/mask_w]
        mask = mask.cpu()
        print("zoom")
        mask = ndimage.zoom(mask, scale, order=1)
        print("zoomed")
        mask = np.argmax(mask, axis=0)
        print("arg max")
        masks.append(mask)
 
    return masks


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.phase = 'test'
    output_dir = 'feature_maps/'
    data_root = ''
    img_list = 'data/Cervix/test.txt'
    
    # getting model
    #checkpoint = torch.load(sets.resume_path, map_location=torch.device('cpu'))
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'], strict=False)

    # data tensor
    testing_data = Cervix30Dataset(img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    print("Data loaded")
    # testing
    
    img_names = [info.split("\t")[0] for info in load_lines(img_list)]
    #img_names, labels = self.img_list[idx].split("\t") 
    masks = test(data_loader, net, img_names, sets)
    print("Masks returned")
    
    for i in range (0,len(masks)):
        print("Iterate " + str(i))
        new_image = nib.Nifti1Image(masks[i], affine= None)
        nib.save(new_image, output_dir + str(i) + ".nii.gz")
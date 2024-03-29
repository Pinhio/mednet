'''
Configs for training & testing
Written by Whalechen
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split_root',
        default='/work/users/jn137pgao/tl_data/Cervix/Train_Test_Split/',
        type=str,
        help='Root directory path of Train Test Split')
    parser.add_argument(
        '--dataset',
        default='Cervix',
        type=str,
        help='Dataset')
    # only used for Brain-Dataset
    parser.add_argument(
        '--img_train_list',
        default='./data/Cervix/',
        type=str,
        help='Path for trainings-image list file')
    # only used for Brain-Dataset
    parser.add_argument(
        '--img_test_list',
        default='./data/Cervix/test.txt',
        type=str,
        help='Path for test-image list file')
    parser.add_argument(
        '--image_clip_type', 
        default='2_filtTrue_min_cut_ahe_org_randTrue_20',
        type=str,
        help='Clipping of the image')
    parser.add_argument(
        '--aug_type',
        default='Aug',
        type=str,
        help='Augmentation and Balanced Types: No_Aug, Aug, Balanced, Aug_Blanced')
    parser.add_argument(
        '--k',
        default='K_2',
        type=str,
        help='Cross-Validation Split: K_0 - K_4')
    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help="Number of classes"
    )
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.02,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--weight_decay',
        default=0.003,
        type=float,
        help='Optimizer weight decay')
    parser.add_argument(
        '--dropout_rate',
        default=0.1,
        type=float,
        help='Dropout rate for classification layer')
    parser.add_argument(
        '--num_workers',
        default=2,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size',
        default=4,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--fixed_layers',
        default=False,
        type=bool,
        help='Backbone Layers fixed or not')
    parser.add_argument(
        '--opimization', #SGD o. Adam
        default='Adam',
        type=str,
        help='Optimization of Traingsparameter')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test')
    parser.add_argument(
        '--save_intervals',
        default=10,
        type=int,
        help='Interation for saving model')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--input_D',
    default=27,
        type=int,
        help='Input size of depth')
    parser.add_argument(
        '--input_H',
        default=216,
        type=int,
        help='Input size of height')
    parser.add_argument(
        '--input_W',
        default=216,
        type=int,
        help='Input size of width')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Path for resume model.'
    )
    parser.add_argument(
        '--pretrain_path',
        # 'pretrain/resnet_10.pth','pretrain/resnet_34.pth', 'pretrain/resnet_50.pth', 'pretrain/resnet_101.pth', 'pretrain/resnet_152.pth', 'pretrain/resnet200.pth',
        default='/work/users/jn137pgao/pretrained/resnet_200.pth',
        type=str,
        help='Path for pretrained model.'
    )
    parser.add_argument(
        '--double_pretrained',
        default=False,
        type=bool,
        help='if pretrained on onwn data')
    parser.add_argument(
        '--new_layer_names',
        # currently implemented: conv_seg, classifier
        default=['classifier'],
        type=list,
        help='New layer except for backbone')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=200,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101 | 152 | 200)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--name',
        default='Selective Cut, 200 layers',
        type=str,
        help='Name used for storing results and plots')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--ci_test', action='store_true', help='If true, ci testing is used.')
    parser.add_argument('--save_folder',default = "/work/users/jn137pgao/tl_data/results/",type=str, help='path for output')
    parser.add_argument('--trails_folder',default = "/work/users/jn137pgao/trails",type=str, help='path for output')

    parser.add_argument('--dataset_name', default='forgot_smth', help='name of input dataset for label creation')
    
    args = parser.parse_args()
    
    return args

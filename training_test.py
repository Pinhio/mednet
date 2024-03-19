from setting import parse_opts 
import pandas as pd
import random
import os


def save_paths(save_dir, df):
    
    df = df.reset_index(drop=True)
        
    with open(save_dir, 'w') as f:
        for i in range(df.shape[0]):
            f.write(str(df.loc[i, 'path']) + '\t' + str(df.loc[i, 'label']) + '\n')
            
def create_split(positive_samples, negative_samples, k, i, double_negatives=False):
    
    p_i = positive_samples.shape[0] / k
    n_i = negative_samples.shape[0] / k
    
    positive_samples = positive_samples.reset_index(drop=True)
    negative_samples = negative_samples.reset_index(drop=True)
    
    training_samples = pd.DataFrame(columns=['path', 'label'])
    test_samples = pd.DataFrame(columns=['path', 'label'])
    for j in range(positive_samples.shape[0]):
        if (j >= round(i*p_i)) & (j < ((i+1) * round(p_i))):
            path, label = positive_samples.iloc[j]
            test_samples = pd.concat([test_samples, positive_samples.iloc[j: j+1]], ignore_index=True)
        else:
            training_samples = pd.concat([training_samples, positive_samples.iloc[j:j+1]], ignore_index=True)       
    for j in range(negative_samples.shape[0]):
        if ((j >= round(i*n_i)) & (j < ((i+1) * round(n_i)))):
            test_samples = pd.concat([test_samples, negative_samples.iloc[j:j+1]], ignore_index=True)
        else:
            training_samples = pd.concat([training_samples, negative_samples.iloc[j:j+1]], ignore_index=True)
            if double_negatives == True:
                training_samples = pd.concat([training_samples, negative_samples.iloc[j:j+1]], ignore_index=True)

    return training_samples, test_samples

def get_aug_training(training_samples:pd.DataFrame, aug_list:list, labels:pd.DataFrame, image_root:str, k:int, max_k:int=5) -> pd.DataFrame:
    ''' Get a k-th pairs of path and label for augmentations.
    '''
    k_th_list = aug_list[k::max_k] # get every k-th item from the augmentation list
    print(k_th_list)
    
    for aug in k_th_list:
        patient = aug.split('_')[0]
        label = label = labels.loc[str(patient)].label
        path = image_root + '/' + aug
        training_samples.loc[len(training_samples)] = [path, label]             
    return training_samples

def get_save_dir(save_root, clipping_name, aug, i):
    save_dir = save_root + '/Train_Test_Split/' + clipping_name + '/' + aug + '/K_' + i
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

# creates imagelist for training and testdata as txt-files with datapath to each image and its label
# for different imageclippings
# for different augmentation / balanceing options: No Augmentation, No Augmentation negativ samples are doubled, With Augmentation, With Augmentation and negativ samples are doubles 
# for 5-fold cross-validation
# folder structure: data/Cervix/Train_Test_Split/<imageclipping>/<Aug/DoubleNegative>/<CrossValidationNr>/
def create_train_test_split(data_root, save_root, labels, seed, k=5):
    
    # list all files in directory
    
    #path_to_first_patient = data_root + '/images/rad8'

    image_root = data_root + 'images'
    file_list = os.listdir(image_root)
    real_list = [elem for elem in file_list if not 'aug' in elem]
    aug_list = [elem for elem in file_list if elem not in real_list]


    random.seed(seed) # added seeding
    random.shuffle(real_list)
    random.shuffle(aug_list)
    
    positive_samples = pd.DataFrame(columns=['path', 'label'])
    negative_samples = pd.DataFrame(columns=['path', 'label'])

    for real in real_list:
        #_, clipping_name, _ = clipping.split('.') 

        # dir name für file-struktur
        clipping_name = 'my_test'

        file_path = image_root + '/' + real
        if os.path.exists(file_path):

            patient = real.split('_')[0] # radXX
            label = labels.loc[str(patient)].label

            if label == 0.0:
                negative_samples.loc[len(negative_samples)] = [file_path, label]
            else:
                positive_samples.loc[len(positive_samples)] = [file_path, label]

    print(positive_samples.head())
    print(negative_samples.head())
        
    for i in range(k):
        # Split into Training and Testdata
        training_samples, test_samples = create_split(positive_samples, negative_samples, k, i)

        save_dir = get_save_dir(save_root, clipping_name, 'No_Aug', str(i))
        save_paths(save_dir + '/train.txt', training_samples)
        save_paths(save_dir + '/test.txt', test_samples)
            
        # Add Augmentation to Trainingdataset
        training_aug_samples = get_aug_training(training_samples, aug_list, labels, image_root, i)
        save_dir = get_save_dir(save_root, clipping_name, 'Aug', str(i))
        save_paths(save_dir + '/train.txt', training_aug_samples)
        save_paths(save_dir + '/test.txt', test_samples)
        
        # Double Negative samples in Trainingsdataset
        training_balanced, test_samples = create_split(positive_samples, negative_samples, k, i, double_negatives=True)

        save_dir = get_save_dir(save_root, clipping_name, 'Balanced', str(i))
        save_paths(save_dir + '/train.txt', training_balanced)
        save_paths(save_dir + '/test.txt', test_samples)
            
        # Add Augmentation to the Double Negative Trainingsdataset
        training_balanced_aug = get_aug_training(training_balanced, aug_list, labels, image_root, i)
        save_dir = get_save_dir(save_root, clipping_name, 'Aug_Balanced', str(i))
        save_paths(save_dir + '/train.txt', training_balanced_aug)
        save_paths(save_dir + '/test.txt', test_samples)
        

if __name__ == '__main__':
                  
    sets = parse_opts()

    if sets.dataset == 'Cervix':
        #data_root = '/work/users/my814hiky/images' #TODO: work dir on SC
        data_root = 'data/' #TODO: work dir on SC
        save_root = 'data/Cervix'
        labels = pd.read_csv(data_root + 'labels.csv', sep=';')
        labels.set_index("id", inplace = True)
        seed = 1
          
        create_train_test_split(data_root, save_root, labels, seed)
                  
    else:
        print("Not defined for other Dataset yet")

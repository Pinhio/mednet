import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

class Training_Evaluator():
    
    def __init__(self, data_path, name):
        self.data_path = data_path
        self.clipping = name
        self.results = pd.DataFrame(columns =['Validation Accuracy', 'Validation Balanced Accuracy', 'Validation DOR', 'Validation Sensitivity', 'Validation Specificity', 'Validation AUC', 'Train Accuracy', 'Train Balanced Accuracy', 'Train DOR', 'Train Sensitivity', 'Train Specificity',
                                   'Train AUC','Epoch'])
    
        self.files_accuracy = [f for f in os.listdir(data_path) if f.startswith('Ac')]
        self.files_t_outcome = [f for f in os.listdir(data_path) if f.startswith('ProbsT')]
        self.files_v_outcome = [f for f in os.listdir(data_path) if f.startswith('ProbsE')]
        self.files_t_loss = [f for f in os.listdir(data_path) if f.startswith('L')]
        self.files_v_loss = [f for f in os.listdir(data_path) if f.startswith('E')]
    
    def plot_loss(self, v_loss, t_loss, epochs, title, save_name):

        # plot Avarage Loss per Fold 
        plt.figure(figsize=(10,5))
        plt.plot(t_loss, label='Training Loss')
        plt.plot(v_loss, label='Validation Loss')
        plt.title(title)
        plt.ylabel('Average Loss')
        plt.xlabel('Epochs')
        plt.ylim([0, 5])
        plt.xlim([0,epochs])
        plt.legend()
        plt.savefig(save_name)

    def loss_evaluation(self):

        mean_v_loss = []
        mean_t_loss = []

        for k in range(5):

            title = 'K ' + str(k)

            for file_accuracy in self.files_accuracy:
                if file_accuracy[-5:-4] == str(k): 
                    accuracy_df = pd.read_csv(self.data_path + file_accuracy)
                    epochs = accuracy_df.shape[0]

                    v_loss = accuracy_df['Validation Loss']
                    t_loss = accuracy_df['Train Loss']

                    title = self.clipping + ', K' + str(k) + ': Log Loss per Epoch'

                    self.plot_loss(v_loss, t_loss, epochs, title, self.data_path + 'Img_Loss_' + str(k) + '.png')

                    mean_v_loss.append(v_loss)
                    mean_t_loss.append(t_loss)

        mean_v_loss = np.array(mean_v_loss)
        mean_t_loss = np.array(mean_t_loss)

        mean_v = mean_v_loss.mean(axis=(0))
        mean_t = mean_t_loss.mean(axis=(0))

        title = self.clipping + ': Average Loss per Epoch'

        self.plot_loss(mean_v, mean_t, epochs, title, self.data_path + 'Img_Mean_Loss.png')

        return mean_v, mean_t


    def plot_auc_curve(self, validation_df, train_df, title, name, epochs):

        plt.figure(figsize=(10,5))
        plt.plot(train_df, label = 'Training Data')
        plt.plot(validation_df, label = 'Validation Data')
        plt.hlines(y=0.5, xmin=0, xmax=epochs, linestyles='--', lw=2, alpha=0.5, label='0.5')
        plt.hlines(y=0.9, xmin=0, xmax=epochs, linestyles='--', lw=2, alpha=0.5, label='0.9')
        plt.xlabel('Epochs')
        plt.ylabel('AUROC')
        plt.title(title)
        plt.ylim(0,1.1)
        plt.legend()
        plt.savefig(self.data_path + name + '.png')


    def auc_evaluation(self):

        auc_df = pd.DataFrame()
        auc_t_df = pd.DataFrame()
        # find associated files for each fold
        for accuracy_file in self.files_accuracy:

            acc_df = pd.read_csv(self.data_path + accuracy_file)
            epochs = acc_df.shape[0]
            k = 'K ' + accuracy_file[-5:-4]
            title = self.clipping + ', ' + k + ': AUROC'
            save_name = 'Img_AUC_' + k

            self.plot_auc_curve(acc_df['Validation AUC'], acc_df['Train AUC'], title, save_name, epochs)

            auc_list = acc_df['Validation AUC']
            auc_train_list = acc_df['Train AUC']
            new_df = pd.DataFrame({k : auc_list})
            auc_df = pd.concat([auc_df, new_df], axis=1)
            new_df = pd.DataFrame({k : auc_train_list})
            auc_t_df = pd.concat([auc_t_df, new_df], axis=1)

        # Add row with Mean over all folds for each epoch
        row_means = auc_df.mean(axis=1)
        auc_df['Mean'] = row_means

        row_means = auc_t_df.mean(axis=1)
        auc_t_df['Mean'] = row_means

        #best_aug = auc_df[auc_df['Mean'] == auc_df['Mean'].max()]
        #best_aug_index = best_aug.index

        title = self.clipping + ': Mean AUROC'
        save_name = 'Img_Mean_AUC_'
        self.plot_auc_curve(auc_df['Mean'], auc_t_df['Mean'], title, save_name, epochs)

        return epochs


    def calulate_means(self, epochs):
        # calculate the mean accross all folds for every Metric for every epoch

        mean_df = pd.DataFrame()
        for i in range(epochs):
            new_df = pd.DataFrame()
            for file_name in self.files_accuracy:
                df = pd.read_csv(self.data_path + file_name)
                row = df.iloc[i:i+1]
                new_df = pd.concat([new_df, row])
            row_means = new_df.mean(axis=0)
            row_means = pd.DataFrame(row_means).T
            mean_df = pd.concat([mean_df, row_means], axis=0, ignore_index= True)

        return mean_df
    
    def get_outcome_of_epoch(self, df, epochs, epoch):
        n = int(df.shape[0] / epochs)
        
        return df.iloc[n*epoch:n*(epoch+1),:]

    def get_mean_outcome_of_epoch(self, epochs, epoch):
        outcomes = pd.DataFrame(columns =['Labels', 'Probabilities'])
        for file_name in self.files_v_outcome:
            df = pd.read_csv(self.data_path + file_name)
            outcome = self.get_outcome_of_epoch(df, epochs, epoch)
            outcomes = pd.concat([outcomes, outcome])
        return outcomes

    # Early Stopping for all folds in the same epoch
    def early_stopping(self, mean_df, metric_a, condition_a, metric_b, condition_b):
        stop_epoch = 0
        outcome_validation = pd.DataFrame(columns =['Label', 'Predicted Probability'])
        outcome_training = pd.DataFrame(columns =['Label', 'Predicted Probability'])
        epochs = mean_df.shape[0]
        for i, row in mean_df.iterrows():
            # first condition: A Minimum of Metric_a of Condition_a
            if (row[metric_a] > condition_a):
                # Check the next condition_b rows for better results
                for j in range(i+1, min(i+ condition_b, len(mean_df))):
                    if mean_df.loc[j, metric_b] < row[metric_b]:
                        break
                else:
                    # if the accuracy declines within the next x rows, return this row
                    stop_epoch = i
                    print(f'Stopping in the same Epoch: {i}')
                    print(row)
                    outcome_validation = self.get_mean_outcome_of_epoch(epochs, stop_epoch)
                    #outcome_training = pd.concat([outcome_training, set2])
                    break
                        
        if outcome_validation.shape[0] > 10:
            self.plot_roc(outcome_validation, 'at Epoch: ' + str(stop_epoch), 'All')
        
        return stop_epoch
    
    # Early Stopping for all folds in the same epoch
    def fixed_early_stopping(self, mean_df, epochs):
        
        outcome_validation = pd.DataFrame(columns =['Label', 'Predicted Probability'])
        outcome_training = pd.DataFrame(columns =['Label', 'Predicted Probability'])
        epochs = mean_df.shape[0]
        for i, row in mean_df.iterrows():
            # first condition: A Minimum of Metric_a of Condition_a
            if (row['Train Accuracy'] > 0.85):
                if((row['Train DOR'] + 0.1) > row['Validation DOR']):
                    for j in range(i+1, min(i+30, len(mean_df))):
                        if ((mean_df.loc[j,'Train DOR'] + 0.1) > mean_df.loc[j,'Validation DOR']):
                            if (mean_df.loc[j,'Validation DOR'] > row['Validation DOR']):
                                break
                    else:
                        stop_epoch = i
                        print(f'Stopping in the same Epoch: {i}')
                        print(row)
                        #outcome_training = pd.concat([outcome_training, set2])
                        break
        else:
            print('Training not successful')
            for i, row in mean_df.iterrows():
                if (row['Train Accuracy'] > 0.95):
                    stop_epoch = i
                    print(f'Stopping in the same Epoch: {i}')
                    print(row)
                    break

        outcome_validation = self.get_mean_outcome_of_epoch(epochs, stop_epoch)          
        self.plot_roc(outcome_validation, 'at Epoch: ' + str(stop_epoch), 'All')
        
        mean_df['Epoch'] = mean_df.index
        results = mean_df.iloc[i:i+1]
        
        return results, stop_epoch

    def get_outcome_files(self, k):

        for file_accuracy in self.files_accuracy:
                if file_accuracy[-5:-4] == str(k):
                    for file_v_outcome in self.files_v_outcome:
                        if file_v_outcome[-5:-4] == str(k):
                            return file_accuracy, file_v_outcome

    def early_stopping_fold(self, metric_a, condition_a, metric_b, condition_b, metric_c, condition_c):
        stop_epoch = 0
        outcome_validation = pd.DataFrame(columns =['Label', 'Predicted Probability'])
        outcome_training = pd.DataFrame(columns =['Label', 'Predicted Probability'])
        folds_results = self.results
        # find associated files for each fold
        for k in range(5):
            #print(k)
            file_accuracy, file_v_outcome = self.get_outcome_files(k)

            acc_df = pd.read_csv(self.data_path + file_accuracy)
            epochs = acc_df.shape[0]
            df = pd.read_csv(self.data_path + file_v_outcome)

            for i, row in acc_df.iterrows():
                break_loop = False
                # first condition: A Minimum of Metric_a of Condition_a
                if (row[metric_a] > condition_a):
                    # Check the next condition_b rows to make sure the condition holds
                    if (row[metric_b] > 0.8):
                        for j in range(i+1, min(i+ condition_b, len(acc_df))):
                            if acc_df.loc[j, metric_b] > row[metric_b]:
                                break_loop = True
                                break
                        if break_loop == True:
                            break
                        else:
                            for j in range(i+1, min(i+ condition_c, len(acc_df))):
                                if acc_df.loc[j, metric_c] > row[metric_c]:
                                    break_loop = True
                                    break
                        if break_loop == True:
                            break
                        else:
                        # if the accuracy declines within the next x rows, return this row
                            stop_epoch = i
                            folds_results = pd.concat([folds_results, acc_df.iloc[i:i+1]])
                            #print(f'Stopping at different Epochs: {i}')
                            #print(row)
                            set1 = self.get_outcome_of_epoch(df, epochs, stop_epoch)
                            outcome_validation = pd.concat([outcome_validation, set1])
                            break

        if outcome_validation.shape[0] > 29:
            self.plot_roc(outcome_validation, 'Individual stopping for each fold', 'fold')
        
        return folds_results
    
    def fixed_early_stopping_fold(self, epochs):
        stop_epoch = 0
        outcome_validation = pd.DataFrame(columns =['Label', 'Predicted Probability'])
        outcome_training = pd.DataFrame(columns =['Label', 'Predicted Probability'])
        folds_results = self.results
        # find associated files for each fold
        for k in range(5):
            print(k)
            file_accuracy, file_v_outcome = self.get_outcome_files(k)
            acc_df = pd.read_csv(self.data_path + file_accuracy)
            epochs = acc_df.shape[0]
            df = pd.read_csv(self.data_path + file_v_outcome)


            for i, row in acc_df.iterrows():
                if (row['Train Accuracy'] > 0.85):
                    if ((row['Train DOR'] + 0.1) > row['Validation DOR']):
                        for j in range(i+1, min(i+30, len(acc_df))):
                            if (acc_df.loc[j, 'Train DOR'] + 0.1) > acc_df.loc[j,'Validation DOR']:
                                if (acc_df.loc[j,'Validation DOR'] > row['Validation DOR']):
                                    break
                        else:   
                            # if the accuracy declines within the next x rows, return this row
                            stop_epoch = i
                            acc_df['Epoch'] = acc_df.index
                            folds_results = pd.concat([folds_results, acc_df.iloc[i:i+1]])
                            print(f'Stopping at different Epochs: {i}')
                            print(row)
                            set1 = self.get_outcome_of_epoch(df, epochs, stop_epoch)
                            outcome_validation = pd.concat([outcome_validation, set1])
                            break
            else:
                print('Training not successful')
                for i, row in acc_df.iterrows():
                    if (row['Train Accuracy'] > 0.95):
                        stop_epoch = i
                        acc_df['Epoch'] = acc_df.index
                        folds_results = pd.concat([folds_results, acc_df.iloc[i:i+1]])
                        print(f'Stopping at different Epochs: {i}')
                        print(row)
                        set1 = self.get_outcome_of_epoch(df, epochs, stop_epoch)
                        outcome_validation = pd.concat([outcome_validation, set1])
                        break

        self.plot_roc(outcome_validation, 'Individual stopping for each fold', 'fold')
        
        return folds_results

    def plot_roc(self, outcome, name, save_name):

        outcome = outcome.apply(pd.to_numeric)
        fpr, tpr, thresholds = roc_curve(outcome['Label'], outcome['Predicted Probability'])

        plt.figure(figsize=(10,5))
        plt.plot(fpr, tpr)
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity') 
        plt.title(self.clipping + ' ROC Curve: ' + name)

        threshold_values = [0.01, 0.2, 0.5, 0.8, 0.99]
        for threshold in threshold_values:
            idx = np.argmin(np.abs(thresholds - threshold))
            plt.scatter(fpr[idx], tpr[idx], marker='o', color='red')
            plt.text(fpr[idx], tpr[idx], f't = {threshold}', color='red', verticalalignment='bottom', horizontalalignment='right')

        plt.savefig(self.data_path + save_name + '_ROC.png')
                    
    def get_mean_results(self, mean_results):
        mean_results = {
            'Name': [self.clipping],
            'Validation Accuracy' :  [mean_results['Validation Accuracy'].mean()],
            'Validation Balanced Accuracy' : [mean_results['Validation Balanced Accuracy'].mean()],
            'Validation DOR' : [mean_results['Validation DOR'].mean()],
            'Validation Sensitivity' : [mean_results['Validation Sensitivity'].mean()],
            'Validation Specificity' : [mean_results['Validation Specificity'].mean()],
            'Validation AUC' : [mean_results['Validation AUC'].mean()],
            'Train Accuracy' : [mean_results['Train Accuracy'].mean()],
            'Train Balanced Accuracy' : [mean_results['Train Balanced Accuracy'].mean()],
            'Train DOR': [mean_results['Train DOR'].mean()],
            'Train Sensitivity' : [mean_results['Train Sensitivity'].mean()],
            'Train Specificity' : [mean_results['Train Specificity'].mean()],
            'Train AUC':[mean_results['Train AUC'].mean()]
              }
                    
        df = pd.DataFrame(mean_results)
        df = df.round(decimals=4)
        return df
    
    def store_results(self, file, new_results):
        
        new_results['Name'] = self.clipping
        if os.path.isfile(file):
            results = pd.read_csv(file)
            results = pd.concat([results, new_results])
        else:
            results = new_results
        
        results.to_csv(file, index=False)
        

    def cross_validation_evaluation(self):

        #mean_v_loss, mean_t_loss = self.loss_evaluation()
        epochs = self.auc_evaluation()
        mean_df = self.calulate_means(epochs)
        #stop_epoch = self.early_stopping(mean_df, 'Validation AUC', 0.75, 'Average Validation Loss', 30)
        #stop_epoch = self.early_stopping(mean_df, 'Validation AUC', 0.75, 'Validation AUC', 30)
        #plot_roc(outcome_validation, 'Early stopping at: ' + str(stop_epoch))

        results, stop_epoch = self.fixed_early_stopping(mean_df, epochs)
        self.store_results('Training_same_epoch.csv', results)
        
        folds_results = self.fixed_early_stopping_fold(epochs)
        for i in range (0,5):
            self.store_results('Training_K' + str(i) + '_results.csv', folds_results[i:i+1])
        
        mean_results = self.get_mean_results(folds_results)
        self.store_results('Training_results.csv', mean_results)
        

if __name__ == '__main__':
    
    # Learning Evaluation
    train_evaluator = Training_Evaluator('./trails/models/2_filtTrue_distorted_ahe_org_randFalse_md200_bs4_flFalse_lr0.02_oAdam_d20_wh160_wd0.003_dr0.1_aAug/', 'Selective Cut, 200 layers, distorted')
    train_evaluator.cross_validation_evaluation()

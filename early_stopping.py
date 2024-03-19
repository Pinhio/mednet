import pandas as pd

class Early_Stopping():
    
    def __init__(self):

        self.best_dor = 0
        self.best_epoch = 0
        self.best_dor = 0

    def early_stopping(self, results_df, epoch):
        current_results = results_df.iloc[-1]
        print(current_results)
        if (current_results['Train Accuracy'] > 0.85):
            if (epoch - self.best_epoch) < 30: 
                if (current_results['Validation DOR'] > self.best_dor) & ((current_results['Train DOR'] + 0.05) > current_results['Validation DOR']):
                    self.best_dor = current_results['Validation DOR']
                    self.best_epoch = epoch
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
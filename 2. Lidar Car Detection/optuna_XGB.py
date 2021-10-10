import numpy as np
import optuna
import os
import time
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tqdm import tqdm



DATA_PATH_TRAIN = r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\train.csv'
DATA_PATH_TEST = r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\test.csv'

# X_train = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\X_train.csv')
# X_valid = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\X_valid.csv')
# y_train = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\y_train.csv')
# y_valid = pd.read_csv(r'C:\Users\user\Downloads\EmEl\Kaggle\30-Days-of-ML-Competition\data\y_valid.csv')

data_dir = r"C:\Users\ollko\Desktop\EmEl\Kaggle\ai-blitz-xi\Lidar Car Detection\data"

# Loading the train data
train_data = np.load(os.path.join(data_dir,"train.npz"), allow_pickle=True)
train_data = train_data['train']

# Loading the test data
test_data = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
test_data = test_data['test']

X = train_data[:, 0]
X = [i.flatten() for i in X]


# labels
y = train_data[:, 1]

# flattening the points into 1d array
X_test = [i.flatten() for i in test_data]




# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    
    # param = {
    #     #'tree_method': 'gpu_hist',
    #     # 'metric': 'rmse',
    #     'objective': 'reg:squarederror',
    #     #'scale_pos_weight': 1,
    #     'n_estimators': 200,
    #     'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0),
    #     'subsample': trial.suggest_float('subsample', 0.2, 1.0), 
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
    #     'grow_policy':'lossguide',
    #     'max_depth': trial.suggest_int('max_depth', 2, 10),
    #     'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 50.0),
    #     'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 50.0),
    #     "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
    # }  

    K = 5
    kf = KFold(n_splits = K, random_state = 134, shuffle = True)

    xgb_preds = []
    mse = []
    train = np.array(X)
    target_train = np.array(y)
    test = np.array(X_test)    

    xgb_params = {
                "objective":"multi:softmax",
                'num_class': 8,
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.5), 
                'learning_rate': trial.suggest_float('learning_rate', 1e-8, 0.3),
                'max_depth': trial.suggest_int('max_depth', 2, 10), 
                'alpha': trial.suggest_float('alpha', 1e-3, 20),
                'tree_method': 'gpu_hist'
                }

    for train_index, test_index in kf.split(train):

        train_X, valid_X = train[train_index], train[test_index]
        train_y, valid_y = target_train[train_index], target_train[test_index]

        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)
        d_test = xgb.DMatrix(test)
        d_valid_x = xgb.DMatrix(valid_X)
        
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(xgb_params,
                        d_train, 
                        num_boost_round=200,  
                        evals=watchlist,
                        verbose_eval=50, 
                        early_stopping_rounds=10,
                        )
                
        xgb_pred = model.predict(d_test)
        xgb_preds.append(list(xgb_pred))
        mse.append(mean_squared_error(valid_y, model.predict(d_valid_x), squared=False ))
        
    accuracy = np.mean(mse)

    return accuracy


if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    stop_time = time.time()
    print("Number of finished trials: {}".format(len(study.trials)))
    print('Time ellapsed: {}'.format(stop_time - start_time))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print('Param importance:')    
    importance = optuna.importance.get_param_importances(study=study)
    # print(importance)  
    
    plt.bar(range(len(importance)), list(importance.values()), align='center')
    plt.xticks(range(len(importance)), list(importance.keys()))
    plt.show()
    

    # Trial 31 finished with value: 1.333255145221258 and
    #  parameters: {'colsample_bytree': 0.43372079833334837,
    #   'learning_rate': 0.02861359769822289, 
    #   'max_depth': 3, 
    #   'alpha': 6.424384586900361}. 
    #   Best is trial 31 with value: 1.333255145221258.
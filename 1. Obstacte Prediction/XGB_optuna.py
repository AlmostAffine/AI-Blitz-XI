import numpy as np
import optuna
import time
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold



data = np.load(r"C:\Users\ollko\Desktop\EmEl\Kaggle\ai-blitz-xi\data.npz", allow_pickle=True)

train_data = data["train"]
test_data = data['test']
# print(train_data.shape, test_data.shape)

X = np.array([sample.flatten() for sample in train_data[:, 0].tolist()])
y = np.array(train_data[:, 1].tolist())

def objective(trial):
    param = {
        'tree_method': 'gpu_hist',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'n_jobs': -1,
        "objective": "binary:logistic",
        'n_estimators': 400,
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, .3),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'grow_policy': trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 50.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 50.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10)
    }   
    
    # xgb_val_pred = np.zeros(len(y))
    # xgb_test_pred = np.zeros(len(test))
    f1 = []
    kf = KFold(n_splits=5, shuffle=True)

    for train_index, valid_index in kf.split(X):
    
        X_train_idx, y_train_idx = X[train_index], y[train_index]
        X_valid_idx, y_valid_idx = X[valid_index], y[valid_index]

        xgb_model = XGBClassifier(**param)
        xgb_model.fit(X_train_idx, y_train_idx)

        f1.append(f1_score(y_valid_idx, xgb_model.predict(X_valid_idx)))
        
    accuracy = np.mean(f1)

    return accuracy


if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    stop_time = time.time()
    print("Number of finished trials: {}".format(len(study.trials)))
    print('Time ellapsed: {}'.format(stop_time - start_time))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))    


# Number of finished trials: 10
# Time ellapsed: 253.59352850914001
# Best trial:
#   Value: 0.9753373302036776
#   Params:
#     learning_rate: 0.22900092921344353
#     subsample: 0.8280274632640745
#     colsample_bytree: 0.232060989586643
#     grow_policy: lossguide
#     max_depth: 2
#     reg_lambda: 4.140329381620182
#     reg_alpha: 2.971876352153861
#     min_child_weight: 2

# GPU
# Number of finished trials: 10
# Time ellapsed: 176.80613040924072
# Best trial:
#   Value: 0.9705959811262972
#   Params:
#     learning_rate: 0.18241002805198292
#     subsample: 0.9442992163119097
#     colsample_bytree: 0.7037047987766014
#     grow_policy: depthwise
#     max_depth: 6
#     reg_lambda: 49.31333120685533
#     reg_alpha: 14.886067624122179
#     min_child_weight: 3

# gpu
# Number of finished trials: 100
# Time ellapsed: 1558.1802666187286
# Best trial:
#   Value: 0.9767351284644757
#   Params:
#     learning_rate: 0.23119131605979304
#     subsample: 0.810758128177923
#     colsample_bytree: 0.4122195688798259
#     grow_policy: depthwise
#     max_depth: 7
#     reg_lambda: 19.27299537835576
#     reg_alpha: 0.7185625975512855
#     min_child_weight: 2

# gpu
# Number of finished trials: 200
# Time ellapsed: 3328.421802043915
# Best trial:
#   Value: 0.9771531696432323
#   Params:
#     learning_rate: 0.2939679994771448
#     subsample: 0.6198231912457959
#     colsample_bytree: 0.6392355971709197
#     grow_policy: depthwise
#     max_depth: 10
#     reg_lambda: 20.320390675035057
#     reg_alpha: 1.1740272387059525
#     min_child_weight: 2
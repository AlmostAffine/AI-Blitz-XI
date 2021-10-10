import numpy as np
import optuna
import time
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


data = np.load(r"C:\Users\ollko\Desktop\EmEl\Kaggle\ai-blitz-xi\data.npz", 
            allow_pickle=True)

train_data = data["train"]
test_data = data['test']
# print(train_data.shape, test_data.shape)

X = np.array([sample.flatten() for sample in train_data[:, 0].tolist()])
y = np.array(train_data[:, 1].tolist())


def objective(trial):
    
    f1 = []
    kf = KFold(n_splits=5, shuffle=True)

    for train_index, valid_index in kf.split(X):
    
        X_train_idx, y_train_idx = X[train_index], y[train_index]
        X_valid_idx, y_valid_idx = X[valid_index], y[valid_index]

        params = {
            'iterations': 1000,                         
            'depth': trial.suggest_int('depth', 1, 10),                                       
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),               
            'random_strength': trial.suggest_int('random_strength', 0, 100),                       
            'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
            'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter'])
        }
        # Learning
        cat_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="F1",
            # task_type="GPU",
            l2_leaf_reg=50,
            random_seed=72,
            border_count=64,
            verbose=100,
            **params
        )        
        
        cat_model.fit(X_train_idx, y_train_idx,
                     eval_set=(X_valid_idx, y_valid_idx))
        # Predict
        f1.append(f1_score(y_valid_idx, cat_model.predict(X_valid_idx)))
        
    accuracy = np.mean(f1)

    return accuracy


if __name__ == "__main__":
    start_time = time.time()
    study = optuna.create_study(direction="maximize")
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


# Number of finished trials: 5
# Time ellapsed: 144.52528977394104
# Best trial:
#   Value: 0.974605051225096
#   Params:
#     iterations: 297
#     depth: 9
#     learning_rate: 0.053217561661965736
#     random_strength: 70
#     bagging_temperature: 1.9057649599088144
#     od_type: IncToDec


# value: 0.9764012487245853.
# {'depth': 4, 
# 'learning_rate': 0.0366939810616369, 
# 'random_strength': 15, 
# 'bagging_temperature': 1.6040535231317334, 
# 'od_type': 'IncToDec'}

# value: 0.9778910803911369 and parameters: {'depth': 4, 
# 'learning_rate': 0.12453342973535175, 
# 'random_strength': 86,
#  'bagging_temperature': 0.4109009126297936, 
# 'od_type': 'IncToDec'


#  value: 0.9784450040961957 and parameters: {'depth': 7, 
# 'learning_rate': 0.1089803195170146, 
# 'random_strength': 67, 
# 'bagging_temperature': 0.04199212199625898,
# 'od_type': 'IncToDec'}

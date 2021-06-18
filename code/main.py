import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("hi")
train=pd.read_csv('C://Users//86180//Desktop//NJU//2.2//2.2_Machine_Learning//PS6 competition//kaggle_june//train.csv')
test=pd.read_csv('C://Users//86180//Desktop//NJU//2.2//2.2_Machine_Learning//PS6 competition//kaggle_june//test.csv')
le = LabelEncoder()
train['target'] = le.fit_transform(train['target'])
cols = list(train.columns)
cols.remove("target")
cols.remove("id")
not_features = ['id', 'target']
features = []
for feat in train.columns:
    if feat not in not_features:
        features.append(feat)
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])
X=train.drop(['target','id'],axis=1)
Y=train['target']

def objective(trial,data=X,target=Y):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2,random_state=42)
    param = {
                "n_estimators" : trial.suggest_int('n_estimators', 500, 1000),
                'max_depth':trial.suggest_int('max_depth', 2, 6),
                'reg_alpha':trial.suggest_loguniform('reg_alpha', 1e-5, 10),
                'reg_lambda':trial.suggest_loguniform('reg_lambda', 1e-5, 10),
                'min_child_weight':trial.suggest_loguniform('min_child_weight', 1e-5, 5),
                'gamma':trial.suggest_loguniform('gamma', 1e-5, 5),
                'learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5),
                'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01),
                'nthread' : -1,
                'use_label_encoder' : False
            }
    model = XGBClassifier(**param)  
    model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=100,verbose=False)
    y_preds = model.predict_proba(X_test)
    log_loss_multi = log_loss(y_test, y_preds)
    return log_loss_multi

OPTUNA_OPTIMIZATION = True
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200, gc_after_trial=True)
print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

from xgboost import XGBClassifier
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

if __name__ == '__main__':
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
    params = {'n_estimators': 917, 
              'max_depth': 6, 
              'reg_alpha': 9.736602014559566, 
              'reg_lambda': 0.0004577827887288241, 
              'min_child_weight': 0.005161022323227045, 
              'gamma': 1.1453113147311293, 
              'learning_rate': 0.04665662582966712, 
              'colsample_bytree': 0.1}
    model = XGBClassifier(**params) 
    model.fit(X, Y)
    cols = list(test.columns)
    cols.remove("id")
    not_features = ['id', 'target']
    features = []
    for feat in test.columns:
        if feat not in not_features:
            features.append(feat)
    scaler = StandardScaler()
    test[features] = scaler.fit_transform(test[features])
    X_test=test.drop(['id'],axis=1)
    proba = model.predict_proba(X_test)
    output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5':proba[:,4], 'Class_6':proba[:,5], 'Class_7':proba[:,6], 'Class_8':proba[:,7], 'Class_9':proba[:,8]})
    output.to_csv('C://Users//86180//Desktop//NJU//2.2//2.2_Machine_Learning//PS6 competition//kaggle_june//my_submission.csv', index=False)

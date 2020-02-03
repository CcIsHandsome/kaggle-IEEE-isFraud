from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost  as xgb


def stacking(model_list, X, y, test_set):
    # First level learning model
    # Create new data
    new_train_matrix = list()
    new_test_matrix = list()

    for model in model_list:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        new_train_col = np.zeros(len(y))
        new_test_col = np.zeros(len(test_set))

        for train_index, test_index in skf.split(train_set, train_label):
            X_train, X_vali = X[train_index], X[test_index]
            y_train, y_vali = y[train_index], y[test_index]

            train_pred, test_pred = model_fit_pred(model, X_train, y_train, X_vali, y_vali, test_set)
            new_train_col[test_index] = train
            new_test_col = new_test_col + test_pred

        new_test_col = new_test_col / 5
        new_train_matrix.append(new_train_col)
        new_test_matrix.append(new_test_col)

    # Second level learning model
    new_train_matrix = np.ndarray(new_train_matrix)
    new_test_matrix = np.ndarray(new_test_matrix)

    layer2_model = xgb_model = XGBClassifier(
        learning_rate=0.1, n_estimators=100, max_depth=3, min_child_weight=1, gamma=0,
        subsample=1, colsample_bytree=1, objective='binary:logistic', nthread=10,
        scale_pos_weight=1, seed=1337, **param_dict)
    layer2_model.fit(new_train_matrix, train_label)
    pred_result = layer2_model.pred(new_test_matrix)
    return pred_result


def model_fit_pred(model_in, train_set, train_label, vali_set, vali_label, test_set):
    if model == 'RF':
        RF_model = RandomForestClassifier(bootstrap=True, max_depth=12, random_state=42, n_estimators=200)
        RF_model.fit(train_set, train_label)
        train_pred = RF_model.predict_proba(vali_set)[:, 1]
        test_pred = RF_model.predict_proba(test_set)[:, 1]
        return train_pred, test_pred
    elif model == 'XGBoost':
        param_dict = {'tree_method': 'gpu_hist'}
        xgb_model = xgb.XGBClassifier(
            learning_rate=0.1, n_estimators=200, max_depth=9, min_child_weight=1, gamma=0,
            subsample=1, colsample_bytree=1, objective='binary:logistic', nthread=10,
            scale_pos_weight=1, seed=1337, **param_dict)
        xgb_model.fit(train_set, train_label)
        train_pred = xgb_model.predict_proba(vali_set)[:, 1]
        test_pred = xgb_model.predict_proba(test_set)[:, 1]
        return train_pred, test_pred

    elif model == 'LGBM':
        lgb_model = lgb.LGBMClassifier(
            num_leaves=491, boosting_type='gbdt', max_depth=-1, learning_rate=0.006883242363721497, n_estimators=800,
            subsample_for_bin=50000, objective='binary', min_child_weight=0.03454472573214212,
            min_child_samples=106, subsample=0.4181193142567742, subsample_freq=1, metric='auc',
            colsample_bytree=0.3797454081646243, reg_alpha=0.3899927210061127, reg_lambda=0.6485237330340494,
            random_state=42, silent=True)

        lgb_model.fit(train_set, train_label)
        train_pred = lgb_model.predict_proba(vali_set)[:, 1]
        test_pred = lgb_model.predict_proba(test_set)[:, 1]

        return train_pred, test_pred


skf = StratifiedKFold(n_splits=5, shuffle=True)
x = pd.read_pickle('../8_14/train_data.pkl').values[:, 1:]
y = pd.read_pickle('../8_14/train_data.pkl').values[:, 0]
test = pd.read_pickle('../8_14/test_data.pkl').values
for train_index, test_index in skf.split(x, y):
    X_train, X_vali = x[train_index], x[test_index]
    y_train, y_vali = y[train_index], y[test_index]
    model_l = ['RF']
    resu = stacking(model_l, X_train, y_train, X_vali)
    vali_auc = metrics.roc_auc_score(y_vali, resu)
    print("vali_auc:" + str(vali_auc))

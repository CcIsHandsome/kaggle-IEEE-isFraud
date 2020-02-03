import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_rows", None)

lgb_path = './lgb_models_stack/'
xgb_path = './xgb_models_stack/'
cb_path = './cb_models_stack/'
# Create dir for models
if not os.path.exists(lgb_path):
    os.mkdir(lgb_path)
if not os.path.exists(xgb_path):
    os.mkdir(xgb_path)
if not os.path.exists(cb_path):
    os.mkdir(cb_path)

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import gc

gc.enable()

emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
          'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol',
          'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple',
          'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other',
          'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other',
          'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other',
          'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink',
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo',
          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft',
          'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other',
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other',
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol',
          'juno.com': 'other', 'icloud.com': 'apple', 'None': 'other'}
us_emails = ['com', 'net', 'edu']


# Reduce_memory
def reduce_memory(df):
    print("Reduce_memory...")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    return df


from sklearn.ensemble.forest import RandomForestClassifier
import time


def RandomForestSelector(A, y, n_estimators, n_features):
    columns = A.columns.values
    rf = RandomForestClassifier(n_estimators=n_estimators, verbose=0, n_jobs=-1, max_depth=9, random_state=2019)
    rf.fit(A, y)
    importance = rf.feature_importances_
    importance_index = np.argsort(importance)[::-1][:n_features]
    importance_columns = columns[importance_index]
    importance_values = importance[importance_index]
    # print(importance_columns, "\n选取后:",  len(importance_columns))
    # importance_dataFrame = pd.DataFrame({
    #         "feature": importance_columns,
    #         "value": importance_values
    # })
    # print(importance_dataFrame)
    A = A[importance_columns]
    return A, importance_columns


# MODELS
# LightGBM Model
def fit_lgb(X_fit, y_fit, X_val, y_val):
    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, max_depth=9, boosting_type='gbdt',
                               objective='binary', metric='auc', num_leaves=7, n_jobs=-1)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=0, early_stopping_rounds=1000)
    cv_val = model.predict_proba(X_val)[:, 1]
    del X_fit, y_fit, X_val, y_val
    return cv_val


# XGBoost Model
def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    model = xgb.XGBClassifier(n_estimators=500, max_depth=9, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                              tree_method='gpu_hist', n_jobs=-1, random_state=2019)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    cv_val = model.predict_proba(X_val)[:, 1]
    # Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter + 1)
    pickle.dump(model, open(save_to, "wb"))
    del X_fit, y_fit, X_val, y_val
    return cv_val


# Catboost Model
def fit_cb(X_fit, y_fit, X_val, y_val, counter, cb_path, name):
    model = cb.CatBoostClassifier(iterations=1000, task_type='GPU', eval_metric='AUC', random_state=2019)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=0, early_stopping_rounds=1000)
    cv_val = model.predict_proba(X_val)[:, 1]
    # Save Catboost Model
    save_to = "{}{}_fold{}.mlmodel".format(cb_path, name, counter + 1)
    model.save_model(save_to, format="coreml", export_parameters={'prediction_type': 'probability'})
    del X_fit, y_fit, X_val, y_val
    return cv_val


print("loading data")
df_trans = pd.read_csv('train_transaction.csv', index_col='TransactionID')
df_test_trans = pd.read_csv('test_transaction.csv', index_col='TransactionID')

df_id = pd.read_csv('train_identity.csv', index_col='TransactionID')
df_test_id = pd.read_csv('test_identity.csv', index_col='TransactionID')

df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True)
df_test = df_test_trans.merge(df_test_id, how='left', left_index=True, right_index=True)
del df_trans, df_test_trans, df_id, df_test_id


def df_process(df):
    drop_columns = [
        'id_09', 'id_10', 'id_04', 'id_20', 'id_13', 'id_17', 'id_11', 'id_12', 'id_05', 'id_06', 'V201', 'V170',
        'V200', 'V188', 'V208', 'V185', 'V198', 'V184', 'V169', 'V195', 'V197', 'V194', 'V174', 'V175', 'V180', 'V199',
        'V190', 'V176', 'V187', 'V189', 'V186', 'V192', 'V196', 'V193', 'V191', 'V181', 'V183', 'V173', 'V172', 'V182',
        'V210', 'V167', 'V177', 'V179', 'V272', 'V270', 'V222', 'V221', 'V259', 'V245', 'V239', 'V220', 'V178', 'V271',
        'V238', 'V256', 'V255', 'V227', 'V251', 'V250', 'V234', 'V206', 'V258', 'V229', 'V168', 'id_19', 'V205', 'V218',
        'V232', 'V219', 'V230', 'V233', 'V217', 'V246', 'V228', 'V231', 'V243', 'V244', 'V242', 'V261', 'V262', 'V236',
        'V237', 'V248', 'V253', 'V235', 'V254', 'V249', 'V252', 'V247', 'V260', 'V225', 'V226', 'V224', 'V223', 'V240',
        'V241', 'V266', 'V10', 'V1', 'V6', 'V8', 'V2', 'V4', 'V7', 'V9', 'V3', 'V5', 'V69', 'V53', 'V54', 'V68', 'V65',
        'V66', 'V67', 'V55', 'V61', 'V62', 'V56', 'id_29', 'C12', 'C4', 'C10', 'C7', 'C11', 'id_26', 'id_22', 'id_24',
        'id_25', 'V161', 'V327', 'V328', 'V326', 'V330', 'V329', 'V140', 'V158', 'V139', 'V147', 'V156', 'V149', 'V157',
        'V155', 'V146', 'V148', 'V154', 'V153', 'V142', 'V141', 'V138', 'V144', 'V324', 'V151', 'V152', 'V336', 'id_32',
        'V323', 'V143', 'V296', 'V301', 'V300', 'V137', 'V135', 'V125', 'V115', 'V100', 'V114', 'V113', 'V111', 'V110',
        'V108', 'V98', 'V109', 'V119', 'V117', 'V118', 'V122', 'V121', 'V120', 'V107', 'V106', 'V38', 'V44', 'V37',
        'V40', 'V52', 'V39', 'V51', 'V43', 'V50', 'V47', 'V46', 'V41', 'V36', 'V35', 'V48', 'V49', 'V78', 'V86', 'V77',
        'V81', 'V80', 'V79', 'V93', 'V85', 'V92', 'V94', 'V84', 'V83', 'V82', 'V88', 'V89', 'V76', 'V75', 'V90', 'V91',
        'V18', 'V17', 'V34', 'V33', 'V16', 'V15', 'V32', 'V31', 'V22', 'V21', 'V24', 'V23', 'V20', 'V19', 'V26', 'V25',
        'V14', 'V27', 'V28', 'V13', 'V12', 'V29', 'V30', 'V58', 'V73', 'V60', 'V64', 'V57', 'V59', 'V71', 'V63', 'V70',
        'D11', 'V278', 'V268', 'id_28'
    ]
    df.drop(drop_columns, axis=1, inplace=True)

    # V_col = ['V126', 'V127','V128','V129','V130','V131','V132','V133','V134', 'V202', 'V203', 'V204', 'V207', 'V209', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216', 'V263', 'V264', 'V265', 'V267', 'V273', 'V274', 'V275', 'V276', 'V277']
    # df['V_col_max'] = df[V_col].max(axis=0)
    # df['n_eq_V_col_max'] = df[V_col].eq(df['V_col_max'], axis=0).sum(axis=1)
    # df['V_col_sum'] = df[V_col].sum(axis=1)

    for feature in ['id_01', 'id_31', 'id_33', 'ProductCD']:
        df[feature + '_count_dist'] = df[feature].map(df[feature].value_counts(dropna=False))

    df['id_23'] = df['id_23'].str.split(":", expand=True)[1]
    df['id_34'] = df['id_34'].str.split(":", expand=True)[1]

    df.loc[df['id_30'].str.contains("Windows", na=False), 'id_30'] = "Windows"
    df.loc[df['id_30'].str.contains("iOS", na=False), 'id_30'] = "iOS"
    df.loc[df['id_30'].str.contains("Mac OS", na=False), 'id_30'] = "Mac"
    df.loc[df['id_30'].str.contains("Android", na=False), 'id_30'] = "Android"
    df['id_30'] = df['id_30'].map(
        lambda x: 'Other' if (x not in ['Windows', 'iOS', 'Mac', 'Android'] and not pd.isna(x)) else x)

    df.loc[df['id_31'].str.contains("chrome", na=False), 'id_31'] = "chrome"
    df.loc[df['id_31'].str.contains("firefox", na=False), 'id_31'] = "firefox"
    df.loc[df['id_31'].str.contains("safari", na=False), 'id_31'] = "safari"
    df.loc[df['id_31'].str.contains("edge", na=False), 'id_31'] = "edge"
    df.loc[df['id_31'].str.contains("ie", na=False), 'id_31'] = "ie"
    df.loc[df['id_31'].str.contains("samsung", na=False), 'id_31'] = "sumsung"
    df.loc[df['id_31'].str.contains("opera", na=False), 'id_31'] = "opera"
    df['id_31'] = df['id_31'].map(lambda x: 'Other' if (
            x not in ['chrome', 'firefox', 'safari', 'edge', 'sumsung', 'opera'] and not pd.isna(x)) else x)

    df['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
    df['screen_height'] = df['id_33'].str.split('x', expand=True)[1]

    df.loc[df['DeviceInfo'].str.contains("Windows", na=False), "DeviceInfo"] = "Windows"
    df.loc[df['DeviceInfo'].str.contains("iOS", na=False), "DeviceInfo"] = "iOS"
    df.loc[df['DeviceInfo'].str.contains("MacOS", na=False), "DeviceInfo"] = "MacOS"
    df.loc[df['DeviceInfo'].str.contains("SM|SAMSUNG|GT", na=False), "DeviceInfo"] = "SM"
    df.loc[df['DeviceInfo'].str.contains("rv", na=False), "DeviceInfo"] = "rv"
    df.loc[df['DeviceInfo'].str.contains("Trident", na=False), "DeviceInfo"] = "Trident"
    df.loc[df['DeviceInfo'].str.contains("HUAWEI|Huawei|ALE-|-L", na=False), "DeviceInfo"] = "HUAWEI"
    df.loc[df['DeviceInfo'].str.contains("Moto|moto", na=False), "DeviceInfo"] = "Moto"
    df.loc[df['DeviceInfo'].str.contains("LG", na=False), "DeviceInfo"] = "LG"
    df.loc[df['DeviceInfo'].str.contains("HTC", na=False), "DeviceInfo"] = "HTC"
    df.loc[df['DeviceInfo'].str.contains("Redmi|MI|Mi", na=False), "DeviceInfo"] = "MI"
    df.loc[df['DeviceInfo'].str.contains("XT", na=False), "DeviceInfo"] = "Sony"
    df.loc[df['DeviceInfo'].str.contains("Blade|BLADE", na=False), "DeviceInfo"] = "ZTE"
    df.loc[df['DeviceInfo'].str.contains("Linux", na=False), "DeviceInfo"] = "Linux"
    df.loc[df['DeviceInfo'].str.contains("ASUS", na=False), "DeviceInfo"] = "ASUS"
    df['DeviceInfo'] = df['DeviceInfo'].apply(lambda x: 'Other' if (
            x not in ["Windows", "iOS", "MacOS", "SM", "rv", "Trident", "HUAWEI", "Moto", "LG", "HTC", "MI", "Sony",
                      "ZTE", "Linux", "ASUS"] and not pd.isnull(x)) else x)

    df['id_02'] = np.log1p(df['id_02'])
    df['D2'] = np.log1p(df['D2'])

    df['TransactionAmt_Log'] = np.log(df['TransactionAmt'])
    df['TransactionAmt_digit'] = np.ceil(np.log10(df['TransactionAmt']))

    # df['add_diff'] = df['addr1'] - df['addr2']
    df['dist1'].fillna(0, inplace=True)
    df['dist2'].fillna(0, inplace=True)
    df['dist_sum'] = df['dist1'] + df['dist2']
    df['second'] = df['TransactionDT'] % 60
    df['minute'] = (df['TransactionDT'] / 60) % 60
    df['hour'] = (df['TransactionDT'] / 3600) % 24
    df['week'] = (df['TransactionDT'] / 86400) % 7
    df['month'] = (df['TransactionDT'] / 2592000) % 12

    # df['id_TF_concat'] = df['id_35']+df['id_36']+df['id_37']+df['id_38']
    # df['M_TF_concat'] = df['M1'] + df['M2'] + df['M3']+df['M5'] + df['M6']+df['M7']+df['M8']+df['M9']

    df = df.drop("TransactionDT", axis=1)
    return df


def train_test_engineer(train, test):
    for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'ProductCD']:
        train[feature + '_count_full'] = train[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
        test[feature + '_count_full'] = test[feature].map(
            pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

    for c in ['P_emaildomain', 'R_emaildomain']:
        train[c + '_bin'] = train[c].map(emails)
        test[c + '_bin'] = test[c].map(emails)

        train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
        test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

        train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    return train, test


def feature_engineer(df_train, df_test):
    start = time.time()
    df_train, df_test = train_test_engineer(df_train, df_test)
    print("df_train process")
    df_train = df_process(df_train)
    print("df_test process")
    df_test = df_process(df_test)
    print("time: ", time.time() - start)

    y_train = df_train['isFraud'].values
    df_train = df_train.drop("isFraud", axis=1)

    print("encoding")
    for f in df_test.columns:
        try:
            if df_train[f].dtype == 'object' or df_test[f].dtype == 'object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(df_train[f].astype(str).values) + list(df_test[f].astype(str).values))
                df_train[f] = lbl.transform(list(df_train[f].values))
                df_test[f] = lbl.transform(list(df_test[f].values))
        except Exception as e:
            print(f)
            raise e

    return df_train, df_test, y_train


df_train, df_test, y_train = feature_engineer(df_train, df_test)

df_train = reduce_memory(df_train)
df_test = reduce_memory(df_test)


def train(df, y_df, lgb_path, xgb_path, cb_path):
    df_ids = np.array(df.index)
    lgb_cv_result = np.zeros(df.shape[0])

    NumFold = 5
    skf = StratifiedKFold(n_splits=NumFold, shuffle=True, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    # skf = TimeSeriesSplit(n_splits=4)

    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter + 1))
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]

        print('\nLigthGBM')
        pred_val = fit_lgb(X_fit, y_fit, X_val, y_val)
        lgb_cv_result[ids[1]] += fit_lgb(X_fit, y_fit, X_val, y_val) / NumFold
        auc_lgb = roc_auc_score(y_df, lgb_cv_result)
        print('LightGBM VAL AUC: {}'.format(auc_lgb))

        del X_fit, X_val, y_fit, y_val
        gc.collect()


# print("train")
train(df_train, y_train, lgb_path, xgb_path, cb_path)


def predict(df, lgb_path, xgb_path, cb_path):
    # lgb_models = sorted(os.listdir(lgb_path))
    # xgb_models = sorted(os.listdir(xgb_path))
    cb_models = sorted(os.listdir(cb_path))

    # lgb_result = np.zeros(df.shape[0])
    # xgb_result = np.zeros(df.shape[0])
    cb_result = np.zeros(df.shape[0])

    print('\nMake predictions...\n')

    # print('With LightGBM...')
    # for m_name in lgb_models:
    #     # Load LightGBM Model
    #     model = lgb.Booster(model_file='{}{}'.format(lgb_path, m_name))
    #     lgb_result += model.predict(df.values)
    # del model
    # print('With XGBoost...')
    # for m_name in xgb_models:
    #     # Load Catboost Model
    #     model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
    #     xgb_result += model.predict_proba(df.values)[:, 1]
    # del model
    print('With CatBoost...')
    for m_name in cb_models:
        # Load Catboost Model
        model = cb.CatBoostClassifier()
        model = model.load_model('{}{}'.format(cb_path, m_name), format='coreml')
        cb_result += model.predict(df.values, prediction_type='Probability')[:, 1]
    del df, model
    # lgb_result /= len(lgb_models)
    # xgb_result /= len(xgb_models)
    cb_result /= len(cb_models)
    submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')

    # submission['isFraud'] = xgb_result
    # submission.to_csv('xgb_starter_submission.csv')
    # submission['isFraud'] = lgb_result
    # submission.to_csv('lgb_starter_submission.csv')
    submission['isFraud'] = cb_result
    submission.to_csv('cb_starter_submission.csv')

    # submission['isFraud'] = (xgb_result + cb_result) / 2
    # submission.to_csv('xgb_cb_starter_submission.csv')

    # submission['isFraud'] = (lgb_result + xgb_result) / 2
    # submission.to_csv('xgb_lgb_starter_submission.csv')

    # submission['isFraud'] = (lgb_result + cb_result) / 2
    # submission.to_csv('cb_lgb_starter_submission.csv')

    # submission['isFraud'] = (lgb_result + xgb_result + cb_result) / 3
    # submission.to_csv('xgb_lgb_cb_starter_submission.csv')


print("predict")
predict(df_test, lgb_path, xgb_path, cb_path)
print("finish")

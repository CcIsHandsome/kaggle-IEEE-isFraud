import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# train = pd.read_csv('./input/train_transaction.csv')
# test = pd.read_csv('./input/test_transaction.csv')
# useful_features = ['TransactionDT', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1',
#                    'addr2', 'dist1',
#                    'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
#                    'C12', 'C13',
#                    'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
#                    'M2', 'M3',
#                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
#                    'V13', 'V17',
#                    'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46',
#                    'V47', 'V48',
#                    'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69',
#                    'V70', 'V71',
#                    'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90',
#                    'V91', 'V92',
#                    'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138',
#                    'V139', 'V140',
#                    'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159',
#                    'V160', 'V161',
#                    'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175',
#                    'V176', 'V177',
#                    'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202',
#                    'V203', 'V204',
#                    'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217',
#                    'V219', 'V220',
#                    'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234',
#                    'V238', 'V239',
#                    'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258',
#                    'V259', 'V261',
#                    'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274',
#                    'V275', 'V276',
#                    'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292',
#                    'V294', 'V303',
#                    'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323',
#                    'V324', 'V326',
#                    'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338']
# train = train[useful_features]
# test = test[useful_features]
# lbl = LabelEncoder()
# for c in train.columns:
#     if train[c].dtype == 'object':
#         lbl.fit(list(train[c].astype(str).values) + list(test[c].astype(str).values))
#         train[c] = lbl.transform(list(train[c].astype(str).values))
#         test[c] = lbl.transform(list(test[c].astype(str).values))
# # for c in useful_features[28:]:
# #     train_c = train[c].dropna(axis=0, how='any')
# #     test_c = test[c].dropna(axis=0, how='any')
# #     sns.distplot(train_c, color='blue')
# #     sns.distplot(test_c, color='red')
# #     plt.show()
#
# # train_amt = train['TransactionAmt'].dropna(axis=0, how='any')
# # test_amt = test['TransactionAmt'].dropna(axis=0, how='any')
# # sns.distplot(train_amt, hist=False, color='red')
# # sns.distplot(test_amt, hist=False, color='blue')
# # plt.show()
# # train['amt_cut'] = pd.cut(train['TransactionAmt'], bins=20)
# # test['amt_cut'] = pd.cut(test['TransactionAmt'], bins=20)
# # print(train.groupby(['amt_cut'])['TransactionAmt'].count())
# # print(test.groupby(['amt_cut'])['TransactionAmt'].count())
# # sns.distplot(train.groupby(['amt_cut'])['TransactionAmt'].count().values, color='red', bins=20)
# # sns.distplot(test.groupby(['amt_cut'])['TransactionAmt'].count().values, color='blue', bins=20)
# # plt.show()
#
#
# D_columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']
# for d in D_columns:
#     train.set_index('TransactionDT')[d].plot(style='.', title=d)
#     test.set_index('TransactionDT')[d].plot(style='.', title=d)
#     plt.show()

# train = pd.read_pickle('./8_20/train_data.pkl')
# test = pd.read_pickle('./8_20/test_data.pkl')
# print(list(train.columns.values))
# clo = ['TransactionAmt_to_mean_card1', 'TransactionAmt_to_mean_card4', 'TransactionAmt_to_std_card1',
#        'TransactionAmt_to_std_card4', 'TransactionAmt_to_mean_card2', 'TransactionAmt_to_mean_card3',
#        'TransactionAmt_to_std_card2', 'TransactionAmt_to_std_card3', 'TransactionAmt_to_mean_card5',
#        'TransactionAmt_to_mean_card6', 'TransactionAmt_to_std_card5', 'TransactionAmt_to_std_card6',
#        'TransactionAmt_to_mean_id_19', 'TransactionAmt_to_mean_id_20', 'TransactionAmt_to_std_id_19',
#        'TransactionAmt_to_std_id_20', 'id_02_to_mean_card1', 'id_02_to_mean_card4', 'id_02_to_std_card1',
#        'id_02_to_std_card4', 'D15_to_mean_card1', 'D15_to_mean_card4', 'D15_to_std_card1', 'D15_to_std_card4',
#        'D15_to_mean_addr1', 'D15_to_std_addr1', 'TransactionAmt_Log', 'TransactionAmt_decimal',
#        'Transaction_day_of_week', 'Transaction_hour', 'id_02__id_20', 'id_02__D8', 'D11__DeviceInfo',
#        'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 'card2__dist1', 'card1__card5', 'card2__id_20',
#        'card5__P_emaildomain', 'addr1__card1', 'uid3__P_emaildomain', 'address_match__R_emaildomain', 'card1_freq',
#        'card2_freq', 'card3_freq', 'card4_freq', 'card5_freq', 'card6_freq', 'id_36_freq', 'addr1_freq',
#        'ProductCD_freq', 'id_01_freq', 'D1_freq', 'D3_freq', 'D4_freq', 'D5_freq', 'D6_freq', 'D8_freq', 'D9_freq',
#        'D10_freq', 'D11_freq', 'D12_freq', 'D13_freq', 'D14_freq', 'D15_freq', 'id_01_count_dist', 'id_31_count_dist',
#        'id_33_count_dist', 'id_36_count_dist', 'P_emaildomain_bin', 'P_emaildomain_suffix', 'R_emaildomain_bin',
#        'R_emaildomain_suffix']
#
# for d in clo:
#     train.set_index('TransactionDT')[d].plot(style='.', title=d)
#     test.set_index('TransactionDT')[d].plot(style='.', title=d)
#     plt.show()


# 画图分析异常值
# train = pd.read_pickle('train_data.pkl')
# test = pd.read_pickle('test_data.pkl')

# train['day_num'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1))
# train['day_d_max'] = train['day_num'].map(train.groupby(['day_num'])['D3'].max())
# test['day_num'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1))
# test['day_d_max'] = test['day_num'].map(test.groupby(['day_num'])['D3'].max())
# print('train')
# print(train['day_d_max'].unique())
# print(train['day_d_max'].nunique())
# print(train['day_d_max'].max(), train['day_d_max'].min(), train['day_d_max'].max() - train['day_d_max'].min())
# print('test')
# print(test['day_d_max'].unique())
# print(test['day_d_max'].nunique())
# print(test['day_d_max'].max(), test['day_d_max'].min(), test['day_d_max'].max() - test['day_d_max'].min())
# train.set_index('TransactionDT')['day_d_max'].plot(style='.', title=d)
# test.set_index('TransactionDT')['day_d_max'].plot(style='.', title=d)
# plt.show()


train = pd.read_pickle('./input/train_transaction.pkl')
test = pd.read_pickle('./input/test_transaction.pkl')

# sns.distplot(train['TransactionAmt'], hist=True, kde=True, color='red')
# sns.distplot(test['TransactionAmt'], hist=True, kde=True, color='yellow')
# plt.show()
#
# sns.distplot(train['TransactionAmt'].apply(np.log1p), hist=True, kde=True, color='red')
# sns.distplot(test['TransactionAmt'].apply(np.log1p), hist=True, kde=True, color='yellow')
# plt.show()

# fig, (axis1, axis2) = plt.subplots(2, 1, sharex=True)
# sns.distplot(train['TransactionAmt'].apply(np.log1p), hist=True, kde=True, color='red', ax=axis1)
# sns.distplot(test['TransactionAmt'].apply(np.log1p), hist=True, kde=True, color='yellow', ax=axis2)
# plt.show()



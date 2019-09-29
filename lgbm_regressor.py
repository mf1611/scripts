import pandas as pd
import numpy as np
import datetime
import yaml
import argparse
import gc
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('-p', help='yaml file of lgbm parameters')

args = parser.parse_args()


# read df_train, df_test
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# features
cols_rm = ['target']
features = [col for col in df_train.columns if col not in cols_rm]

X_train = df_train[features]
X_test = df_test[features]
y_train = df_train['target']


# fold predictions
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=16)
oof = np.zeros(len(X_train))
pred = np.zeros(len(X_test))
feature_importance_df = pd.DataFrame()


# read params
with open(args.p, 'r') as f:
    params = yaml.load(f)


for fold_, (tr_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):
    print('-')
    print("Fold {}".format(fold_ + 1))

    tr_data = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train[val_idx])

    num_round = 10000
    clf = lgb.train(params, tr_data, num_round, valid_sets = [tr_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    oof[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=clf.best_iteration)
    pred += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

val_score = np.sqrt(mean_squared_error(y_train, oof))
print('CV_RMSE: ', val_score)

# output importance
feature_importance_mean = feature_importance_df[['feature','importance']].groupby('feature').mean().sort_values(by="importance", ascending=False)
print(feature_importance_mean.head(20))
feature_importance_mean.to_csv("../importance/lgbm_{}.csv".format(val_score), index=False)

# output prediction
df_submit = pd.DataFrame({"id": X_test["id"].values})
df_submit["target"] = pred
df_submit.to_csv("../output/submit_val_{}.csv".format(val_score), index=False)
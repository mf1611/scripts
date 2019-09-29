import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling

# read file
df = pd.read_csv('./input/sample.csv')
print(df.shape)
display(df.sample(20))

# pandas-profiling, 重いので注意
# profile = train.profile_report()
# profile.to_file('../input/train_profile.html')


# 重複行の確認，もし0でないなら処理
print('重複行数: ', df.duplicated().sum())
df = df.loc[~df.duplicated(),:]


# 欠損値の確認
missing_dict = df.isnull().sum()
missing_rate = missing_dict.values / df.shape[0]
df_missing = pd.DataFrame({'feature': missing_dict.index, '#_of_missing': missing_dict.values, 'missing_rate': missing_rate}).sort_values(by='missing_rate', ascending=False)
df_missing = df_missing.set_index('feature')
display(df_missing.head(20))

# カラムごとのuniqueな数
display('ユニーク数: ', df.nunique().sort_values(ascending=False))

# 中身の確認
for col in df.columns:
    print('-'*60)
    print(col)
    print('# of unique: ', df[col].nunique())
    print('statistics: ', df[col].describe())
    if df[col].dtype=='object':
        print('value_counts: ')
        val_cnt = df[col].value_counts(dropna=False, normalize=True).sort_values(ascending=False)
        display(val_cnt)
        sns.barplot(x=val_cnt.index, y=val_cnt.values)
        plt.show()
    else:
        # sns.distplot(df[col].dropna())
        # plt.show()

        # log10
        df_rm0 =df[df[col]!=0][col]
        sns.distplot(np.log10(df_rm0).dropna())
        plt.show()


# 相関行列のheatmap，重いので注意
sns.heatmap(df.corr(), vmax=1, vmin=-1, center=0)
plt.show()
#plt.savefig('figure/heatmap_corr.png')


##############################################################
# 目的変数との相関
corr_list = []
col_list = []
for col in df.columns:
    if col not in ['target']:
        col_list.append(col)
        corr_list.append(np.corrcoef(df[col], df['target'])[0,1])
df_corr = pd.DataFrame({'feature': col_list, 'corr_with_object': corr_list}).sort_values(by='corr_with_object', ascending=False)
display('目的変数との相関: ', df_corr.head(20))


# 目的変数ごとの各カラムの分布の確認
# ここでは，2値変数を例に
for col in df.columns:
    print('-'*60)
    print(col)
    if df[col].dtype=='object':
        print('value_counts: ')
        val_cnt_0 = df.loc[df['target']==0, col].value_counts(dropna=False, normalize=True).sort_values(ascending=False)
        val_cnt_1 = df.loc[df['target']==1, col].value_counts(dropna=False, normalize=True).sort_values(ascending=False)
        sns.barplot(x=val_cnt_0.index, y=val_cnt_0.values, color='r', label='0')
        sns.barplot(x=val_cnt_1.index, y=val_cnt_1.values, color='b', label='1')
        plt.legend()
        plt.show()
    else:
        # sns.distplot(df.loc[df['target]==0, col].dropna(), color='r', label='0')
        # sns.distplot(df.loc[df['target]==1, col].dropna(), color='b', label='1')
        # plt.legend()
        # plt.show()

        # log10
        df_0_rm0 =df[((df['target']==0) & (df[col]!=0))][col]
        df_1_rm0 =df[((df['target']==1) & (df[col]!=0))][col]
        sns.distplot(np.log10(df_0_rm0).dropna(), color='r', label='0')
        sns.distplot(np.log10(df_1_rm0).dropna(), color='b', label='1')
        plt.legend()
        plt.show()









#############################################################
# 意味のないかもしれないものの処理
# - 1つしか値を持たないカラム
# - 90%以上欠損のカラム
#- 全体の90%以上が最頻値のカラム
#############################################################
# one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]

# many_null_cols = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.9]

# big_top_value_cols = [col for col in df.columns if df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

# cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
# print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))

# df.drop(cols_to_drop, axis=1, inplace=True)
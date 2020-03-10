import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling

# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", 200)


# # エクセルを読み込む場合
# !pip install xlrd
# df = pd.read_excel('./input/sample.csv')


# read file
df = pd.read_csv('./input/sample.csv')
print(df.shape)
display(df.sample(20))

# pandas-profiling, 重いので注意
# profile = train.profile_report()
# profile.to_file('../input/train_profile.html')


# 型確認
display(df.dtypes)

# 数値カラム
display('数値カラム：', list(df.select_dtypes(include='number').columns))

# datetime型
df['日時'] = pd.to_datetime(df['日時'])
# 細かい時間まで必要ない場合
df['日時'] = df['日時'].apply(lambda x: x.strftime('%Y%m%d'))


# 重複行の確認，もし0でないなら処理
print('重複行数: ', df.duplicated().sum())
df = df.loc[~df.duplicated(),:]


# 欠損値の確認
missing_dict = df.isnull().sum()
missing_rate = missing_dict.values / df.shape[0]
df_missing = pd.DataFrame({'カラム': missing_dict.index, '欠損数': missing_dict.values, '欠損割合': missing_rate}).sort_values(by='欠損割合', ascending=False)
df_missing = df_missing.set_index('カラム')

# 欠損割合可視化
plt.figure(figsize=(20, 10))
sns.barplot(df_missing.index, df_missing['欠損割合'])
plt.xlabel("カラム", fontsize=18)
plt.ylabel("欠損割合", fontsize=18)
plt.xticks(fontsize=18, rotation=90)
plt.yticks(fontsize=18)
plt.show()
 
# 1つ以上欠損あるものだけ表示
display(df_missing[df_missing['欠損数']>0])

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
        sns.distplot(df[col].dropna())
        plt.show()
        
        # 箱ひげ図
        print('-'*60)
        df[col].plot.box()
        plt.show()

#         # log10
#         df_rm0 =df[df[col]!=0][col]
#         sns.distplot(np.log10(df_rm0).dropna())
#         plt.show()


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
#         plt.xticks(rotation=90)
#         plt.yticks(fontsize=18)
        plt.show()
    else:
        sns.distplot(df.loc[df['target]==0, col].dropna(), kde=True, norm_hist=True, color='r', label='0')
        sns.distplot(df.loc[df['target]==1, col].dropna(), kde=True, norm_hist=True, color='b', label='1')
        plt.legend()
#         plt.xlabel("値", fontsize=18)
#         plt.ylabel("頻度割合", fontsize=18)
#         plt.xticks(fontsize=18)
#         plt.yticks(fontsize=18)
        plt.legend(fontsize=18)
        plt.show()

#         # log10
#         df_0_rm0 =df[((df['target']==0) & (df[col]!=0))][col]
#         df_1_rm0 =df[((df['target']==1) & (df[col]!=0))][col]
#         sns.distplot(np.log10(df_0_rm0).dropna(), kde=True, norm_hist=True, color='r', label='0')
#         sns.distplot(np.log10(df_1_rm0).dropna(), kde=True, norm_hist=True, color='b', label='1')
#         plt.legend()
#         plt.show()

                     
                               
                               

# 時系列プロットの場合
# indexがdatetime型だと仮定
plt.figure(figsize=(20, 50))
df[col].plot()
plt.legend(fontsize=15)
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15)
plt.xlabel("datetime", fontsize=15)
plt.ylabel("値", fontsize=15)
# plt.rcParams["font.size"] = 15

                               
                               

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

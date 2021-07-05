import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from sklearn.metrics import roc_auc_score, auc, precision_score, recall_score, roc_curve, precision_recall_curve


# グラフ表示用の関数
def count_categorical_values(df_, col, fontsize=15):
    """カテゴリカル変数の頻度分布を表示"""
    val_cnt = df_[col].value_counts(dropna=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(val_cnt.index, val_cnt.values)
    plt.xticks(rotation=45, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('件数', fontsize=fontsize)
    plt.show()
    return


def count_cntinuousl_values(df_, col, log10=False, fontsize=15):
    """連続値変数の頻度分布を表示"""
    plt.figure(figsize=(10, 5))
    if log10:
        sns.distplot(df_[col].apply(lambda x: np.log10(x)), kde=False)
    else:
        sns.distplot(df_[col], kde=False)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(col, fontsize=fontsize)
    plt.ylabel('件数', fontsize=fontsize)
    plt.grid(color = "gray", linestyle="--")
    plt.show()
    return
    


def count_date_plot(df_, col_date):
    cnt_date = df[[col_date]].reset_index().groupby(col_date).count().sort_index(by='index', ascending=False)
    cnt_date.columns = ['count']
	
    #plt.figure(figsize=(30, 10))
    cnt_date.plot(figsize=(30, 10))
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20)
    plt.xlabel("date", fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.show()



def pos_neg_distplot(df_, target_col, cols_contib):
    """正例負例毎の各カラムの分布比較"""
    df_pos = df_[df_[target_col]==1].copy()
    df_neg = df_[df_[target_col]==0].copy()
    for col in cols_contib:
        print(col)
    	weight_pos = np.ones(df_pos[col].dropna().shape[0])/float(df_pos[col].dropna().shape[0])
	weight_neg = np.ones(df_neg[col].dropna().shape[0])/float(df_neg[col].dropna().shape[0])
	x_max = df[col].max()
	x_min = df[col].min()
	range_bin_width = np.linspace(x_min, x_max, 20)
	sns.distplot(df_pos[col], kde=False, hist_kws={'weights': weight_pos}, bins=range_bin_width, label='positive', color='r')
	sns.distplot(df_neg[col], kde=False, hist_kws={'weights': weight_neg}, bins=range_bin_width, label='negative', color='b')
	plt.legend()
	plt.grid(color = "gray", linestyle="--")
	plt.show()
    return


# 正例と負例のスコア分布
def display_score_dist(y, p, fontsize=15):
    score = pd.DataFrame()
    score['pred'] = p
    score['label'] = y.values

    score_pos = score.loc[score['label']==1, 'pred']
    score_neg = score.loc[score['label']==0, 'pred']
    range_bin_width = np.linspace(0, 1, 50)  # 2%刻みのビン幅
    weights_pos = np.ones(score_pos.shape[0])/float(len(score_pos))
    weights_neg = np.ones(score_neg.shape[0])/float(len(score_neg))
    plt.figure(figsize=(10, 5))
    sns.distplot(score_pos, kde=False, bins=range_bin_width, hist_kws={'weights': weights_pos}, label='positive', color='r')
    sns.distplot(score_neg, kde=False, bins=range_bin_width, hist_kws={'weights': weights_neg}, label='negative', color='b')
    plt.xlabel("スコア", fontsize=fontsize)
    plt.ylabel("頻度割合", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper center')
    plt.grid(color = "gray", linestyle="--")
    plt.show()
    return



# ROC_Curve
def plot_auc(y, p, fontsize=15):
    fpr, tpr, thresholds = roc_curve(y, p)
    print('ROC-AUC:', roc_auc_score(y, p))
#     plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR（偽陽性率）", fontsize=fontsize)
    plt.ylabel("TPR（真陽性率(再現率)）", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(color = "gray", linestyle="--")
    plt.show()
    return
    

# PR_Curve
def plot_pr(y, p, fontsize=15):
    precision, recall, thresholds = precision_recall_curve(y, p)
    print('PR-AUC:', auc(recall, precision))
#     plt.figure(figsize=(10, 5))
    plt.plot(recall, precision)
    plt.xlabel("recall", fontsize=fontsize)
    plt.ylabel("precision", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(color = "gray", linestyle="--")
    plt.show()
    return
    


def display_confusion_matrix(y, p, thresh=0.95):
	"""指定閾値で切った場合の混同行列表示"""
    precision, recall, thresholds = precision_recall_curve(y, p)
    idx_recall = np.where(recall>=thresh)[0].max()
    thresh_recall = thresholds[idx_recall]
    y_pred_thresh = np.where(p>=thresh_recall, 1, 0)  # 指定閾値の下で定めた、0/1の予測結果
    tn, fp, fn, tp = confusion_matrix(y, y_pred_thresh).ravel()
    print('FP削減率：', round(tn/(tn+fp), 3))
    print('混同行列：')
    print(np.array([[tp, fn], [fp, tn]]))
    return




def plot_histgram(counter_values, bin_width, min_value, max_value, val_log10=False, normalize=False):
    """
    指定したビン幅でのヒストグラムを作成
    - 最初は細かなパラメータの当たりをつけるのは難しいので、まずは、sns.distplot(counter_values)で見てみてから、この関数を用いるのが良い
    
    counter_values: ヒストグラムの元とするvalue部
    - 例えば、pd.DataFrameの場合、対応する列を、上記にそれぞれ代入すればいい
    
    bin_width: ビンの幅
    min_value: 最小値
    max_value: 最大値
    val_log10: 常用対数取るかどうか(bool)
    normalize: ヒストグラムの縦軸を正規化するかどうか(bool)
    """
    # ビンの数, max_value以上のビンも加えるため、+1
    num_bins = int((max_value - min_value) / bin_width) + 1

    # histgramのcounter_dictの初期化
    hist_count = {}
    for i in range(num_bins):
        hist_count[min_value+bin_width*i] = 0

    # 各ビンに入る場合に+1していく
    for val in counter_values:
        if val_log10:
            val = np.log10(val)

        for i in range(num_bins):
            if val>=max_value:
                hist_count[max_value] += 1
            elif (val>=min_value+bin_width*i) & (val<min_value+bin_width*(i+1)): 
                hist_count[min_value+bin_width*i] += 1

    df_ = pd.DataFrame(columns=['value', 'count'])
    df_['value'] = hist_count.keys()
    df_['count'] = hist_count.values()
    df_['count_norm'] = df_['count'] / df_['count'].sum()
    df_ = df_.sort_values(by='value', ascending=True)

    if not normalize:
        sns.barplot(x=df_['value'], y=df_['count'], palette="Blues_d")
        plt.show()
    else:
        sns.barplot(x=df_['value'], y=df_['count_norm'], palette="Blues_d")
        plt.show()
    
    return df_

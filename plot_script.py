import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from sklearn.metrics import roc_auc_score, auc, precision_score, recall_score, roc_curve, precision_recall_curve


# �O���t�\���p�̊֐�
def count_categorical_values(df_, col, fontsize=15):
    """�J�e�S���J���ϐ��̕p�x���z��\��"""
    val_cnt = df_[col].value_counts(dropna=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(val_cnt.index, val_cnt.values)
    plt.xticks(rotation=45, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('����', fontsize=fontsize)
    plt.show()
    return


def count_cntinuousl_values(df_, col, log10=False, fontsize=15):
    """�A���l�ϐ��̕p�x���z��\��"""
    plt.figure(figsize=(10, 5))
    if log10:
        sns.distplot(df_[col].apply(lambda x: np.log10(x)), kde=False)
    else:
        sns.distplot(df_[col], kde=False)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(col, fontsize=fontsize)
    plt.ylabel('����', fontsize=fontsize)
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
	"""���ᕉ�ᖈ�̊e�J�����̕��z��r"""
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
	    plt.show()
	return


# ����ƕ���̃X�R�A���z
def display_score_dist(y, p, fontsize=15):
    score = pd.DataFrame()
    score['pred'] = p
    score['label'] = y.values

    score_pos = score.loc[score['label']==1, 'pred']
    score_neg = score.loc[score['label']==0, 'pred']
    range_bin_width = np.linspace(0, 1, 50)  # 2%���݂̃r����
    weights_pos = np.ones(score_pos.shape[0])/float(len(score_pos))
    weights_neg = np.ones(score_neg.shape[0])/float(len(score_neg))
    plt.figure(figsize=(10, 5))
    sns.distplot(score_pos, kde=False, bins=range_bin_width, hist_kws={'weights': weights_pos}, label='positive', color='r')
    sns.distplot(score_neg, kde=False, bins=range_bin_width, hist_kws={'weights': weights_neg}, label='negative', color='b')
    plt.xlabel("�X�R�A", fontsize=fontsize)
    plt.ylabel("�p�x����", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper center')
    plt.show()
    return



# ROC_Curve
def plot_auc(y, p, fontsize=15):
    fpr, tpr, thresholds = roc_curve(y, p)
    print('ROC-AUC:', roc_auc_score(y, p))
#     plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR�i�U�z�����j", fontsize=fontsize)
    plt.ylabel("TPR�i�^�z����(�Č���)�j", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
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
    plt.show()
    return
    


def display_confusion_matrix(y, p, thresh=0.95):
	"""�w��臒l�Ő؂����ꍇ�̍����s��\��"""
    precision, recall, thresholds = precision_recall_curve(y, p)
    idx_recall = np.where(recall>=thresh)[0].max()
    thresh_recall = thresholds[idx_recall]
    y_pred_thresh = np.where(p>=thresh_recall, 1, 0)  # �w��臒l�̉��Œ�߂��A0/1�̗\������
    tn, fp, fn, tp = confusion_matrix(y, y_pred_thresh).ravel()
    print('FP�팸���F', round(tn/(tn+fp), 3))
    print('�����s��F')
    print(np.array([[tp, fn], [fp, tn]]))
    return
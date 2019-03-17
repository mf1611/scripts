import pandas as pd

# 期間集計した特徴量の追加
# 基本的に、1つのIDに対して複数のレコードが存在するdataframeに使うもの

def features_agg(df, id_col, agg_func):
    """
    注意として、入力のdfに対して、出力は集計されたid_colに対して集計されたdataframeを返す
    agg_func: {col名: ['mean', 'max', 'min', 'std',...], }のdict
    """
    df_agg = df.groupby(id_col).agg(agg_func)
    df_agg.columns = ['_'.join(col) for col in df_agg.columns.values]
    df_agg.reset_index(inplace=True)

    return df_agg


def features_period_agg(df, id_col, datetime_col, period, agg_func):
    """
    datetime型のカラムdatetime_colがある場合の期間集計
    periodを指定して、その期間ごとの集計された特徴量を返す

    period: '2d', '2w', '1m', '1y' など
    """
    df_agg = df.groupby(pd.Grouper(key=datetime_col, freq=preriod)).agg(agg_func)
    df_agg.columns = ['_'.join(col) + f'_{period}' for col in df_agg.columns.values]
    df_agg.reset_index(inplace=True)

    return df_agg

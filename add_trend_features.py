import pandas as pd
from sklearn.linear_model import LinearRegression


def extract_trend(group):
    """
    groupbyしたオブジェクトに対して用いる関数
    groupは、IDでgroupbyされた1つのfeatureを想定、また時間などの順序
    そして、そのfeatureのtrendを求める
    """
    y = group.values

    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan

    return trend



def features_group_trend(df, id_col, sort_col, feature):
    """
    ここでは、ある期間に絞られたdataframeに対して、あるfeatureの値のトレンドを特徴量として抽出する
    注意として、1つのIDに対して、1つのトレンド特徴量を持つdataframeを返す

    df: ある期間に絞られたdataframeを想定、1つのIDに対して複数レコードを想定
    id_col: IDとなるカラム名
    sort_col: ソートしたいカラム名、例えば日付でソートとか
    feature: トレンドを抽出したいカラム名
    """
    df = df.sort_values(by=sort_col, ascending=False)  # ある列の値でソートする
    trend = df.groupby(id_col)[feature].apply(extract_trend).reset_index().rename(
                                        columns={feature: feature+'_trend'})

    return trend

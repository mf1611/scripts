import pandas as pd

# 指定したグループカラムで、集計した特徴量の追加
# 注意として、これは１つのIDに対して1レコードのdataframeにも使えるもの

def add_group_agg_features(df, group_cols, counted_col, calcs):
    """
    group_cols: [col1, col2,...,coln] グループとして指定するカラムのリスト
    counted_col: グループ集計したいカラム名
    calcs: 集計操作名のリスト、例えば、['mean', 'median', 'max', 'min', 'var'
    """
    for calc in calcs:
        gp = df[group_cols + [counted_col]].groupby(group_cols)[counted_col].agg(calc).reset_index().rename(
                                            columns={counted_col: counted_col+'_G'+calc.upper()})
        df = df.merge(gp, on=group_cols, how='left')
        del gp
        gc.collect()

    return df

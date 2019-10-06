import MeCab
import neologdn  # 表記ゆれの是正をしてくれるlibrary

"""
・WindowsでのMeCabのインストール
https://www.gis-py.com/entry/mecab

1か2でダウンロードして，インストール実行
        1. 公式サイトからMeCabの実行プログラム本体をダウンロード（32bit版）http://taku910.github.io/mecab/#download
        2. 有志がビルドした64bit版をダウンロードhttps://github.com/ikegami-yukino/mecab/releases/tag/v0.996
インストールしたpathを環境変数に設定
    - MeCabのインストール先/binを追加

・Python上でMeCabを使えるようにする
    1. Python上のMeCabバインディングの導入
        $ pip install ipykernel
        $ pip install mecab-python-windows
    2. libmecab.dll をコピペ
        MeCabインストール先/binの中のlibmecab.dllを，(Pyhthonのインストール先)/Lib/site-packagesにコピペする

・Windows上でのneolgd使用
https://www.pytry3g.com/entry/MeCab-NEologd-Windows
- WSL上で，neologdをインストールして，その辞書を，windowsのC直下に置く
    - $ sudo cp /usr/lib/mecab/dic/mecab-ipadic-neologd/* /mnt/c/neologd/
"""

sample = "8月3日に放送された「中居正広の金曜日のスマイルたちへ」"

# 形態素解析
tagger_chasen = MeCab.Tagger(r"-Ochasen -d C:\neologd")
result = tagger_chasen.parse(sample)
print(result)

# 分かち書き
tagger_wakati = MeCab.Tagger(r"-Owakati -d C:\neologd")
result = tagger_wakati.parse(sample)
print(result)


# dataframeによる品詞情報の格納
# 品詞(type)での絞り込み、不要な品詞の除去など
# https://datumstudio.jp/blog/pythonによる日本語前処理備忘録
import pandas as pd

def parse_text(text: str):
    parsed_text = tagger_chasen.parse(text).split('\n')  # 改行で区切る
    parsed_results = pd.Series(parsed_text).str.split('\t').tolist()  # 半角区切りでリストにし、それをSeriesにする
    df = pd.DataFrame.from_records(parsed_results)
    columns = ['surface', 'spell', 'orig', 'type', 'katsuyoukei', 'katsuyoukata']
    df.columns = columns
    return df.query("surface != 'EOS'").query("surface != ''")


print(parse_text('庭には二羽鶏がいる'))


# 名詞を中心とした形態素のみを抽出する関数
def extract_noun(text: str):
    norm_text = neologdn.normalize(text)
    parsed = parse_text(norm_text)
    noun_df = parsed[
        parsed.type.str.startswith('名詞-一般') | 
        parsed['type'].str.startswith('名詞-固有名詞') |
        parsed.type.str.startswith('名詞-サ変接続') |
        parsed.type.str.startswith('名詞-形容動詞語幹')
    ]
    return ' '.join(noun_df.orig.tolist())


text = '私達はラーメンが大好きです。'
print(parse_text(text))

split_text = extract_noun(text)
print(split_text)


# Bag of Words
# from sklearn.feature_extraction.text import CountVectorizer
# # ワードカウント疎行列の抽出
# cv = CountVectorizer()
# cv.fit_transform()
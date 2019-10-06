import MeCab

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

sample = "8月3日に放送された「中居正広の金曜日のスマイルたちへ」\n　\
            私は東京に住んでいました。\n \
            私は横浜に住んでいました。"

# 形態素解析
tagger = MeCab.Tagger(r"-Ochasen -d C:\neologd")
result = tagger.parse(sample)
print(result)

# 分かち書き
tagger = MeCab.Tagger(r"-Owakati -d C:\neologd")
result = tagger.parse(sample)
print(result)

import MeCab
import argparse
import gensim
from gensim.models import word2vec

"""
参考：
https://m0t0k1ch1st0ry.com/blog/2016/08/28/word2vec/
"""

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input_wakati_text')
parser.add_argument('-o', help='output_model')
args = parser.parse_args()

# 1行ごとに読み込む(1行1センテンスのデータ)
# 改行気にしないで、1つのものとして読み込ます場合は, Text8Corpus()を用いる
sentences = word2vec.LineSentence(args.i)

model = word2vec.Word2Vec(
        sentences=sentences,
        size=200,
        min_count=1,
        window=10,
        sg=1,  # 1:Skip-gram, 0: CBOW
        hs=0,  # 0: Negative Sampling, 1: Hierachical Softmax
        negative=5,
        seed=0
)
model.save(args.o)

# example:
# $ python w2v_basic.py -i ./output/roujinto_umi.txt -o output/rojinto_umi.model
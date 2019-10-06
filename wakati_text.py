import MeCab
import argparse

"""
参考：
https://m0t0k1ch1st0ry.com/blog/2016/08/28/word2vec/
"""

parser = argparse.ArgumentParser()

parser.add_argument('-i', help='input_text')
parser.add_argument('-o', help='output_text')

args = parser.parse_args()

tagger = MeCab.Tagger(r"-Owakati -d C:\neologd")

input = open(args.i, "r", encoding='utf-8')
output = open(args.o, "w", encoding='utf-8')

line = input.readline()
while line:
    result = tagger.parse(line)
    output.write(result[1:])  # skip first \s
    line = input.readline()

input.close()
output.close()

# example: 
# $ python .\wakati_text.py -i .\input\rojinto_umi.txt -o output/roujinto_umi.txt
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import csv

# NLTKデータのダウンロード
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ファイルパスの設定
input_file = "dataset_text/Pereira_sentence.csv"
output_file = "dataset_pos/Pereira_sentence_POS.csv"

# CSVファイルの読み込み
df = pd.read_csv(input_file)

# 品詞タグ付けを適用する関数
def pos_tag_text(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    # タグだけを抽出し括弧付きにする
    tags_only = [f"('{tag}')" for _, tag in pos_tags]
    return " ".join(tags_only)

# 'FormattedText'列に品詞タグ付けを適用
df['POS_tag'] = df['FormattedText'].apply(pos_tag_text)

# 結果をCSVに保存
df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f"品詞タグだけを括弧付きで保存しました: {output_file}")

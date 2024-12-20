import nltk
import pandas as pd
import csv

# 必要なNLTKデータをダウンロード
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ファイルパス
input_file = "dataset_text/Huth_sentence.csv"
output_file = "dataset_pos/Huth_sentence_with_POS.csv"

# CSVファイルを読み込む
df = pd.read_csv(input_file)

# 品詞タグ付け用の関数
def pos_tag_text(text):
    tokens = nltk.word_tokenize(text)  # トークン化
    pos_tags = nltk.pos_tag(tokens)   # 品詞タグ付け
    return pos_tags

# 各行のテキストに対して品詞タグ付けを適用
df['POS_Tags'] = df['FormattedText'].apply(pos_tag_text)

# タグ情報を文字列に変換して保存
df['POS_Tags'] = df['POS_Tags'].apply(lambda x: " ".join([f"{word}/{tag}" for word, tag in x]))

# 結果を新しいCSVファイルに保存
df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)

print(f"品詞タグ付けされたデータが {output_file} に保存されました。")

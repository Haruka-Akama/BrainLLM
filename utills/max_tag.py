import csv

# ファイルパスを指定
data_file = 'dataset_pos/Pereira_sentence_POS.csv'

# 最大POSタグ数を記録する変数
max_pos_count = 0

# ファイルを読み込み
with open(data_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # POS_tag列を解析してタグの数を数える
        pos_tags = row['POS_tag'].strip("() ").split(') (')
        max_pos_count = max(max_pos_count, len(pos_tags))

# 結果を出力
print(f"POSタグの最大数: {max_pos_count}")

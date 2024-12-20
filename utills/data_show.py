import pickle

# ファイルを読み込む
with open("dataset/Huth/1.pca1000.wq.pkl.dic", "rb") as file:
    data = pickle.load(file)

# # 内容を確認
# print(type(data))  # データ型を確認

# # ラベルを確認
# if isinstance(data, dict):
#     print("ラベル一覧:", data.keys())  # 辞書型の場合はキーを表示
# elif isinstance(data, list):
#     print(f"リスト内のラベル数: {len(data)}")
#     print("最初のラベル例:", data[0])  # リスト型の場合は最初の要素を表示
# else:
#     print("データの内容:", data)  # その他のデータ型の場合は全体を表示
# print(data)        # データの中身を確認

# 特定のラベルに関連するデータを取得
label_data = data['exorcism']

# ラベルに関連するすべての情報を表示
print("キー167のラベルに関連する情報:")
for key, value in label_data.items():
    print(f"{key}: {value}")


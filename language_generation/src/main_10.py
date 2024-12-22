from config_00 import get_config
from data_01 import FMRI_dataset
import pickle
import random
import numpy as np
import torch
import json
from model_09 import Decoding_model
from post_hoc_evaluate_08 import save_evaluation_results
import os
from post_hoc_evaluate_08 import compare
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def initialize_dataset(args, dataset_name, subject_name):
    """
    データセットの初期化
    """
    dataset_path = args['dataset_path']
    if 'Huth' in args['task_name']:
        input_dataset = pickle.load(open(f'dataset/{dataset_name}/{subject_name}.wq.pkl', 'rb'))
    elif 'Pereira' in args['task_name']:
        input_dataset = pickle.load(open(f'dataset/{dataset_name}/{subject_name}.wq.pkl', 'rb'))
    elif 'Narratives' in args['task_name']:
        u2s = json.load(open('dataset_info/u2s.json'))
        args['Narratives_stories'] = u2s[f'sub-{subject_name}']
        input_dataset = {story_name: pickle.load(open(f'{dataset_name}/{story_name}.wq.pkl', 'rb'))
        for story_name in args['Narratives_stories']}
    else:
        raise ValueError("Unsupported task name in args['task_name']")

    return input_dataset

if __name__ == '__main__':
    # 設定の取得
    args = get_config()
    # print("設定:", args)

    save_name = 'language_generation/results/'
    for key in args.keys():
        if key not in ['cuda']:
            save_name += key+'('+str(args[key])+')_'
    save_name = save_name[:-1]
    dataset_name = args['task_name'].split('_')[0]
    subject_name = args['task_name'].split('_')[1]
    if 'example' not in args['task_name']:
        args['dataset_path'] = os.path.join(args['dataset_path'], dataset_name)
    dataset_path = args['dataset_path']

    # データセットクラスとタスク名
    dataset_class = FMRI_dataset
    dataset_name, subject_name = args['task_name'].split('_')
    input_dataset = initialize_dataset(args,dataset_name, subject_name)
 # デバッグ関数の追加
    def debug_content_prev(self, num_samples=5):
        """
        `content_prev` がデータセット内で正しく処理されているか確認します。
        """
        print("Debugging `content_prev` in Dataset:")
        for idx, sample in enumerate(self.inputs[:num_samples]):  # 最初の num_samples 件をチェック
            print(f"Sample {idx}:")
            print(f"  content_prev (decoded): {self.tokenizer.decode(sample['content_prev'])}")
            print(f"  content_prev (token IDs): {sample['content_prev']}")
        print("Dataset check completed.")

    # FMRI_dataset クラスにデバッグ関数を組み込む
    FMRI_dataset.debug_content_prev = debug_content_prev

    # 必要なモデル・データセットの初期化
    # POSタグの処理
    pos_tags_ids, pos_tag_to_id, id_to_pos_tag = FMRI_dataset.process_pos_tags(args['pos_csv_path'])

    # デコーディングモデルの初期化
    decoding_model = Decoding_model(args, pos_tag_to_id)

    # データセットの初期化
    dataset = FMRI_dataset(input_dataset, args, tokenizer=decoding_model.tokenizer)

    # デバッグ関数の呼び出し
    dataset.debug_content_prev()



    print('データセット初期化完了')

if args['mode'] in ['train', 'only_train', 'all']:
    # トレーニングモード
    decoding_model.train(dataset.train_dataset, dataset.valid_dataset)

if args['mode'] in ['acc', 'train']:
    # 精度確認モード
    decoding_model.args['load_check_point'] = True
    decoding_model.load_check_point()
    decoding_model.prompt_model.check_point = decoding_model.check_point
    decoding_model.prompt_model.init_encoding_model()

    # テストデータでの検証
    loss_list = decoding_model.valid(dataset.test_dataset)

    # 入力方式を変更して再検証
    args['input_method'] = 'permutated'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = dataset_class(input_dataset, args, tokenizer=decoding_model.tokenizer, decoding_model=decoding_model)
    loss_list_baseline = decoding_model.valid(dataset.test_dataset)

    # ペアワイズ比較
    from post_hoc_evaluate_08 import compare
    pairwise_list = [compare(np.array(loss_list[idx]), np.array(loss_list_baseline[idx])) for idx in range(len(loss_list_baseline))]
    print(f"Pairwise accuracy: {np.sum(pairwise_list)/len(loss_list_baseline):.4f}")

if args['mode'] in ['train', 'only_train', 'all']:
    # トレーニングモード
    decoding_model.train(dataset.train_dataset, dataset.valid_dataset)

if args['mode'] in ['acc', 'train']:
    # 精度確認モード
    decoding_model.args['load_check_point'] = True
    decoding_model.load_check_point()
    decoding_model.prompt_model.check_point = decoding_model.check_point
    decoding_model.prompt_model.init_encoding_model()

    # テストデータでの検証
    loss_list = decoding_model.valid(dataset.test_dataset)

    # 入力方式を変更して再検証
    args['input_method'] = 'permutated'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = dataset_class(input_dataset, args, tokenizer=decoding_model.tokenizer, decoding_model=decoding_model)
    loss_list_baseline = decoding_model.valid(dataset.test_dataset)

    # ペアワイズ比較
    from post_hoc_evaluate_08 import compare
    pairwise_list = [compare(np.array(loss_list[idx]), np.array(loss_list_baseline[idx])) for idx in range(len(loss_list_baseline))]
    print(f"Pairwise accuracy: {np.sum(pairwise_list)/len(loss_list_baseline):.4f}")

if args['mode'] in ['all', 'evaluate']:
    # テストモード
    decoding_model.args['load_check_point'] = True
    decoding_model.load_check_point()

    # テストデータでの評価
    test_results = decoding_model.test(dataset.test_dataset, args['output'])

    # 必要に応じて保存
    if args['save_results']:
        # print("Results being saved:", test_results)
        save_evaluation_results(test_results, decoding_model.args['checkpoint_path'], file_name="test_results")

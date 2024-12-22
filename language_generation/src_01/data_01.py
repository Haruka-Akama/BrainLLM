import pickle
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import random
import gc
import json
import copy
import csv

class MyStandardScaler:
    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
try:
    from sklearn.preprocessing import StandardScaler
except:
    StandardScaler = MyStandardScaler

class Splited_FMRI_dataset(Dataset):
    # def debug_pos_tags_in_dataloader(dataloader, num_batches=5):
    #     """
    #     データローダーから取得したバッチにPOSタグが含まれているか確認します。
    #     """
    #     print("Debugging POS Tags in Dataloader:")
    #     for batch_idx, batch in enumerate(dataloader):
    #         if batch_idx >= num_batches:
    #             break
    #         print(f"Batch {batch_idx}:")
    #         for sample_idx, sample in enumerate(batch):
    #             print(f"  Sample {sample_idx}, POS Tags: {sample[-1].shape if sample[-1] is not None else 'None'}")
    #     print("Dataloader check completed.")

    def __init__(self,inputs,most_epoch=-1, args = None):
        self.device = torch.device(f"cuda:{args['cuda']}")
        # self.device = torch.device("cpu")
        self.inputs = inputs
        self.most_epoch = most_epoch
        self.args = args
    def __len__(self):
        if self.most_epoch > -1:
            return min(self.most_epoch, len(self.inputs))
        return len(self.inputs)
    def __getitem__(self, idx):
        input_sample = self.inputs[idx]

        max_pos_len = self.args.get('max_pos_len', 75)
        pos_tags = input_sample.get('pos_tags')
        if pos_tags is not None:
            pos_tags = FMRI_dataset.tokenize_pos_tags(pos_tags, max_pos_len)

        # print(f"Sample {idx}:")
        # print(f"content_prev: {input_sample['content_prev'].shape}")
        # print(f"additional_bs: {input_sample['additional_bs'].shape}")
        # print(f"content_true: {input_sample['content_true'].shape}")
        # print(f"pos_tags: {input_sample['pos_tags'].shape if input_sample['pos_tags'] is not None else 'None'}")
        return (
                input_sample['content_prev'],
                input_sample['additional_bs'],
                input_sample['content_prev_sep'],
                input_sample['content_true'],
                input_sample['content_prev_mask'],
                input_sample['content_true_mask'],
                input_sample['content_all'],
                input_sample['content_all_mask'],
                input_sample['id'],
                input_sample.get('pos_tags')
            )

class FMRI_dataset():
    # def debug_pos_tags_in_dataset(self):
    #     """
    #     POSタグがデータセット内で正しく処理されているか確認します。
    #     """
    #     print("Debugging POS Tags in Dataset:")
    #     for idx, sample in enumerate(self.inputs[:5]):  # 最初の5サンプルをチェック
    #         print(f"Sample {idx}:")
    #         print(f"  content_prev: {sample['content_prev'].shape}")
    #         print(f"  additional_bs: {sample['additional_bs'].shape}")
    #         print(f"  content_true: {sample['content_true'].shape}")
    #         print(f"  pos_tags: {sample['pos_tags']}")  # POSタグのデバッグ
    #     print("Dataset check completed.")

    @staticmethod
    def process_pos_tags(file_path):
        """
        POSタグ情報をCSVから読み取り、タグをIDにマッピングし、トークナイズされた形式で返す。
        """
        pos_tags_list = []
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)  # ヘッダーを読み取る
            for row in reader:
                # POSタグ列を抽出
                pos_tags = row["POS_Tags"]

                # 各単語のPOSタグを抽出 (形態素/タグの形式を処理)
                pos_tags = [token.split('/')[-1] for token in pos_tags.split()]

                pos_tags_list.append(pos_tags)

        # POSタグ語彙辞書を作成
        unique_pos_tags = set(tag for tags in pos_tags_list for tag in tags)
        pos_tag_to_id = {tag: idx for idx, tag in enumerate(unique_pos_tags)}
        id_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_id.items()}

        # トークナイズされたPOSタグをIDに変換
        pos_tags_ids = [[pos_tag_to_id[tag] for tag in tags] for tags in pos_tags_list]

        return pos_tags_ids, pos_tag_to_id, id_to_pos_tag
    @staticmethod
    def tokenize_pos_tags(pos_tags, max_pos_len):
        """
        トークナイズされたPOSタグをエンコードし、指定された長さにパディングします。
        """
        if pos_tags is None:
            return torch.zeros(max_pos_len, dtype=torch.long)
        tokenized_pos = pos_tags[:max_pos_len]
        if len(tokenized_pos) < max_pos_len:
            tokenized_pos += [0] * (max_pos_len - len(tokenized_pos))
        return torch.tensor(tokenized_pos, dtype=torch.long)

    def pack_info(self, content_prev, additional_bs, content_true, trail_id, id, pos_tags=None, pos_tag_to_id=None):
        content_all = self.tokenizer.encode_plus(
            content_prev.strip() + ' ' + content_true,
            max_length=self.args['prev_mask_len'] + self.args['max_generate_len'],
            truncation=True,
            return_tensors='pt',
            add_special_tokens=self.add_special_tokens,
            padding='max_length'
        )
        content_true = self.tokenizer.encode_plus(
            content_true if self.args['model_name'] in ['llama-7b'] else ' ' + content_true,
            max_length=self.args['max_generate_len'],
            add_special_tokens=self.add_special_tokens,
            truncation=True,
            return_tensors='pt',
            padding='max_length'
        )
        content_prev = self.tokenizer.encode_plus(
            content_prev.strip(),
            max_length=self.args['prev_mask_len'],
            truncation=True,
            return_tensors='pt',
            add_special_tokens=self.add_special_tokens,
            padding='max_length'
        )


        # POSタグをトークナイズ・パディング
        max_pos_len = self.args.get('max_pos_len', 75)
        tokenized_pos_tags = FMRI_dataset.tokenize_pos_tags(pos_tags, max_pos_len)

        return {
            'content_prev': content_prev['input_ids'][0],
            'content_prev_mask': content_prev['attention_mask'][0],
            'additional_bs': torch.tensor(additional_bs, dtype=torch.float32),
            'content_prev_sep': self.tokenizer.encode_plus(['<brain/>', '</brain>'], return_tensors='pt')['input_ids'][0],
            'content_true': content_true['input_ids'][0],
            'content_true_mask': content_true['attention_mask'][0],
            'trail_id': trail_id,
            'content_all': content_all['input_ids'][0],
            'content_all_mask': content_all['attention_mask'][0],
            'id': id,
            'pos_tags': tokenized_pos_tags  # POS タグを追加
        }
    def __init__(self, input_dataset, args, tokenizer, decoding_model=None):
        self.decoding_model = decoding_model
        self.args = args
        self.add_special_tokens = False
        self.inputs = []
        self.shuffle_times = args['shuffle_times']
        dataset_path = args['dataset_path']
        self.tokenizer = tokenizer

        if args['normalized']:
            self.scaler = StandardScaler()

        id2info = {}
        tmp_id = 0

        # POSタグ情報の読み込み
        pos_csv_path = args['pos_csv_path']
        self.pos_tags_ids, self.pos_tag_to_id, self.id_to_pos_tag = FMRI_dataset.process_pos_tags(pos_csv_path)

        # POSデータを辞書として保持
        self.pos_data_dict = {
            str(idx): pos_tags for idx, pos_tags in enumerate(self.pos_tags_ids)
        }

        # POSタグを input_dataset に統合
        for story_id, story in enumerate(input_dataset.keys()):
            for item_id, item in enumerate(input_dataset[story]):
                for word_id, word in enumerate(item['word']):
                    # POS タグを統合（対応しない場合は 'UNK'）
                    if item_id < len(self.pos_tags_ids):
                        word['pos_tag'] = (
                            self.pos_tags_ids[item_id][word_id]
                            if word_id < len(self.pos_tags_ids[item_id])
                            else 'UNK'
                        )
                    else:
                        word['pos_tag'] = 'UNK'

        # 分岐処理
        if 'Pereira' in args['task_name']:
            dataset_name, subject_name = args['task_name'].split('_')
            pere_dataset = pickle.load(
                # open(f'{dataset_path}/{subject_name}.pca1000.wq.pkl.dic', 'rb')
                open(f'dataset/{dataset_name}/{subject_name}.pca1000.wq.pkl.dic', 'rb')
            ) if args['fmri_pca'] else pickle.load(
                # open(f'{dataset_path}/{subject_name}.wq.pkl.dic', 'rb')
                open(f'dataset/{dataset_name}/{subject_name}.pca1000.wq.pkl.dic', 'rb')
            )
            if args['normalized']:
                self.normalize_fmri(pere_dataset)
            for story in input_dataset.keys():
                for item_id, item in enumerate(input_dataset[story]):
                    for k in range(1, len(item['word'])):
                        content_prev = ' '.join([item['word'][j]['content'] for j in range(0, k)])
                        additional_bs = np.array([pere_dataset[story]['fmri'][idx] for idx in item['word'][k]['additional']])
                        content_true = item['word'][k]['content']
                        pos_tags = [item['word'][j]['pos_tag'] for j in range(0, k)]  # POS タグを収集
                        if args['add_end']:
                            content_true += '<|endoftext|>'
                        trail_id = random.random()
                        pos_tags = self.pos_data_dict.get(str(item_id), None)
                        packed_info = self.pack_info(
                            content_prev, additional_bs, content_true, trail_id, tmp_id, pos_tags, self.pos_tag_to_id
                        )
                        tmp_id += 1
                        if torch.sum(packed_info['content_true_mask']) > 0:
                            self.inputs.append(packed_info)

        elif 'Narratives' in args['task_name']:
            subject_name = args['task_name'].split('_')[1]
            for story in args['Narratives_stories']:
                narratives_dataset = pickle.load(
                    open(f'{dataset_path}/{story}.pca1000.wq.pkl.dic', 'rb')
                ) if args['fmri_pca'] else pickle.load(
                    open(f'{dataset_path}/{story}.wq.pkl.dic', 'rb')
                )
                for subject in [f'sub-{subject_name}']:
                    for item_id, item in enumerate(input_dataset[story][subject]):
                        for k in range(1, len(item['word'])):
                            content_prev = ' '.join([item['word'][j]['content'] for j in range(0, k)])
                            additional_bs = np.array([narratives_dataset[subject]['fmri'][idx] for idx in item['word'][k]['additional']])
                            content_true = item['word'][k]['content']
                            if args['add_end']:
                                content_true += '<|endoftext|>'
                            trail_id = random.random()
                            pos_tags = self.pos_tags_ids[item_id] if item_id < len(self.pos_tags_ids) else None
                            packed_info = self.pack_info(
                                content_prev, additional_bs, content_true, trail_id, tmp_id, pos_tags, self.pos_tag_to_id
                            )
                            tmp_id += 1
                            if torch.sum(packed_info['content_true_mask']) > 0:
                                self.inputs.append(packed_info)

        elif 'Huth' in args['task_name'] and args['mode'] == 'end2end':
            subject_name = args['task_name'].split('_')[1]
            huth_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.pca1000.wq.pkl.dic', 'rb'))
            for story in input_dataset.keys():
                for item_id, item in enumerate(input_dataset[story]):
                    content_prev = ' '.join([item['word'][j]['content'] for j in range(0, len(item['word']))])
                    additional_bs = np.array([huth_dataset[story]['fmri'][idx] for idx in item['word'][0]['additional']])
                    content_true = item['word'][0]['content']
                    if args['add_end']:
                        content_true += '<|endoftext|>'
                    trail_id = random.random()
                    pos_tags = self.pos_tags_ids[item_id] if item_id < len(self.pos_tags_ids) else None
                    packed_info = self.pack_info(
                        content_prev, additional_bs, content_true, trail_id, tmp_id, pos_tags, self.pos_tag_to_id
                    )
                    tmp_id += 1
                    if torch.sum(packed_info['content_true_mask']) > 0:
                        self.inputs.append(packed_info)

        elif 'Huth' in args['task_name']:
            subject_name = args['task_name'].split('_')[1]
            huth_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.pca1000.wq.pkl.dic', 'rb'))
            for story in input_dataset.keys():
                for item_id, item in enumerate(input_dataset[story]):
                    for k in range(1, len(item['word'])):
                        content_prev = ' '.join([item['word'][j]['content'] for j in range(0, k)])
                        additional_bs = np.array([huth_dataset[story]['fmri'][idx] for idx in item['word'][k]['additional']])
                        content_true = item['word'][k]['content']
                        if args['add_end']:
                            content_true += '<|endoftext|>'
                        trail_id = random.random()
                        pos_tags = self.pos_tags_ids[item_id] if item_id < len(self.pos_tags_ids) else None
                        packed_info = self.pack_info(
                            content_prev, additional_bs, content_true, trail_id, tmp_id, pos_tags, self.pos_tag_to_id
                        )
                        tmp_id += 1
                        if torch.sum(packed_info['content_true_mask']) > 0:
                            self.inputs.append(packed_info)
        # # データセット作成完了後の確認用出力
        # if len(self.inputs) > 0:
        #     print(f"First sample POS tags: {self.inputs[0]['pos_tags']}")

        self.pack_data_from_input(args)
        json.dump(id2info, open(self.args['checkpoint_path'] + '/' + 'id2info.json', 'w'))

        if args['use_bad_words_ids']:
            self.get_bad_word_ids()
            self.decoding_model.prompt_model.bad_words_ids = np.array(self.bad_word_ids).reshape(-1, 1).tolist()


    def get_bad_word_ids(self,):
        vocabulary = np.unique([item['content_true'] for item in self.test])
        print('length of vocabulary: ', len(vocabulary))
        self.bad_word_ids = np.setdiff1d(np.array(list(self.tokenizer.get_vocab().values())), vocabulary)

    def pack_data_from_input(self, args, ):
        self.train = []
        self.test = []
        self.valid = []
        self.is_shuffled = False
        test_ids = args['test_trail_ids']
        valid_ids = args['valid_trail_ids']
        for idx,item in enumerate(self.inputs):
            if item['trail_id'] > test_ids[0] and item['trail_id'] <= test_ids[1]:
                self.test.append(item)
            elif item['trail_id'] > valid_ids[0] and item['trail_id'] <= valid_ids[1]:
                self.valid.append(item)
            else:
                self.train.append(item)
        if args['input_method'] == 'permutated':
            tmp_additional_bs = copy.deepcopy([self.test[(idx+int(len(self.test)/2))%len(self.test)]['additional_bs'] for idx in range(len(self.test))])
            random.shuffle(tmp_additional_bs)
            for idx,item in enumerate(self.test):
                self.test[idx]['additional_bs'] = tmp_additional_bs[idx]
        if args['data_size'] != -1:
            random.shuffle(self.train)
            self.train = self.train[:args['data_size']]

        self.train_dataset = Splited_FMRI_dataset(self.train, args = args)
        self.valid_dataset = Splited_FMRI_dataset(self.valid, args = args) if len(self.valid) > 0 else Splited_FMRI_dataset(self.test, args = args)
        if len(self.test) == 0:
            raise ValueError("Test dataset is empty. Please check the data splitting configuration.")
        self.test_dataset = Splited_FMRI_dataset(self.test, args=args)
        # print(f"Number of training samples: {len(self.train)}")
        # print(f"Test dataset length: {len(self.test_dataset)}")
        # self.debug_pos_tags_in_dataset()


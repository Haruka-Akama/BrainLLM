import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import torch.optim as optim
import json
import wandb
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
import random
try:
    from settings_03 import model_name2path, model2hidden
    from prompt_with_pos_model_05 import Prompt_model
    # from prompt_with_pos_model import Prompt_model
    from optimizer_07 import Adam16
    from GPT_02 import GPT, GPT_Tokenizer
except:
    from language_generation.src.settings_03 import model_name2path, model2hidden
    from language_generation.src.prompt_with_pos_model_05 import Prompt_model
    # from prompt_with_pos_model import Prompt_model
    from language_generation.src.optimizer_07 import Adam16
    from language_generation.src.GPT_02 import GPT, GPT_Tokenizer
from post_hoc_evaluate_08 import save_evaluation_results

# def check_dataset(dataset, dataset_name):
#     """
#     データセット内の不正なデータをチェックします。
#     """
#     for index, item in enumerate(dataset):
#         if item is None or any(i is None for i in item):
#             print(f"Invalid data found in {dataset_name} at index {index}: {item}")
#             raise ValueError(f"{dataset_name} contains invalid data at index {index}.")

class Decoding_model:

    def put_data_into_cuda(self, content_prev,additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, ):
        content_prev, content_prev_sep, content_true, content_prev_mask, content_true_mask = content_prev.to(self.device), content_prev_sep.to(self.device), content_true.to(self.device), content_prev_mask.to(self.device), content_true_mask.to(self.device)
        if type(additional_bs) == list:
            for k in range(len(additional_bs)):
                additional_bs[k] = additional_bs[k].to(self.device)
        else:
            additional_bs = additional_bs.to(self.device)
        if self.args['model_name'] in ['llama-7b']:
            additional_bs_mask = torch.ones([additional_bs.shape[0], additional_bs.shape[1]+2+1]).to(self.device)
        else:
            additional_bs_mask = torch.ones([additional_bs.shape[0], additional_bs.shape[1]+2]).to(self.device)
        if self.args['model_name'] in ['llama-7b',]:
            additional_bs = additional_bs.half()
        return content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask

    @staticmethod
    def add_pos_tags_to_tokenizer(tokenizer, pos_tag_to_id):
        """
        トークナイザーにPOSタグを特殊トークンとして追加する。
        """
        special_tokens = [f"<POS_{tag}>" for tag in pos_tag_to_id.keys()]
        tokenizer.add_tokens(special_tokens)
        # print("特殊トークンを追加しました:", special_tokens)


    def __init__(self, args, pos_tag_to_id):
        # load model
        self.device = torch.device(f"cuda:{args['cuda']}")
        # self.device = torch.device("cpu")
        self.args = args
        if args['model_name'] in ['llama-7b',]:
            if args['model_name'] in model_name2path.keys():
                self.tokenizer = LlamaTokenizer.from_pretrained(model_name2path[args['model_name']])
                self.model = LlamaForCausalLM.from_pretrained(model_name2path[args['model_name']]).to(self.device)
            else:
                self.tokenizer = LlamaTokenizer.from_pretrained(args['model_name'])
                self.model = LlamaForCausalLM.from_pretrained(args['model_name']).to(self.device)
            self.model.half()
        elif 'huth' in args['model_name']:
            vocab = json.load(open(f"{model_name2path[args['model_name']]}/vocab.json"))
            path = f"{model_name2path[args['model_name']]}/model"
            self.GPT = GPT(vocab=vocab, path=path, device=self.device,)
            self.model = self.GPT.model
            self.tokenizer = GPT_Tokenizer(gpt=self.GPT)
        elif 'gpt' in args['model_name']:
            if args['model_name'] in model_name2path.keys():
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_name2path[args['model_name']])
                self.model = GPT2LMHeadModel.from_pretrained(model_name2path[args['model_name']]).to(self.device)
            else:
                self.tokenizer = GPT2Tokenizer.from_pretrained(args['model_name'])
                self.model = GPT2LMHeadModel.from_pretrained(args['model_name']).to(self.device)
        # add special token <brain/> and </brain>
        if self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.eos_token = self.tokenizer.mask_token
            self.tokenizer.pad_token = self.tokenizer.mask_token

        Decoding_model.add_pos_tags_to_tokenizer(self.tokenizer, pos_tag_to_id)

        if len(args['roi_selected']) > 0:
            self.new_tokens = []
            for k in range(len(args['roi_selected'])):
                self.new_tokens += ([f"<roi{k}/>", f"</roi{k}>"])
            self.tokenizer.add_tokens(self.new_tokens )
        self.new_tokens = ["<brain/>", "</brain>"]
        self.tokenizer.add_tokens(self.new_tokens)

        # トークナイザーにPOSタグを追加
        special_pos_tokens = [f"<POS_{tag}>" for tag in pos_tag_to_id.keys()]
        self.tokenizer.add_tokens(special_pos_tokens)
        # print(f"Added POS tags to tokenizer: {special_pos_tokens}")
        if args['model_name'] in ['llama-7b', 'vicuna-7b','llama-7b-old']:
            self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)
        else:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if args['enable_grad']==False:
            self.freeze_model()
        args['word_embed_size'] = model2hidden[args['model_name']]

        if args['enable_grad']==False:
            for new_token in self.new_tokens:
                new_token_id = self.tokenizer.convert_tokens_to_ids(f"{new_token}")
                if 'gpt2' in self.args['model_name']:
                    self.model.transformer.wte.weight[new_token_id].requires_grad = True
                elif 'llama' in self.args['model_name']:
                    self.model.model.embed_tokens.weight[new_token_id].requires_grad = True
                elif 'huth' in self.args['model_name']:
                    self.model.transformer.tokens_embed.weight[new_token_id].requires_grad = True

        self.prompt_model = Prompt_model(self, args, new_tokens, pos_vocab_size=50)
        self.max_norm = 0.1 if args['model_name'] in ['llama-7b','llama-7b-old'] else 10

        if args['load_check_point']:
            self.load_check_point()
        else:
            self.prompt_model.init_encoding_model()

    def freeze_model(self,):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_model_dict(self,):
        re = {'new_tokens':[]}
        for new_token in self.new_tokens:
            re['new_tokens'] = self.prompt_model.token_weights.detach()

        if self.args['enable_grad']:
            re['total_model'] = self.model.state_dict()

        if type(self.prompt_model.encoding_model) != list:
            re['encoding_model'] = self.prompt_model.encoding_model.state_dict()
        else:
            re['encoding_model'] = [item.state_dict() for item in self.prompt_model.encoding_model]
        return re

    # todo: it is difficult to calculate the uncertainty because the sequence has too many samples?
    # use the not rational setting and see if it works
    def get_entrophy(self,output, content_all_mask, content_all, content_true_mask, split=False):
        def entropy(p):
            # 过滤掉概率为0的事件，因为log(0)是未定义的
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        logits = output.logits[:, :-1, :] # b * seq_all-1 * logits
        content_all_mask = content_all_mask[:,1:]

        labels_mask = torch.zeros(content_all_mask.shape)
        content_true_mask_sum = torch.sum(content_true_mask, dim=1).int()
        content_all_mask_sum = torch.sum(content_all_mask, dim=1).int()
        for batch_id in range(labels_mask.shape[0]):
            labels_mask[batch_id][content_all_mask_sum[batch_id]-content_true_mask_sum[batch_id]:content_all_mask_sum[batch_id]] = 1
        labels_mask = labels_mask.to(self.device) # b * seq_true
        labels = content_all[:, :]
        if split:
            loss = []
            for batch_id in range(labels_mask.shape[0]):
                labels_tmp = labels[batch_id][content_true_mask[batch_id]==1]
                logits_tmp = logits[batch_id][labels_mask[batch_id]==1]
                loss.append(torch.nn.functional.cross_entropy(logits_tmp, labels_tmp, reduction='mean'))
        else:
            labels = labels[content_true_mask==1]
            logits = logits[labels_mask==1]
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        return loss

    def get_loss(self, output, content_all_mask, content_all, content_true_mask, split=False):
        print(f"Output type: {type(output)}")
        print(f"Output keys (if dict): {output.keys() if isinstance(output, dict) else 'N/A'}")
        logits = output.logits[:, :-1, :] # b * seq_all-1 * logits
        content_all_mask = content_all_mask[:,1:]

        labels_mask = torch.zeros(content_all_mask.shape)
        content_true_mask_sum = torch.sum(content_true_mask, dim=1).int()
        content_all_mask_sum = torch.sum(content_all_mask, dim=1).int()
        for batch_id in range(labels_mask.shape[0]):
            labels_mask[batch_id][content_all_mask_sum[batch_id]-content_true_mask_sum[batch_id]:content_all_mask_sum[batch_id]] = 1
        labels_mask = labels_mask.to(self.device) # b * seq_true
        labels = content_all[:, :]
        if split:
            loss = []
            for batch_id in range(labels_mask.shape[0]):
                labels_tmp = labels[batch_id][content_true_mask[batch_id]==1]
                logits_tmp = logits[batch_id][labels_mask[batch_id]==1]
                loss.append(torch.nn.functional.cross_entropy(logits_tmp, labels_tmp, reduction='mean'))
        else:
            labels = labels[content_true_mask==1]
            logits = logits[labels_mask==1]
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        # loss /= content_true.shape[1]
        return loss

    def load_check_point(self, path=None):
        if path is None:
            path = f'{self.args["llm_model_path"]}/model.pt'
        re = torch.load(path, map_location=torch.device("cpu"))
        if self.args['enable_grad']:
            self.model.load_state_dict(re['total_model'])
        self.prompt_model.token_weights.data = re['new_tokens'].detach().to(self.device)
        self.check_point = re
        self.prompt_model.check_point = re
        self.prompt_model.init_encoding_model()

    def get_distribute_loss(self, output, content_all_mask, content_all, content_true_mask, split=False, top_k = 100):
        logits = output.logits[:, :-1, :] # b * seq_all-1 * logits
        content_all_mask = content_all_mask[:,1:]

        labels_mask = torch.zeros(content_all_mask.shape)
        content_true_mask_sum = torch.sum(content_true_mask, dim=1).int()
        content_all_mask_sum = torch.sum(content_all_mask, dim=1).int()
        for batch_id in range(labels_mask.shape[0]):
            labels_mask[batch_id][content_all_mask_sum[batch_id]-content_true_mask_sum[batch_id]:content_all_mask_sum[batch_id]] = 1
        labels_mask = labels_mask.to(self.device) # b * seq_true
        labels = content_all[:, :]
        info = []
        if split:
            loss = []
            for batch_id in range(labels_mask.shape[0]):
                labels_tmp = labels[batch_id][content_true_mask[batch_id]==1]
                logits_tmp = logits[batch_id][labels_mask[batch_id]==1]
                values, indices = torch.topk(logits_tmp, dim=1, k = top_k)
                new_info = [indices.detach().cpu().numpy().tolist()]
                new_info.append((torch.argsort(logits_tmp,dim=1, descending=True) == labels_tmp.unsqueeze(1)).nonzero(as_tuple=True)[1].detach().cpu().numpy().tolist())
                info.append(new_info)
                loss.append(torch.nn.functional.cross_entropy(logits_tmp, labels_tmp, reduction='mean'))
        else:
            labels = labels[content_true_mask==1]
            logits = logits[labels_mask==1]
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        # loss /= content_true.shape[1]
        return loss, info

    def test_distribution(self, test_dataset, file_name=None):
        test_dataloader = DataLoader(test_dataset, batch_size = 4 if self.args['model_name'] in ['llama-7b'] and self.args['batch_size'] > 4 else self.args['batch_size'] , shuffle = False, num_workers =1)
        for batch in test_dataloader:
            print("content_true:", batch[3])  # content_true
            print("content_pred:", batch[2])  # content_pred
            break

        re = {'valid_loss':[], 'content_pred':[], 'content_true':[], 'content_prev':[],'content_pred_token_ids':[],'content_prev_tokens_length':[], 'info':[]}
        self.prompt_model.eval()
        if self.args['generation_method'] == 'greedy':
            file_name += '_' + self.args['generation_method']
        for content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, content_all, content_all_mask, data_id, pos_tags in tqdm.tqdm(test_dataloader, mininterval=300):
            content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask)
            content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)
            # self.debug_pos_tags_in_model_input(content_prev, additional_bs, content_true, pos_tags)


            output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags)

            loss_list, info = self.get_distribute_loss(output, content_all_mask, content_true, content_true_mask, split=True)
            for loss in loss_list:
                re['valid_loss'].append(loss.item())
            for item in info:
                re['info'].append(info)
            if len(re['valid_loss']) > 10 and self.args['mode'] in ['train','evaluate_test']:
                break

        if file_name is not None:
            json.dump(re, open(self.args['checkpoint_path']+'/'+file_name+'.json', 'w'))


    def valid(self, test_dataset):
        test_dataloader = DataLoader(test_dataset, batch_size = 4 if self.args['model_name'] in ['llama-7b'] and self.args['batch_size'] > 4 else self.args['batch_size'] , shuffle=False, num_workers=1)
        re = []
        self.prompt_model.eval()
        for content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, content_all, content_all_mask, data_id, pos_tags in tqdm.tqdm(test_dataloader, mininterval=300):
            content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask)
            content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)
            # self.debug_pos_tags_in_model_input(content_prev, additional_bs, content_true, pos_tags)

        if self.args['input_method'] == 'without_text':
            output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags)
        else:
            output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags)

            loss_list = self.get_loss(output, content_all_mask, content_true, content_true_mask, split=True)
            for loss in loss_list:
                re.append(loss.item())
        return re

    def test(self, test_dataset, file_name=None):
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=4 if self.args['model_name'] in ['llama-7b'] and self.args['batch_size'] > 4 else self.args['batch_size'],
            shuffle=False,
            num_workers=1
        )
        for batch in test_dataloader:
            content_prev, additional_bs, pos_tags = batch[0], batch[1], batch[-1]
            print("content_prev:", [self.tokenizer.decode(t) for t in content_prev])
            print("additional_bs shape:", additional_bs.shape)
            print("pos_tags:", pos_tags)
            break

        re = {
            'valid_loss': [],
            'content_pred': [],
            'content_true': [],
            'content_prev': [],
            'content_pred_token_ids': [],
            'content_prev_tokens_length': [],
            'data_id': []
        }
        self.prompt_model.eval()

        if file_name and self.args['generation_method'] == 'greedy':
            file_name += '_' + self.args['generation_method']

        for content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, content_all, content_all_mask, data_id, pos_tags in tqdm.tqdm(test_dataloader, mininterval=300):
            content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(
                content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask
            )
            content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)

            all_predicted_tokens = self.prompt_model.generate(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, mode='test')
            data_id = data_id.numpy().tolist()

            for i in range(content_all.shape[0]):
                re['content_true'].append(
                    self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(content_true[i]))
                    .replace('<|endoftext|>', '').replace('⁇', '').replace('</s>', '').replace('<unk>', '').strip()
                )
                predicted_tokens = all_predicted_tokens[i]
                try:
                    content_pred_tokens = self.tokenizer.convert_ids_to_tokens(predicted_tokens)
                except Exception:
                    content_pred_tokens = []
                    for item in predicted_tokens:
                        try:
                            content_pred_tokens.append(self.tokenizer.convert_ids_to_tokens([item])[0])
                        except Exception:
                            continue
                re['content_pred_token_ids'].append([item.detach().cpu().numpy().tolist() for item in predicted_tokens])
                re['content_pred'].append(self.tokenizer.convert_tokens_to_string(content_pred_tokens))
                re['content_prev'].append(
                    self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(content_prev[i]))
                    .replace('<|endoftext|>', '').replace('⁇', '').replace('</s>', '').replace('<unk>', '').strip()
                )
                re['data_id'].append(data_id[i])
                re['content_prev_tokens_length'].append(float(torch.sum(content_prev_mask[i]).detach().cpu().numpy()))

            if self.args['input_method'] == 'without_text':
                output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags)
            else:
                output, content_all_mask = self.prompt_model(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags, use_fake=False, mode='test')
            # デバッグコード
            print(f"Debugging Prompt_model output:")
            print(f"Output type: {type(output)}")
            if isinstance(output, str):
                print(f"Unexpected string output from prompt_model: {output}")
                raise ValueError(f"Invalid output type: {type(output)}")

            loss_list = self.get_loss(output, content_all_mask, content_true if self.args['loss'] != 'all' else content_all, content_true_mask if self.args['loss'] != 'all' else content_all_mask, split=True)
            re['valid_loss'].extend(loss.item() for loss in loss_list)

            if len(re['content_pred']) > 10 and self.args['mode'] in ['train', 'evaluate_test']:
                break

        # 保存処理: 元の出力
        if file_name:
            with open(self.args['checkpoint_path'] + '/' + file_name + '.txt', 'w') as f:
                for i in range(len(re['content_prev'])):
                    f.write(re['content_prev'][i] + '\n')
                    f.write('content_pred: ' + re['content_pred'][i] + '\n')
                    f.write('content_true: ' + re['content_true'][i] + '\n')
                    f.write('-----------------------------\n')

            json.dump(re, open(self.args['checkpoint_path'] + '/' + file_name + '.json', 'w'))

        # 評価結果の計算
        evaluation_results = {
            "BLEU-1": np.mean(re['corpus_bleu_score'][1]) if 'corpus_bleu_score' in re and 1 in re['corpus_bleu_score'] else 0,
            "BLEU-2": np.mean(re['corpus_bleu_score'][2]) if 'corpus_bleu_score' in re and 2 in re['corpus_bleu_score'] else 0,
            "ROUGE-1": np.mean(re['rouge_scores']['rouge-1']['r']) if 'rouge_scores' in re and 'rouge-1' in re['rouge_scores'] else 0,
            "ROUGE-L": np.mean(re['rouge_scores']['rouge-l']['r']) if 'rouge_scores' in re and 'rouge-l' in re['rouge_scores'] else 0,
            "WER": np.mean(re['wer']) if 'wer' in re else 0,
            "Validation Loss": np.mean(re['valid_loss'])
        }

        # 評価結果の保存
        save_evaluation_results(evaluation_results, self.args['checkpoint_path'], file_name or 'evaluation_results')

        # 結果を返す
        return re



    def pre_train(self, dataset, dataloader, optimizer, parameters, epoch=0):
        def __call__(self, content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags):
            print("Inside Prompt_model:")
            print(f"content_all shape: {content_all.shape}")
            print(f"content_all_mask shape: {content_all_mask.shape}")
            print(f"additional_bs shape: {additional_bs.shape if isinstance(additional_bs, torch.Tensor) else 'list'}")
            print(f"additional_bs_mask shape: {additional_bs_mask.shape}")
            # 本来の処理を実行
            output = self.model(...)
            print(f"Model output type: {type(output)}")
            return output

        total_additional_loss = 0

        # 1. Check for None in the batch data
        for index, batch in enumerate(tqdm.tqdm(dataloader, mininterval=300)):
            if any(item is None for item in batch):
                # print(f"None found in batch at index {index}: {batch}")
                raise ValueError("DataLoader batch contains None")

        # 2. Iterate over the dataloader
        for content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, content_all, content_all_mask, data_id, pos_tags in tqdm.tqdm(dataloader, mininterval=300):
            # self.debug_pos_tags_in_model_input(content_prev, additional_bs, content_true, pos_tags)

            # Move data to the appropriate device
            content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask, additional_bs_mask = self.put_data_into_cuda(
                content_prev, additional_bs, content_prev_sep, content_true, content_prev_mask, content_true_mask
            )

            # Compute the additional loss
            additional_loss = self.prompt_model.additional_loss(content_prev, content_prev_mask, additional_bs)
            total_additional_loss += additional_loss.item()

            # Zero the gradient, backpropagate, and update parameters
            optimizer.zero_grad()
            additional_loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, parameters), max_norm=self.max_norm)
            optimizer.step()

        # Return the average loss
        return total_additional_loss / len(dataset)


    def train(self, train_dataset, valid_dataset, test_dataset=None):
        output = None
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.args['batch_size'],
            shuffle=False,
            num_workers=1
        ) if test_dataset is not None else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args['batch_size'],
            shuffle=True,
            num_workers=1
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.args['batch_size'],
            shuffle=False,
            num_workers=1
        )

        # モデルとオプティマイザの初期化
        best_loss = float('inf')
        early_stop = self.args['early_stop']
        early_stop_epochs = 0
        parameters = list(self.prompt_model.parameters())

        if isinstance(self.prompt_model.encoding_model, list):
            for model in self.prompt_model.encoding_model:
                parameters += model.parameters()

        optimizer = (
            optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=self.args['lr'], weight_decay=self.args['l2'])
            if self.args['model_name'] not in ['llama-7b', 'llama-7b-old'] else
            Adam16(filter(lambda p: p.requires_grad, parameters), lr=self.args['lr'], weight_decay=self.args['l2'])
        )

        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args['weight_decay'])

        for epoch in range(self.args['num_epochs']):
            self.prompt_model.train()
            total_loss = 0
            total_additional_loss = 0 if self.args['additional_loss'] > 0 else None

            for batch in tqdm.tqdm(train_dataloader, mininterval=300):
                (
                    content_prev, additional_bs, content_prev_sep, content_true,
                    content_prev_mask, content_true_mask, content_all, content_all_mask,
                    data_id, pos_tags
                ) = batch

                # 必要なデータをCUDAに転送
                (
                    content_prev, additional_bs, content_prev_sep, content_true,
                    content_prev_mask, content_true_mask, additional_bs_mask
                ) = self.put_data_into_cuda(
                    content_prev, additional_bs, content_prev_sep,
                    content_true, content_prev_mask, content_true_mask
                )

                content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)

                # モデル出力の取得
                if self.args['input_method'] == 'without_text':
                    output, content_all_mask = self.prompt_model(
                        content_all, content_all_mask, additional_bs,
                        additional_bs_mask, content_prev_sep, pos_tags
                    )
                else:
                    output, content_all_mask = self.prompt_model(
                        content_all, content_all_mask, additional_bs,
                        additional_bs_mask, content_prev_sep, pos_tags
                    )

                # ロス計算
                if self.args['loss'] == 'all':
                    loss = self.get_loss(output, content_all_mask, content_all, content_all_mask)
                else:
                    loss = self.get_loss(output, content_all_mask, content_true, content_true_mask)

                if self.args['additional_loss'] > 0:
                    additional_loss = self.prompt_model.additional_loss(content_prev, content_prev_mask, additional_bs)
                    total_additional_loss += additional_loss.item()
                    loss = loss * (1 - self.args['additional_loss']) + additional_loss * self.args['additional_loss']

                total_loss += loss.item()

                # 勾配の更新
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.max_norm)
                optimizer.step()

            # 検証
            self.prompt_model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch in tqdm.tqdm(valid_dataloader, mininterval=300):
                    (
                        content_prev, additional_bs, content_prev_sep, content_true,
                        content_prev_mask, content_true_mask, content_all, content_all_mask,
                        data_id, pos_tags
                    ) = batch

                    (
                        content_prev, additional_bs, content_prev_sep, content_true,
                        content_prev_mask, content_true_mask, additional_bs_mask
                    ) = self.put_data_into_cuda(
                        content_prev, additional_bs, content_prev_sep,
                        content_true, content_prev_mask, content_true_mask
                    )

                    content_all, content_all_mask = content_all.to(self.device), content_all_mask.to(self.device)

                    if self.args['input_method'] == 'without_text':
                        output, content_all_mask = self.prompt_model(
                            content_all, content_all_mask, additional_bs,
                            additional_bs_mask, content_prev_sep, pos_tags
                        )
                    else:
                        output, content_all_mask = self.prompt_model(
                            content_all, content_all_mask, additional_bs,
                            additional_bs_mask, content_prev_sep, pos_tags
                        )

                    if self.args['loss'] == 'all':
                        loss = self.get_loss(output, content_all_mask, content_all, content_all_mask)
                    else:
                        loss = self.get_loss(output, content_all_mask, content_true, content_true_mask)

                    valid_loss += loss.item()

            valid_loss /= len(valid_dataloader)
            total_loss /= len(train_dataloader)

            # 早期終了
            if valid_loss < best_loss:
                best_loss = valid_loss
                early_stop_epochs = 0
                best_model_wts = self.get_model_dict()
                torch.save(best_model_wts, self.args['checkpoint_path'] + '/model.pt')
            else:
                early_stop_epochs += 1
                if early_stop_epochs >= early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # ログ出力
            output_str = f"Epoch {epoch}: Train Loss = {total_loss:.3f}, Validation Loss = {valid_loss:.3f}"
            if self.args['additional_loss'] > 0:
                output_str += f", Additional Loss = {total_additional_loss:.3f}"

            print(output_str)
            with open(self.args['checkpoint_path'] + '/log.txt', 'a') as log_file:
                log_file.write(output_str + '\n')

            if self.args['wandb'] != 'none':
                wandb.log({
                    "Train Loss": total_loss,
                    "Validation Loss": valid_loss,
                    "Additional Loss": total_additional_loss if self.args['additional_loss'] > 0 else None
                })

            scheduler.step()



if __name__ == '__main__':
    args = {'model_name':'gpt2','brain_embed_size':1000, 'word_embed_size':768,'cuda':0}
    pos_tag_to_id = {...}  # 必要に応じて初期化
    decoding_model = Decoding_model(args, pos_tag_to_id)




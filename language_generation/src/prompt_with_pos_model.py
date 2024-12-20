import torch
import torch.nn as nn
import random
import csv
try:
    from top_model_utils import generate_beam
    from sub_models import Encoding_model
except Exception as e:
    # print(e)
    from src.top_model_utils import generate_beam
    from src.sub_models import Encoding_model

def read_pos_tags(file_path):
	pos_tags_list = []
	with open(file_path, 'r') as file:
		reader = csv.reader(file)
		next(reader)  # ヘッダーをスキップ
		for row in reader:
			pos_tags = row[2]  # POSタグがCSVの3列目にあると仮定
			pos_tags = eval(pos_tags.replace("(", "[").replace(")", "]"))  # タグをリスト形式に変換
			pos_tags_list.append(pos_tags)
	return pos_tags_list

# CSVファイルのパスを指定
file_path = "dataset_pos/Pereira_sentence_POS.csv"
pos_tags_list = read_pos_tags(file_path)

# POSタグ語彙辞書を作成
unique_pos_tags = set(tag for tags in pos_tags_list for tag in tags)
pos_tag_to_id = {tag: idx for idx, tag in enumerate(unique_pos_tags)}
id_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_id.items()}

# POSタグをIDに変換
def pos_tags_to_ids(pos_tags_list, pos_tag_to_id):
	return [[pos_tag_to_id[tag] for tag in tags] for tags in pos_tags_list]

pos_tags_ids = pos_tags_to_ids(pos_tags_list, pos_tag_to_id)

def pad_pos_tags(pos_tags_ids, max_len, padding_value=0):
    padded_pos_tags = []
    for tags in pos_tags_ids:
        if len(tags) < max_len:
            padded_pos_tags.append(tags + [padding_value] * (max_len - len(tags)))
        else:
            padded_pos_tags.append(tags[:max_len])
    return torch.tensor(padded_pos_tags)

#max_tag数
max_len = 75
padded_pos_tags = pad_pos_tags(pos_tags_ids, max_len)

class Prompt_model(nn.Module):
    def __init__(self, args, model, tokenizer, device,new_tokens,):
        super(Prompt_model, self).__init__()
        self.model = model
        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        self.mse_loss = nn.MSELoss()
        tmp_weights = []
        for new_token in new_tokens:
            new_token_id = self.tokenizer.convert_tokens_to_ids(f"{new_token}")
            if 'gpt2' in self.args['model_name']:
                tmp_weight = self.model.transformer.wte.weight[new_token_id]
            elif 'llama' in self.args['model_name']:
                tmp_weight = self.model.model.embed_tokens.weight[new_token_id]
            elif 'huth' in self.args['model_name']:
                tmp_weight = self.model.transformer.tokens_embed.weight[new_token_id]
            tmp_weights.append(tmp_weight)
        tmp_weights = torch.stack(tmp_weights,)
        self.token_weights = nn.Parameter(tmp_weights.clone().detach(), requires_grad=True)

    def init_encoding_model(self,):
        self.encoding_model = Encoding_model(self.args, device = self.device)
        self.encoding_model.to(self.device)
        if self.args['model_name'] in ['llama-7b',]:
            self.encoding_model.half()
        if self.args['load_check_point']:
            if type(self.encoding_model) == list:
                for i in range(len(self.encoding_model)):
                    self.encoding_model[i].load_state_dict(self.check_point['encoding_model'][i])
                    self.encoding_model[i].to(self.device)
            else:
                self.encoding_model.load_state_dict(self.check_point['encoding_model'])
                self.encoding_model.to(self.device)

    def words2embedding(self, input_ids):
        if self.args['model_name'] in ['llama-7b', 'huth']:
            return self.model.get_input_embeddings()(input_ids)
        else:
            if type(input_ids) == list:
                re = []
                for item in input_ids:
                    re.append(self.model.transformer.wte(item))
                return re
            else:
                return self.model.transformer.wte(input_ids)

    def get_prev(self, additional_bs, content_prev_sep):
        if type(additional_bs) == list:
            re = []
            for k in range(len(additional_bs)):
                k_roi_toknizer = self.tokenizer.encode_plus([f'<roi{k}/>', f'<roi{k}/>'],return_tensors='pt')['input_ids'].to(self.device)
                k_roi_toknizer = self.words2embedding(k_roi_toknizer)
                re += [k_roi_toknizer[:,:1,:], additional_bs[k], k_roi_toknizer[:,1:,:],]
            return re
            #
        else:
            if self.args['model_name'] in ['llama-7b',]:
                return [content_prev_sep[:,:1,:], content_prev_sep[:,1:2,:], additional_bs, content_prev_sep[:,2:,:],]
            else:
                return [content_prev_sep[:,:1,:], additional_bs, content_prev_sep[:,1:,:],]

    def get_tokens(self, content_prev_sep):
        # batchsize * seqlength * shape
        content_prev_sep = self.words2embedding(content_prev_sep)
        content_prev_sep[:,-1] = self.token_weights[-1]
        content_prev_sep[:,-2] = self.token_weights[-2]
        return content_prev_sep

    def tokenize(self, content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags, use_fake=True, mode='train'):
        content_all = self.words2embedding(content_all)  # テキスト情報のエンベディング
        content_prev_sep = self.get_tokens(content_prev_sep)  # セパレーターのエンベディング

        if use_fake == False:
            # POSタグを統合した脳情報エンベディング
            if isinstance(additional_bs, list):  # 複数の脳情報がリストで渡される場合
                additional_bs_tokenized = []
                for i, bs in enumerate(additional_bs):
                    additional_bs_tokenized.append(self.encoding_model(bs, pos_tags[:, i, :]))
                    print ('additional_bsはリスト形式')
            else:  # 単一の脳情報の場合
                additional_bs_tokenized = self.encoding_model(additional_bs, pos_tags)
                print ('additional_bsはテンソル形式')
        else:
            # フェイクデータの場合はそのままテキストを使用
            additional_bs_tokenized = self.words2embedding(content_all)

        if self.args['input_method'] == 'without_brain':
            content_all_list = [content_all]
            content_all_mask = content_all_mask
        else:
            content_all_list = self.get_prev(additional_bs_tokenized, content_prev_sep) + [content_all]
            content_all_mask = torch.cat([additional_bs_mask, content_all_mask], dim=-1)

        content_all = torch.cat(content_all_list, dim=-2)
        return content_all, content_all_mask


    def forward(self, content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=True,mode='train'):
        content_all, content_all_mask = self.tokenize(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake,mode)
        output = self.model(inputs_embeds=content_all, attention_mask = content_all_mask)
        return output, content_all_mask

    def pad2left(self, content_prev, content_prev_mask):
        padding_counts = (content_prev_mask == 1).sum(dim=1)
        # initialize new tensors for fill
        front_padded_input_embeds = torch.zeros_like(content_prev)
        front_padded_mask = torch.zeros_like(content_prev_mask)

        for i in range(content_prev.size(0)):  # go through each sample
            # calculate the number of positions we need to move
            shift = padding_counts[i].item()
            # fill the input_embeds and the mask
            front_padded_input_embeds[i, content_prev.size(1) - shift:] = content_prev[i, :shift]
            front_padded_input_embeds[i, :content_prev.size(1) - shift] = content_prev[i, shift:]
            front_padded_mask[i, content_prev.size(1) - shift:] = content_prev_mask[i, :shift]
        return front_padded_input_embeds, front_padded_mask

    def get_perplexity(self, content_prev, content_prev_mask,  additional_bs, additional_bs_mask, content_prev_sep, candidate, ):
        content_prev, content_prev_mask = self.tokenize(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False, mode='test')
        content_prev, content_prev_mask = self.pad2left(content_prev, content_prev_mask)
        total_prob = 1
        for target_id in candidate[0]:
            # predict and get the logits
            with torch.no_grad():
                outputs = self.model(inputs_embeds = content_prev, )
            logits = outputs.logits.squeeze(0)[-1]
            # transform logits into probability distribution
            probs = torch.softmax(logits, dim=0)
            # Get the probability that the output [MASK] is predicted as a special token
            prob = probs[target_id].item() / probs.sum().item()
            total_prob *= prob
            content_prev = torch.cat([content_prev, self.words2embedding(torch.tensor([[target_id]]).to(self.device))], dim=1)
        return total_prob

    def generate(self, content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, mode='test'):
        content_prev, content_prev_mask = self.tokenize(content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, use_fake=False, mode='test')
        content_prev, content_prev_mask = self.pad2left(content_prev, content_prev_mask)

        if self.args['generation_method'] == 'greedy':
            seq2seqLMoutput = self.model.generate(inputs_embeds = content_prev, attention_mask = content_prev_mask, min_new_tokens = 4, max_new_tokens=32,return_dict_in_generate=True,num_beams=1,do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        elif self.args['generation_method'] == 'beam':
            if self.args['model_name'] == 'huth':
                # batch_size should be 1 in huth generation
                seq2seqLMoutput = {'sequences':[]}
                for i in range(content_prev.shape[0]):
                    seq2seqLMoutput['sequences'].append(generate_beam(self.model, self.tokenizer, beam_size = 5, embed= content_prev[i].unsqueeze(0),))
            else:
                if self.args['use_bad_words_ids']:
                    seq2seqLMoutput = self.model.generate(inputs_embeds = content_prev, attention_mask = content_prev_mask, min_new_tokens = 4, max_new_tokens=32,return_dict_in_generate=True,num_beams=5,do_sample=False, repetition_penalty=self.args['repetition_penalty'], pad_token_id=self.tokenizer.eos_token_id, bad_words_ids=self.bad_words_ids, )
                else:
                    seq2seqLMoutput = self.model.generate(inputs_embeds = content_prev, attention_mask = content_prev_mask, min_new_tokens = 4, max_new_tokens=32,return_dict_in_generate=True,num_beams=5,do_sample=False, repetition_penalty=self.args['repetition_penalty'], pad_token_id=self.tokenizer.eos_token_id, )

        all_truncated_predictions = []
        for i in range(len(seq2seqLMoutput['sequences'])):
            predictions = seq2seqLMoutput['sequences'][i]
            truncated_prediction = [] if predictions[0] == self.tokenizer.eos_token_id else [predictions[0]]
            for t in predictions[1:]:
                if t != self.tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            all_truncated_predictions.append(truncated_prediction)
        return all_truncated_predictions

    def fuse(self, data, content_true_mask, fuse_len=4):
        # data is b * n * m, fuse to b * 4 * m, use a mean fusing for simplicity
        return torch.mean(data[:,:content_true_mask.shape[1],:], axis=1).unsqueeze(1).tile(1, fuse_len, 1)

    def additional_loss(self, content_true, content_true_mask, additional_bs):
        fuse_len = additional_bs.shape[1]
        content_true = self.words2embedding(content_true)
        mean_content_true = self.fuse(content_true, content_true_mask, fuse_len)
        additional_bs = self.encoding_model(additional_bs)
        return self.mse_loss(additional_bs, mean_content_true)

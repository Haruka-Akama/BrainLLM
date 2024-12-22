import torch
import torch.nn as nn
import random
import csv
try:
    from top_model_utils_06 import generate_beam
    from sub_models_04 import Encoding_model
except Exception as e:
    # print(e)
    from src.top_model_utils_06 import generate_beam
    from src.sub_models_04 import Encoding_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM
import csv

def process_pos_tags(file_path):
    """
    POSタグ情報をCSVから読み取り、タグをIDにマッピングし、リスト形式で返す。
    """
    pos_tags_list = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # ヘッダーをスキップ
        for row in reader:
            # POSタグをパース
            pos_tags = row[2]  # POSタグがCSVの3列目にあると仮定
            pos_tags = [tag.rsplit('/', 1)[-1] for tag in pos_tags.split()]  # POSタグのみ抽出
            pos_tags_list.append(pos_tags)

    # POSタグ語彙辞書を作成
    unique_pos_tags = set(tag for tags in pos_tags_list for tag in tags)
    pos_tag_to_id = {tag: idx for idx, tag in enumerate(unique_pos_tags)}
    id_to_pos_tag = {idx: tag for tag, idx in pos_tag_to_id.items()}

    # POSタグをIDに変換
    pos_tags_ids = [[pos_tag_to_id[tag] for tag in tags] for tags in pos_tags_list]

    return pos_tags_ids, pos_tag_to_id, id_to_pos_tag

# def process_pos_tags(pos_tag_string):
#     # "word/POS" を ["word", "<POS>"] 形式に変換
#     tokens_with_pos = []
#     for word_pos in pos_tag_string.split():
#         word, pos = word_pos.rsplit('/', 1)
#         tokens_with_pos.extend([word, f"<POS_{pos}>"])
#     return tokens_with_pos

class Prompt_model(nn.Module):
    def __init__(self, args, new_tokens, pos_vocab_size=50):
        super(Prompt_model, self).__init__()

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.mse_loss = nn.MSELoss()
        self.pos_vocab_size = pos_vocab_size
        self.embedding_dim = self.model.config.hidden_size

        if args['model_name'] == 'gpt2':
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.embedding_layer = self.model.transformer.wte
        elif args['model_name'] == 'llama':
            self.model = LlamaForCausalLM.from_pretrained("llama")
            self.embedding_layer = self.model.model.embed_tokens
        else:
            raise ValueError(f"Unsupported model_name: {args['model_name']}")
        self.model.to(self.device)

        self._resize_token_embeddings(new_tokens)

        self.pos_embedding = nn.Embedding(pos_vocab_size, self.embedding_layer.embedding_dim).to(self.device)
        tmp_weights = []
        for new_token in new_tokens:
            new_token_id = self.tokenizer.convert_tokens_to_ids(f"{new_token}")
            if 'gpt2' in self.args['model_name']:
                tmp_weight = self.model.transformer.wte.weight[new_token_id]
            elif 'llama' in self.args['model_name']:
                tmp_weight = self.model.model.embed_tokens.weight[new_token_id]
            else:
                raise ValueError(f"Unsupported model_name: {self.args['model_name']}")
            tmp_weights.append(tmp_weight)

    def _resize_token_embeddings(self, new_tokens):
        if new_tokens:
            num_added_tokens = self.tokenizer.add_tokens(new_tokens)
            if num_added_tokens > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"Added {num_added_tokens} new tokens.")

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
        # 文埋め込み
        content_all = self.words2embedding(content_all)
        content_prev_sep = self.get_tokens(content_prev_sep)

        # 脳活動情報を処理
        if random.random() > self.args['fake_input'] or not use_fake:
            additional_bs_tokenized = self.encoding_model(additional_bs) if not isinstance(additional_bs, list) else [
                self.encoding_model[k](additional_bs[k]) for k in range(len(additional_bs))
            ]
        else:
            additional_bs_tokenized = self.words2embedding(content_all) if not isinstance(additional_bs, list) else [
                self.words2embedding(content_all) for _ in range(len(additional_bs))
            ]

        # POS タグの埋め込み
        if pos_tags is not None:
            pos_tags = torch.tensor(pos_tags, dtype=torch.long).to(self.device) if not isinstance(pos_tags, torch.Tensor) else pos_tags.to(self.device)
        else:
            max_pos_len = self.args.get('max_pos_len', 75)
            pos_tags = torch.zeros(max_pos_len, dtype=torch.long).to(self.device)

        pos_embeddings = self.pos_embedding_layer(pos_tags)
        pos_embeddings = torch.nn.functional.pad(pos_embeddings, (0, content_all.shape[-1] - pos_embeddings.shape[-1]), mode='constant', value=0)

        # 各埋め込みを結合
        content_all_list = self.get_prev(additional_bs_tokenized, content_prev_sep) + [content_all, pos_embeddings]
        if content_all.size(1) != content_all_mask.size(1):
            print(f"Mismatch detected: content_all size {content_all.size()}, content_all_mask size {content_all_mask.size()}")
            content_all_mask = torch.cat([
                content_all_mask,
                torch.ones(content_all.size(1) - content_all_mask.size(1), device=content_all.device)
            ], dim=-1)

        content_all = torch.cat(content_all_list, dim=-2)
        print(f"content_all.shape: {content_all.shape}")
        print(f"content_all_mask.shape: {content_all_mask.shape}")

        return content_all, content_all_mask

    def forward(self, content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags, use_fake=True, mode='train'):
        # 文埋め込み、脳活動情報、POSタグの統合
        content_all, content_all_mask = self.tokenize(content_all, content_all_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags, use_fake, mode)

        # content_all_mask の再生成
        batch_size, seq_len, _ = content_all.shape
        print(f"Regenerating content_all_mask with shape: ({batch_size}, {seq_len})")
        content_all_mask = torch.ones((batch_size, seq_len), dtype=torch.float32, device=content_all.device)

        # モデルへの入力処理
        output = self.model(inputs_embeds=content_all, attention_mask=content_all_mask)
        print(f"Output type: {type(output)}")
        return output


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

    def generate(self, content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags=None, use_fake=False, mode='test'):
        content_prev, content_prev_mask = self.tokenize(
            content_prev, content_prev_mask, additional_bs, additional_bs_mask, content_prev_sep, pos_tags, use_fake=use_fake, mode=mode)

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

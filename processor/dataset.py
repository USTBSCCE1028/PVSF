import csv
import unicodedata
import os
import re
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import json
import ast
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizerFast
import logging
from utils import EXTERNAL_TOKENS

logger = logging.getLogger(__name__)

class EEProcessor(object):
    def __init__(self, data_path, bert_name):
        self.data_path = data_path
        self.re_path = data_path['re_path']
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<s>', '</s>', '<o>', '</o>']})

    def load_MEE_file(self, mode="train", sample_ratio=1.0):
        load_file = self.data_path[mode]
        words, relations, imgids, dataid = [], [], [], []
        aux_imgs = {}
        id = 0
        with open(load_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for dic in data:
                entities = []
                words.append(dic['words'])
                relations.append(dic['golden-event-mentions'][0]['event_type'])
                dataid.append(id)
                image = dic['image'][0]
                imgids.append(image)
                id += 1
                if len(words) == len(relations) == len(imgids) == len(dataid):
                    aux_path = self.data_path['auximgs']
                    with open(aux_path, 'r') as f:
                        for line in f:
                            key, value = line.strip().split(':')
                            value = ast.literal_eval(value.strip())
                            aux_imgs[key] = value

        return {'words': words, 'relations': relations, 'imgids': imgids, 'dataid': dataid, 'aux_imgs': aux_imgs}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

class NERProcessor(object):
    def __init__(self, data_path, bert_name) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)

    def load_MEE_file(self, mode='train', sample_ratio=1.0):
        load_file = self.data_path[mode]
        sentences, words, imgids, NER_targets,  relations = [], [], [],  [], []
        aux_imgs, entity = {}, {}
        with open(load_file, encoding='utf-8') as f:
            data = json.load(f)
            for dic in data:
                entities = []
                relations.append(dic['golden-event-mentions'][0]['event_type'])
                sentences.append(dic['sentence'])
                words.append(dic['words'])
                image = dic['image'][0]
                imgids.append(image)
                NER_targets.append(dic['entity_labels'])
                # ROLE_targets.append(dic['labels'])
                for entity_l in dic['golden-entity-mentions']:
                    entities.append(entity_l['text'])
                    entity[dic['sentence']] = entities
                if len(words) == len(imgids) == len(NER_targets) == len(sentences):
                    aux_path = self.data_path['auximgs']
                    with open(aux_path, 'r') as f:
                        for line in f:
                            key, value = line.strip().split(':')
                            value = ast.literal_eval(value.strip())
                            aux_imgs[key] = value
        return {"words": words, "NER_targets": NER_targets, "imgids": imgids, "aux_imgs": aux_imgs, 'relations': relations}

    def get_label_mapping(self):
        '''
        ROLE_LABEL= ["O", 'B-Agent', 'I-Agent', 'B-Artifact', 'I-Artifact',
           'B-Entity', 'I-Entity', 'B-Instrument', 'I-Instrument',
           'B-Person', 'I-Person', 'B-Recipient', 'I-Recipient',
           'B-Police', 'I-Police', 'B-Attacker', 'I-Attacker',
           'B-Target', 'I-Target', 'B-Vehicle', 'I-Vehicle',
           'B-Victim', 'I-Victim', 'B-Destination', 'I-Destination',
           'B-Giver', 'I-Giver', 'B-Origin', 'I-Origin',
           'B-Place', 'I-Place', "X", "[CLS]", "[SEP]"]
        ROLE_label_mapping = {label:idx for idx, label in enumerate(ROLE_LABEL, 1)}
        ROLE_label_mapping["PAD"] = 0
        '''
        NER_LABEL=['O', 'B-FAC', 'I-FAC', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG',
                    'B-PER', 'I-PER', 'B-VALUE', 'I-VALUE', 'B-VEH', 'I-VEH', 'B-WEA', 'I-WEA',
                    'X', '[CLS]', '[SEP]']
        NER_label_mapping = {label: idx for idx, label in enumerate(NER_LABEL, 1)}
        NER_label_mapping["PAD"] = 0

        return NER_label_mapping

class Events:
    def __init__(self, doc_id, context, event_type_2_events):
        self.doc_id = doc_id
        self.context = context
        self.event_type_2_events = event_type_2_events
class MEProcessor(object):
    def __init__(self, data_path, bert_name, max_seq) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.max_seq_len = max_seq


    def load_MEE_file(self, mode='train', sample_ratio=1.0):
        load_file = self.data_path[mode]
        prompt_file = self.data_path['prompts']
        df = pd.read_csv(prompt_file, header=None)
        df.columns = ['EventType', 'Prompt']
        prompts_dict = df.set_index('EventType')['Prompt'].to_dict()
        sentences, words, imgids, NER_targets, ROLE_targets, relations = [], [], [],  [], [], []
        aux_imgs, entity = {}, {}
        examples = []
        with open(load_file, encoding='utf-8') as f:
            data = json.load(f)
            for dic in data:
                entities, trigger, temp_dict, temp_args, args, example = [], [], [], [], [], []
                temp_dict, events_arg_trigger = {}, {}
                information = dic['golden-event-mentions'][0]
                relations.append(dic['golden-event-mentions'][0]['event_type'])
                ########
                trigger.append(information['trigger']['start'])
                trigger.append(information['trigger']['end'])
                trigger.append(information['trigger']['text'])
                for arg in information['arguments']:
                    temp_args.append(arg['start'])
                    temp_args.append(arg['end'])
                    temp_args.append(arg['text'])
                    temp_args.append(arg['role'])
                args.append(temp_args)
                temp_dict['event_type'] = information['event_type']
                temp_dict['trigger'] = trigger
                temp_dict['args'] = args
                example.append(temp_dict)
                events_arg_trigger[information['event_type']] = example
                event = Events(doc_id=dic['sentence_id'], context=dic['words'], event_type_2_events=events_arg_trigger)
                examples.append(event)
                #########
                sentences.append(dic['sentence'])
                words.append(dic['words'])
                image = dic['image'][0]
                imgids.append(image)
                NER_targets.append(dic['entity_predict'])
                ROLE_targets.append(dic['labels'])
                for entity_l in dic['golden-entity-mentions']:
                    entities.append(entity_l['text'])
                    entity[dic['sentence']] = entities
                if len(words) == len(imgids) == len(NER_targets) == len(sentences):
                    aux_path = self.data_path['auximgs']
                    with open(aux_path, 'r') as f:
                        for line in f:
                            key, value = line.strip().split(':')
                            value = ast.literal_eval(value.strip())
                            aux_imgs[key] = value

        filtered_sentences, filtered_words, filtered_imgids, filtered_NER_targets, filtered_ROLE_targets, filtered_relations, filtered_examples = [], [], [], [], [], [], []


        n = 0
        for i in tqdm(range(len(sentences))):
            sentence = sentences[i]
            word_list = words[i]
            example = examples[i]
            context = example.context
            event_type_2_events = example.event_type_2_events
            triggers = [tuple(e['trigger']) for events in event_type_2_events.values() for e in events]
            offset = 0
            marked_context = deepcopy(context)
            marker_indice = list(range(len(triggers)))
            for j, t in enumerate(triggers):
                t_start = t[0]
                t_end = t[1]
                marked_context = marked_context[:(t_start + offset)] + ['<t-%d>' % marker_indice[j]] + context[t_start: t_end] + ['</t-%d>' % marker_indice[j]] + context[t_end:]
                offset += 2
            enc_text = " ".join(marked_context)

            enc = self.tokenizer(enc_text, add_special_tokens=True)
            enc_input_ids = enc["input_ids"]
            if len(enc_input_ids) <= self.max_seq_len:
                filtered_sentences.append(sentence)
                filtered_words.append(word_list)
                filtered_imgids.append(imgids[i])
                filtered_NER_targets.append(NER_targets[i])
                filtered_ROLE_targets.append(ROLE_targets[i])
                filtered_relations.append(relations[i])
                filtered_examples.append(example)
            else:
                n += 1

        print(f"{n} samples were dropped")

        return {'sentence': filtered_sentences, "words": filtered_words, "NER_targets": filtered_NER_targets,
                'ROLE_targets': filtered_ROLE_targets, "imgs": filtered_imgids, "aux_imgs": aux_imgs,
                'relations': filtered_relations, 'examples': filtered_examples, 'prompts': prompts_dict}


    def read_roles(self, role_path):
        template_dict = {}
        role_dict = {}

        with open(role_path, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                event_type_arg, template = line
                template_dict[event_type_arg] = template

                event_type, arg = event_type_arg.split('_')
                if event_type not in role_dict:
                    role_dict[event_type] = []
                role_dict[event_type].append(arg)

        return template_dict, role_dict



    def get_label_mapping(self):
        ROLE_LABEL= ["O", 'B-Agent', 'I-Agent', 'B-Artifact', 'I-Artifact',
           'B-Entity', 'I-Entity', 'B-Instrument', 'I-Instrument',
           'B-Person', 'I-Person', 'B-Recipient', 'I-Recipient',
           'B-Police', 'I-Police', 'B-Attacker', 'I-Attacker',
           'B-Target', 'I-Target', 'B-Vehicle', 'I-Vehicle',
           'B-Victim', 'I-Victim', 'B-Destination', 'I-Destination',
           'B-Giver', 'I-Giver', 'B-Origin', 'I-Origin',
           'B-Place', 'I-Place', "X", "[CLS]", "[SEP]"]
        ROLE_label_mapping = {label:idx for idx, label in enumerate(ROLE_LABEL, 1)}
        ROLE_label_mapping["PAD"] = 0

        return ROLE_label_mapping

class EEDataset(Dataset):
    def __init__(self, args, caption_data, processor, transform, img_path=None, aux_img_path=None, max_seq=40,
                 sample_ratio=1.0, mode="train") -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = os.path.join('data/preprocessed_images') if img_path is not None else img_path
        self.aux_img_path = os.path.join('data/preprocessed_images',
                                         'aux_images') if aux_img_path is not None else aux_img_path
        self.mode = mode
        self.args = args
        self.caption_data = caption_data
        self.data_dict = self.processor.load_MEE_file(mode, sample_ratio)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
        self.data_dict['imgids'][idx]
        item_id = self.data_dict['dataid'][idx]
        encode_dict = self.tokenizer.encode_plus(text=word_list, max_length=self.max_seq, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
        encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)
        re_label = self.re_dict[relation]

        # image process
        img_path = os.path.join(self.img_path, imgid + '.pt')
        image = torch.load(img_path)
        sequence = self.caption_data['data/images/' + imgid]['seq']
        seq_dict = self.tokenizer.encode_plus(text=sequence, max_length=16, truncation=True, padding='max_length')
        seq_input_ids, seq_token_type_ids, seq_attention_mask = seq_dict['input_ids'], seq_dict['token_type_ids'], \
        seq_dict['attention_mask']
        seq_input_ids, seq_token_type_ids, seq_attention_mask = torch.tensor(seq_input_ids), torch.tensor(
            seq_token_type_ids), torch.tensor(seq_attention_mask)

        if self.aux_img_path is not None:
            aux_imgs = []
            aux_img_paths = self.data_dict['aux_imgs'][imgid]
            for path in aux_img_paths:
                aux_img_path = os.path.join(self.aux_img_path, path + '.pt')
                aux_img = torch.load(aux_img_path)
                aux_imgs.append(aux_img)
            for i in range(3 - len(aux_imgs)):
                aux_imgs.append(torch.zeros((3, 224, 224)))
            aux_imgs = torch.stack(aux_imgs, dim=0)
            assert len(aux_imgs) == 3
            return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs, \
                ' '.join(self.data_dict['words'][idx]), self.data_dict['imgids'][idx], relation, \
                seq_input_ids, seq_token_type_ids, seq_attention_mask

        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label)

class NERDataset(Dataset):
    def __init__(self, args, caption_data, processor, transform, img_path=None, aux_img_path=None, max_seq=40,
                 sample_ratio=1, mode='train', ignore_idx=0) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_MEE_file(mode, sample_ratio)
        self.tokenizer = processor.tokenizer
        self.NER_label_mapping = processor.get_label_mapping()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = os.path.join('data/preprocessed_images') if img_path is not None else img_path
        self.aux_img_path = os.path.join('data/preprocessed_images',
                                         'aux_images') if aux_img_path is not None else aux_img_path
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.args = args
        self.caption_data = caption_data

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, NER_label_list, img = self.data_dict['words'][idx], self.data_dict['NER_targets'][idx], self.data_dict['imgids'][idx]
        event_type = self.data_dict['relations'][idx]
        word_list.append(event_type)
        NER_label_list.append('O')
        tokens, NER_labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            N_label = NER_label_list[i]
            for m in range(len(token)):
                if m == 0:
                    NER_labels.append(self.NER_label_mapping[N_label])
                else:
                    NER_labels.append(self.NER_label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            NER_labels = NER_labels[0:(self.max_seq - 2)]

        # text
        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
        encode_dict['attention_mask']
        # NER_labels
        NER_labels = [self.NER_label_mapping["[CLS]"]] + NER_labels + [self.NER_label_mapping["[SEP]"]] + \
                     [self.ignore_idx] * (self.max_seq - len(NER_labels) - 2)

        # image process
        img_path = os.path.join(self.img_path, img + '.pt')
        image = torch.load(img_path)
        sequence = self.caption_data['data/images/' + img]['seq']
        seq_dict = self.tokenizer.encode_plus(text=sequence, max_length=16, truncation=True, padding='max_length')
        seq_input_ids, seq_token_type_ids, seq_attention_mask = seq_dict['input_ids'], seq_dict['token_type_ids'], \
        seq_dict['attention_mask']
        seq_input_ids, seq_token_type_ids, seq_attention_mask = torch.tensor(seq_input_ids), torch.tensor(
            seq_token_type_ids), torch.tensor(seq_attention_mask)

        if self.aux_img_path is not None:
            aux_imgs = []
            aux_img_paths = self.data_dict['aux_imgs'][img]
            for path in aux_img_paths:
                aux_img_path = os.path.join(self.aux_img_path, path + '.pt')
                aux_img = torch.load(aux_img_path)
                aux_imgs.append(aux_img)
            for i in range(3 - len(aux_imgs)):
                aux_imgs.append(torch.zeros((3, 224, 224)))
            aux_imgs = torch.stack(aux_imgs, dim=0)
            assert len(aux_imgs) == 3
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), \
                torch.tensor(NER_labels), image, aux_imgs, seq_input_ids, seq_token_type_ids, seq_attention_mask, idx

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(NER_labels)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(
            NER_labels)

class MEDataset(Dataset):
    def __init__(self, args, caption_data, processor, transform, img_path=None, aux_img_path=None, max_seq=40,
                 sample_ratio=1, mode='train', ignore_idx=0) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_MEE_file(mode, sample_ratio)
        self.template_dict, self.argument_dict = processor.read_roles(args.role_path)
        self.tokenizer = processor.tokenizer
        self.ROLE_label_mapping = processor.get_label_mapping()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = os.path.join('data/preprocessed_images') if img_path is not None else img_path
        self.aux_img_path = os.path.join('data/preprocessed_images', 'aux_images') if aux_img_path is not None else None
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.args = args
        self.caption_data = caption_data

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        sentence, word_list, ROLE_label_list, img = self.data_dict['sentence'][idx], self.data_dict['words'][idx], self.data_dict['ROLE_targets'][idx],\
                                         self.data_dict['imgs'][idx]
        tokens, ROLE_labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            R_label = ROLE_label_list[i]
            for m in range(len(token)):
                if m == 0:
                    ROLE_labels.append(self.ROLE_label_mapping[R_label])
                else:
                    ROLE_labels.append(self.ROLE_label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            ROLE_labels = ROLE_labels[0:(self.max_seq - 2)]

        # prompts
        prompts = self.data_dict['prompts']
        example = self.data_dict['examples'][idx]
        counter = [0, 0, 0]
        example_id = example.doc_id
        context = example.context
        event_type_2_events = example.event_type_2_events
        list_event_type = []
        triggers = []
        for event_type, events in event_type_2_events.items():
            list_event_type += [e['event_type'] for e in events]
            triggers += [tuple(e['trigger']) for e in events]

        set_triggers = list(set(triggers))
        set_triggers = sorted(set_triggers)

        trigger_overlap = False
        for t1 in set_triggers:
            for t2 in set_triggers:
                if t1[0] == t2[0] and t1[1] == t2[1]:
                    continue
                if (t1[0] < t2[1] and t2[0] < t1[1]) or (t2[0] < t1[1] and t1[0] < t2[1]):
                    trigger_overlap = True
                    break
        if trigger_overlap:
            print('[trigger_overlap]', event_type_2_events)
            exit(0)

        offset = 0
        marked_context = deepcopy(context)
        marker_indice = list(range(len(triggers)))
        for i, t in enumerate(set_triggers):
            t_start = t[0]
            t_end = t[1]
            marked_context = marked_context[:(t_start + offset)] + ['<t-%d>' % marker_indice[i]] + \
                             context[t_start: t_end] + ['</t-%d>' % marker_indice[i]] + context[t_end:]
            offset += 2
        enc_text = " ".join(marked_context)

        old_tok_to_char_index = []  # old tok: split by oneie
        old_tok_to_new_tok_index = []  # new tok: split by BART

        curr = 0
        for tok in marked_context:
            if tok not in EXTERNAL_TOKENS:
                old_tok_to_char_index.append(
                    [curr, curr + len(tok) - 1])  # exact word start char and end char index
            curr += len(tok) + 1

        enc = self.tokenizer(enc_text, add_special_tokens=True)
        enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
        if len(enc_input_ids) > self.args.max_seq:
            raise ValueError(f"Please increase max_seq above {len(enc_input_ids)}")
        while len(enc_input_ids) < self.args.max_seq:
            enc_input_ids.append(self.tokenizer.pad_token_id)
            enc_mask_ids.append(0)

        #手动实现
        def remove_accents(text):
            # 使用unicodedata将带重音符号的字符转换为普通字符
            return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

        def manual_char_to_token(text, tokenizer, char_idx):
            tokens = tokenizer.tokenize(text)
            current_pos = 0
            token_offsets = []
            # 移除原始文本中的重音符号
            cleaned_text = remove_accents(text.lower())
            # 计算每个 token 在原始文本中的起始和结束位置
            for token in tokens:
                token_clean = remove_accents(token.replace("##", "").lower())
                start_pos = cleaned_text.find(token_clean, current_pos)
                if start_pos == -1:
                        print(f"Token '{token_clean}' not found in text '{cleaned_text}' from position {current_pos}")
                end_pos = start_pos + len(token_clean)
                token_offsets.append((start_pos, end_pos))
                current_pos = end_pos

            # 找到字符索引对应的 token 索引
            for i, (start_pos, end_pos) in enumerate(token_offsets):
                if start_pos <= char_idx < end_pos:
                    return i

            print(f"Failed to find token for character index {char_idx} in text: {text}")
            print(f"Tokens: {tokens}")
            print(f"Token offsets: {token_offsets}")
            return -1

        for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
            new_tok_s = manual_char_to_token(enc_text, self.tokenizer, char_idx_s)
            new_tok_e = manual_char_to_token(enc_text, self.tokenizer, char_idx_e) + 1
            new_tok = [new_tok_s, new_tok_e]
            old_tok_to_new_tok_index.append(new_tok)

        trigger_enc_token_index = []
        for t in triggers:
            t_start = t[0]
            t_end = t[1]
            new_t_start = old_tok_to_new_tok_index[t_start][0]
            new_t_end = old_tok_to_new_tok_index[t_end - 1][1]
            trigger_enc_token_index.append([new_t_start, new_t_end])

        dec_table_ids = []

        """ Deal with prompt template """
        list_arg_2_prompt_slots = []
        list_num_prompt_slots = []
        list_dec_prompt_ids = []
        list_arg_2_prompt_slot_spans = []
        for i, event_type in enumerate(event_type_2_events):
            dec_prompt_text = prompts[event_type].strip()
            assert dec_prompt_text
            dec_prompt = self.tokenizer(dec_prompt_text, add_special_tokens=True)
            dec_prompt_ids = dec_prompt["input_ids"]

            arg_list = self.argument_dict[event_type]
            arg_2_prompt_slots = dict()
            arg_2_prompt_slot_spans = dict()
            num_prompt_slots = 0
            if os.environ.get("DEBUG", False): arg_set = set()
            for arg in arg_list:
                prompt_slots = {
                    "tok_s": list(), "tok_e": list(),
                }
                prompt_slot_spans = []


                arg_ = arg
                # Using this more accurate regular expression might further improve rams results
                for matching_result in re.finditer(r'\b' + re.escape(arg_) + r'\b', dec_prompt_text.split('.')[0]):
                    char_idx_s, char_idx_e = matching_result.span()
                    char_idx_e -= 1
                    tok_prompt_s = manual_char_to_token(dec_prompt_text, self.tokenizer, char_idx_s)
                    tok_prompt_e = manual_char_to_token(dec_prompt_text, self.tokenizer, char_idx_e) + 1
                    prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
                    tok_prompt_s += len(dec_table_ids)
                    tok_prompt_e += len(dec_table_ids)
                    prompt_slots["tok_s"].append(tok_prompt_s)
                    prompt_slots["tok_e"].append(tok_prompt_e)
                    num_prompt_slots += 1

                arg_2_prompt_slots[arg] = prompt_slots
                arg_2_prompt_slot_spans[arg] = prompt_slot_spans

            dec_table_ids += dec_prompt_ids
            list_arg_2_prompt_slots.append(arg_2_prompt_slots)
            list_num_prompt_slots.append(num_prompt_slots)
            list_dec_prompt_ids.append(dec_prompt_ids)
            list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)

        dec_prompt_lens = len(dec_table_ids)

        row_index = 0
        list_trigger_pos = []
        list_arg_slots = []
        list_target_info = []
        list_roles = []
        """ Deal with target arguments """
        for i, (event_type, events) in enumerate(event_type_2_events.items()):
            arg_2_prompt_slots = list_arg_2_prompt_slots[i]
            num_prompt_slots = list_num_prompt_slots[i]
            dec_prompt_ids = list_dec_prompt_ids[i]
            arg_2_prompt_slot_spans = list_arg_2_prompt_slot_spans[i]
            for event in events:
                row_index += 1
                dec_event_ids = [self.tokenizer.mask_token_id] * (
                            1 + num_prompt_slots)  # 1 is for the place holder of event trigger

                list_trigger_pos.append(len(dec_table_ids))

                arg_slots = []
                cursor = len(dec_table_ids) + 1
                event_args = event['args']
                if all(not sublist for sublist in event_args):
                    event_args_name = []
                else:
                    arg_set = set([tuple(arg[:2]) for arg in event_args])
                    event_args_name = [arg[-1] for arg in event_args]

                target_info = dict()
                for arg, prompt_slots in arg_2_prompt_slots.items():
                    num_slots = len(prompt_slots['tok_s'])
                    arg_slots.append([cursor + x for x in range(num_slots)])
                    cursor += num_slots

                    arg_target = {"text": list(), "span_s": list(), "span_e": list()}
                    answer_texts, start_positions, end_positions = list(), list(), list()
                    if arg in event_args_name:
                        # Deal with multi-occurance
                        if os.environ.get("DEBUG", False): arg_set.add(arg)
                        arg_idxs = [j for j, x in enumerate(event_args_name) if x == arg]
                        if os.environ.get("DEBUG", False): counter[0] += 1; counter[1] += len(arg_idxs)

                        for arg_idx in arg_idxs:
                            event_arg_info = event_args[arg_idx]
                            answer_text = event_arg_info[2]
                            answer_texts.append(answer_text)
                            start_old, end_old = event_arg_info[0], event_arg_info[1]
                            start_position = old_tok_to_new_tok_index[start_old][0]
                            start_positions.append(start_position)
                            end_position = old_tok_to_new_tok_index[end_old - 1][1]
                            end_positions.append(end_position)

                    arg_target["text"] = answer_texts
                    arg_target["span_s"] = start_positions
                    arg_target["span_e"] = end_positions
                    target_info[arg] = arg_target

                assert sum([len(slots) for slots in arg_slots]) == num_prompt_slots

                dec_table_ids += dec_event_ids
                list_arg_slots.append(arg_slots)
                list_target_info.append(target_info)
                roles = self.argument_dict[event_type]
                assert len(roles) == len(arg_slots)
                list_roles.append(roles)

        max_dec_seq_len = self.args.max_seq
        assert len(dec_table_ids) <= max_dec_seq_len, f"\n{example.doc_id}\n{dec_table_ids}"
        while len(dec_table_ids) < max_dec_seq_len:
            dec_table_ids.append(self.tokenizer.pad_token_id)

        assert len(list_trigger_pos) == len(list_arg_slots) == len(list_target_info)

        """ Stucture-aware Attention Mask """
        dec_table_attention_mask = torch.zeros((max_dec_seq_len,), dtype=torch.int64)
        # prompt ~ prompt
        dec_table_attention_mask[:dec_prompt_lens] = 1

        event_nums_per_type = [len(events) for events in event_type_2_events.values()]
        cum_event_nums_per_type = np.cumsum(event_nums_per_type)
        cusor = 0
        for i, (arg_2_prompt_slots, dec_prompt_ids) in enumerate(zip(list_arg_2_prompt_slots, list_dec_prompt_ids)):
            event_index_start = cum_event_nums_per_type[i - 1] if i > 0 else 0
            event_index_end = cum_event_nums_per_type[i]

            arg_slots = list_arg_slots[event_index_start: event_index_end]
            assert len(arg_slots[0]) == len(arg_2_prompt_slots)
            for j, prompt_slots in enumerate(arg_2_prompt_slots.values()):
                arg_slots_same_role = [arg_slot[j] for arg_slot in arg_slots]
                for k, (start, end) in enumerate(zip(prompt_slots['tok_s'], prompt_slots['tok_e'])):
                    arg_slots_same_cloumn = [arg_slot[k] for arg_slot in arg_slots_same_role]
                    # prompt_slots -> arg_slots
                    dec_table_attention_mask[start:end] = 1
                    dec_table_attention_mask[arg_slots_same_cloumn] = 1

            len_prompt = len(dec_prompt_ids)
            list_trigger_pos_ = list_trigger_pos[event_index_start: event_index_end]
            # prompt -> triggers
            dec_table_attention_mask[cusor:cusor + len_prompt] = 1
            cusor += len_prompt

        # triggers ~ triggers
        for trigger_pos in list_trigger_pos:
            dec_table_attention_mask[trigger_pos] = 1

        for i, trigger_pos in enumerate(list_trigger_pos):
            arg_slots = list_arg_slots[i]
            num_arg_slots = sum([len(slots) for slots in arg_slots])
            # triggers ~ arg_slots
            dec_table_attention_mask[trigger_pos:trigger_pos + 1 + num_arg_slots] = 1

    ##############

        # text
        encode_dict = self.tokenizer(text=sentence, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids = encode_dict['input_ids']
        attention_mask = encode_dict['attention_mask']
        token_type_ids = encode_dict.get('token_type_ids', [0] * len(input_ids))

        # ROLE_labels
        ROLE_labels = [self.ROLE_label_mapping["[CLS]"]] + ROLE_labels + [self.ROLE_label_mapping["[SEP]"]] + \
                     [self.ignore_idx] * (self.max_seq - len(ROLE_labels) - 2)

        # image process
        img_path = os.path.join(self.img_path, img + '.pt')
        image = torch.load(img_path)
        sequence = self.caption_data['data/images/' + img]['seq']
        seq_dict = self.tokenizer(text=sequence, max_length=16, truncation=True, padding='max_length')
        seq_input_ids = seq_dict['input_ids']
        seq_attention_mask = seq_dict['attention_mask']
        seq_token_type_ids = seq_dict.get('token_type_ids', [0] * len(seq_input_ids))
        seq_input_ids, seq_token_type_ids, seq_attention_mask = torch.tensor(seq_input_ids), torch.tensor(
            seq_token_type_ids), torch.tensor(seq_attention_mask)

        if self.aux_img_path is not None:
            aux_imgs = []
            aux_img_paths = self.data_dict['aux_imgs'][img]
            for path in aux_img_paths:
                aux_img_path = os.path.join(self.aux_img_path, path + '.pt')
                aux_img = torch.load(aux_img_path)
                aux_imgs.append(aux_img)

            for i in range(3 - len(aux_imgs)):
                aux_imgs.append(torch.zeros((3, 224, 224)))

            aux_imgs = torch.stack(aux_imgs, dim=0)
            assert len(aux_imgs) == 3
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), \
                   torch.tensor(ROLE_labels), image, aux_imgs, seq_input_ids, seq_token_type_ids, seq_attention_mask, \
                   dec_table_attention_mask, idx

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(ROLE_labels)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(ROLE_labels), dec_table_attention_mask, idx


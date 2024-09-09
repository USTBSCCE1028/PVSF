from torch import nn
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch.nn.functional as F
import torch
import timm

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch32_224'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        self.outputs = {}
        for i, layer in enumerate(self.model.blocks):
            layer.register_forward_hook(self.get_layer_output(f'transformer_{i}'))

    def get_layer_output(self, name):
        def hook(model, input, output):
            self.outputs[name] = output.detach()
        return hook

    def forward(self, input_batch):
        with torch.no_grad():
            features = self.model(input_batch)
        return features, self.outputs

class ViTModel(nn.Module):
    def __init__(self, args):
        super(ViTModel, self).__init__()
        self.args = args
        self.ViT = ViTFeatureExtractor()

    def forward(self, x, aux_imgs=None):
        global_images = []
        prompt_guids = self.ViT(x)
        for name, output in prompt_guids[1].items():
            global_images.append(output)
        if aux_imgs is not None:
            aux_0, aux_1, aux_2 = [], [], []
            imgs = []
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])
            for i in range(len(aux_imgs)):
                aux_image = self.ViT(aux_imgs[i])
                for name, output in aux_image[1].items():
                    if i == 0:
                        aux_0.append(output)
                    elif i == 1:
                        aux_1.append(output)
                    elif i == 2:
                        aux_2.append(output)
            for idx in range(12):
                img = torch.cat((global_images[idx], aux_0[idx], aux_1[idx], aux_2[idx]), dim=-1)
                imgs.append(img)
            return imgs
        return prompt_guids

class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim*80, num_labels)  # Transform from 768-dim to 9-dim

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the tensor from [32, 80, 768] to [32, 80*768]
        x = self.fc(x)  # Pass data through the linear layer
        return x

class EEModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(EEModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.dropout = nn.Dropout(0.5)
        self.classifier = Classifier(self.bert.config.hidden_size, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

        if self.args.use_prompt:
            self.image_model = ViTModel(args)
            self.gates = nn.ModuleList([nn.Linear(4 * 768, 2 * 768) for i in range(12)])
            self.encoder_text = nn.Linear(768, 2*768)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None,
                seq=None, img=None, relation=None, seq_input_ids=None, seq_token_type_ids=None, seq_attention_mask=None,
                dec_table_attention_mask=None):
        #image caption
        seq_output = self.bert(
            input_ids=seq_input_ids,
            token_type_ids=seq_token_type_ids,
            attention_mask=seq_attention_mask,
            output_attentions=True,
            return_dict=True
        )
        seq_last_hidden_state, seq_pooler_output = seq_output.last_hidden_state, seq_output.pooler_output
        bsz = input_ids.size(0)
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs, seq_last_hidden_state)#门控
            prompt_guids_length = prompt_guids[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_guids = None
            prompt_attention_mask = attention_mask

        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_guids,
                    output_attentions=True,
                    return_dict=True
        )

        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        logits = self.classifier(last_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits, seq, img, relation
        return logits

    def get_visual_prompt(self, images, aux_imgs, seq_last_hidden_state):
        bsz = images.size(0)
        img_features = self.image_model(images, aux_imgs)  # [bsz,50,768*4]*12
        text_key_val = F.softmax(F.leaky_relu(self.encoder_text(seq_last_hidden_state)))
        result = []
        for idx in range(12):
            imgs_key_val = F.softmax(F.leaky_relu(self.gates[idx](img_features[idx])))
            key_val_temp = torch.cat((text_key_val, imgs_key_val), dim=1)
            key_val_temp = key_val_temp.split(768, dim=-1)
            key, value = key_val_temp[0].reshape(bsz, 12, -1, 64).contiguous(), key_val_temp[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result

class NERModel(nn.Module):
    def __init__(self, NER_label_list,args):
        super(NERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        # self.BertTokenizer = BertTokenizer.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config
        if args.use_prompt:
            self.image_model = ViTModel(args)
            self.gates = nn.ModuleList([nn.Linear(4 * 768, 2 * 768) for i in range(12)])
            self.encoder_text = nn.Linear(768, 2 * 768)
        self.NER_num_labels = len(NER_label_list)  # pad
        self.NER_crf = CRF(self.NER_num_labels, batch_first=True)
        self.NER_fc = nn.Linear(self.bert.config.hidden_size, self.NER_num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, NER_labels=None, images=None, aux_imgs=None,
                seq_input_ids=None, seq_token_type_ids=None, seq_attention_mask=None):
        if self.args.use_prompt:
            #image caption
            seq_output = self.bert(
                input_ids=seq_input_ids,
                token_type_ids=seq_token_type_ids,
                attention_mask=seq_attention_mask,
                output_attentions=True,
                return_dict=True
            )
            seq_last_hidden_state, seq_pooler_output = seq_output.last_hidden_state, seq_output.pooler_output
            prompt_guids = self.get_visual_prompt(images, aux_imgs, seq_last_hidden_state)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        bert_output = self.bert(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            token_type_ids=token_type_ids,
                            past_key_values=prompt_guids,
                            return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.NER_fc(sequence_output)  # bsz, len, labels
        logits = self.NER_crf.decode(emissions, attention_mask.byte())
        loss = None
        if NER_labels is not None:
            loss = -1 * self.NER_crf(emissions, NER_labels, attention_mask.byte(), reduction='mean')

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def get_visual_prompt(self, images, aux_imgs, seq_last_hidden_state):
        bsz = images.size(0)
        img_features = self.image_model(images, aux_imgs)  # [bsz,50,768*4]*12
        text_key_val = F.softmax(F.leaky_relu(self.encoder_text(seq_last_hidden_state)))
        result = []
        for idx in range(12):
            imgs_key_val = F.softmax(F.leaky_relu(self.gates[idx](img_features[idx])))
            key_val_temp = torch.cat((text_key_val, imgs_key_val), dim=1)
            key_val_temp = key_val_temp.split(768, dim=-1)
            key, value = key_val_temp[0].reshape(bsz, 12, -1, 64).contiguous(), key_val_temp[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result


class MEModel(nn.Module):
    def __init__(self, ROLE_label_list, args):
        super(MEModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        if args.use_prompt:
            self.image_model = ViTModel(args)
            self.gates = nn.ModuleList([nn.Linear(4 * 768, 2 * 768) for i in range(12)])
            self.encoder_text = nn.Linear(768, 1536)
        self.ME_num_labels = len(ROLE_label_list)  # pad
        self.ME_crf = CRF(self.ME_num_labels, batch_first=True)
        self.ME_fc = nn.Linear(self.bert.config.hidden_size, self.ME_num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, ROLE_labels=None, images=None, aux_imgs=None,
                seq_input_ids=None, seq_token_type_ids=None, seq_attention_mask=None, dec_table_attention_mask=None):
        if self.args.use_prompt:
            #image caption
            seq_output = self.bert(
                input_ids=seq_input_ids,
                token_type_ids=seq_token_type_ids,
                attention_mask=seq_attention_mask,
                output_attentions=True,
                return_dict=True
            )
            seq_last_hidden_state, seq_pooler_output = seq_output.last_hidden_state, seq_output.pooler_output
            prompt_guids = self.get_visual_prompt(images, aux_imgs, seq_last_hidden_state)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        if dec_table_attention_mask is not None:
            padded_dec_table_attention_mask = torch.zeros_like(prompt_attention_mask)
            padded_dec_table_attention_mask[:, :dec_table_attention_mask.size(1)] = dec_table_attention_mask
            prompt_attention_mask = (prompt_attention_mask.bool() | padded_dec_table_attention_mask.bool()).float()


        bert_output = self.bert(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            token_type_ids=token_type_ids,
                            past_key_values=prompt_guids,
                            return_dict=True)

        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.ME_fc(sequence_output)  # bsz, len, labels
        logits = self.ME_crf.decode(emissions, attention_mask.byte())
        loss = None
        if ROLE_labels is not None:
            loss = -1 * self.ME_crf(emissions, ROLE_labels, attention_mask.byte(), reduction='mean')
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    # 修改 self.get_visual_prompt 函数，使得返回的 prompt_guids 维度正确
    def get_visual_prompt(self, images, aux_imgs, seq_last_hidden_state):
        bsz = images.size(0)
        img_features = self.image_model(images, aux_imgs)  # [bsz,50,768*4]*12
        text_key_val = F.softmax(F.leaky_relu(self.encoder_text(seq_last_hidden_state)))
        result = []
        for idx in range(12):
            imgs_key_val = F.softmax(F.leaky_relu(self.gates[idx](img_features[idx])))
            key_val_temp = torch.cat((text_key_val, imgs_key_val), dim=1)
            key_val_temp = key_val_temp.split(768, dim=-1)
            key, value = key_val_temp[0].reshape(bsz, 12, -1, 64).contiguous(), key_val_temp[1].reshape(bsz, 12, -1,
                                                                                                        64).contiguous()# bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result



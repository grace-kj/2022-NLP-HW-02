# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

def _expand_mask(mask, tgt_len = None):
    """
        Inputs
            mask.shape = (B, S_L)
        Outputs
            output.shape = (B, 1, T_L, S_L)
    """
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(torch.float)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(torch.float).min)

def _make_causal_mask(dec_ids):
    """
        Inputs
            dec_ids.shape = (B, D_L)
        Outputs
            output.shape = (B, 1, D_L, D_L)
    """
    batch_size, tgt_len = dec_ids.size()
    device = dec_ids.device

    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(torch.float).to(device)

    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoding = torch.zeros(1024, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, 1024)
        pos = pos.float().unsqueeze(dim = 1)

        _2i = torch.arange(0, d_model, step = 2).float()

        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        device = x.device

        return self.encoding[:seq_len, :].unsqueeze(0).to(device)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_head = self.d_model // self.num_heads
        self.scaling = self.d_head ** -0.5

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(0.1)
    
    #num_heads 개수만큼 q, k, v를 split
    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2).contiguous()

    def forward(self, query_states, key_value_states, attention_mask):
        '''
            Inputs
                query_states.shape = (B, target_len, H) 
                key_value_states.shape = (B, source_len, H)
                attention_mask.shape = (B, 1, target_len, source_len)
                    example)
                        target : ['나는', '점심', '먹는다', '<pad>', '<pad>']
                        source : ['점심', '먹는다']
                        attention_mask : 
                                [[0, 0, 0, -inf, -inf],
                                [0, 0, 0, -inf, -inf]]
                    tip)
                        query와 key를 이용해 attention score를 구하고 난 후, 해당 값에 attention_mask를 더해준다.
                        이후 softmax를 취해 attention prob를 계산.
            Outputs
                attn_output.shape = (B, target_len, H)
        '''
        attn_output = None
        ############################################## EDIT ################################################
        #Transformer의 multi-headattention구현.               
        
        query, key, value, mask = query_states, key_value_states, key_value_states, attention_mask
        batch_size = query.shape[0]

        # 1. 각각 W 지나기
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # 2. head 나누기
        query_split = self._shape(query, query.shape[1], batch_size)
        key_split = self._shape(key, key.shape[1], batch_size)
        value_split = self._shape(value, value.shape[1], batch_size)

        # 3. scaled dot product 어텐션
        # Q행렬과 K행렬을 전치한 행렬을 곱하고, 소프트맥스 함수를 사용하여 어텐션 분포 행렬을 얻은 뒤에 V행렬과 곱한다
        d_K = torch.tensor(key.size()[-1])
        scores = torch.matmul(query_split, key_split.permute(0, 1, 3, 2)) / torch.sqrt(d_K)
        if mask is not None:
            scores = scores.masked_fill(mask!=0,-1e9)
        attention = torch.softmax(scores, dim=-1)

        x_raw = torch.matmul(self.dropout(attention),value_split)
        x_rsh1 = x_raw.permute(0,2,1,3).contiguous()

        # 4. 헤드 concatenate하기
        # (batch_size, query의 문장 길이, d_model)
        x_concat = x_rsh1.view(batch_size,-1,self.d_model)

        # 5. WO에 해당하는 밀집층 지나기
        attn_output = self.out_proj(x_concat)
        
        ############################################## EDIT ################################################
        return attn_output

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        self.self_attn = MultiHeadAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.activation_fn = nn.ReLU()

        self.fc1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.fc2 = nn.Linear(4 * self.d_model, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, enc_self_mask):
        '''
            Inputs
                hidden_states.shape = (B, E_L, H)
                enc_self_mask.shape = (B, 1, E_L, E_L)
            Outputs
                hidden_states.shape = (B, E_L, H)
        '''

        ############################################## EDIT ################################################
        #구현한 multi-headattention을 바탕으로 TransformerEncoderlayer구현

        # 패딩 마스크 사용
        padding_mask = enc_self_mask

        # 1. 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
        attention = self.self_attn(hidden_states, hidden_states, padding_mask)

        # 2. 드롭아웃 + 잔차 연결과 층 정규화
        attention = self.dropout(attention)
        attention = self.self_attn_layer_norm(hidden_states + attention)

        # 3. 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
        outputs = self.activation_fn(self.fc1(attention))
        outputs = self.fc2(outputs)

        # 4. 드롭아웃 + 잔차 연결과 층 정규화
        outputs = self.dropout(outputs)
        hidden_states = self.self_attn_layer_norm(hidden_states + outputs)

        ############################################## EDIT ################################################
        
        return hidden_states

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        self.self_attn = MultiHeadAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.activation_fn = nn.ReLU()

        self.cross_attn = MultiHeadAttention(config)
        self.cross_attn_layer_norm = nn.LayerNorm(self.d_model)

        self.fc1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.fc2 = nn.Linear(4 * self.d_model, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, dec_self_mask = None, enc_hidden_states = None, enc_dec_mask = None):
        '''
            Inputs
                hidden_states.shape = (B, D_L, H),
                dec_self_mask.shape = (B, 1, D_L, D_L),
                enc_hidden_states.shape = (B, E_L, H),
                enc_dec_mask.shape = (B, 1, D_L, E_L)

            Outputs
                hidden_states.shape = (B, D_L, H)
        '''

        ############################################## EDIT ################################################
        #구현한 multi-headattention을 바탕으로 TransformerDecoderlayer구현

        # 룩어헤드 마스크(첫번째 서브층 용)
        look_ahead_mask = dec_self_mask

        # 패딩 마스크(두번째 서브층 용)
        padding_mask = enc_dec_mask

        # 1-1. 멀티-헤드 어텐션 (첫번째 서브층 / 마스크드 셀프 어텐션)
        attention1 = self.self_attn(hidden_states, hidden_states, look_ahead_mask)

        # 1-2. 잔차 연결과 층 정규화
        attention1 = self.dropout(attention1)
        attention1 = self.self_attn_layer_norm(hidden_states + attention1)

        # 2-1. 멀티-헤드 어텐션 (두번째 서브층 / 디코더-인코더 어텐션)
        attention2 = self.cross_attn(attention1, enc_hidden_states, padding_mask)

        # 2-2. 드롭아웃 + 잔차 연결과 층 정규화
        attention2 = self.dropout(attention2)
        attention2 = self.cross_attn_layer_norm(attention1 + attention2)

        # 3-1. 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
        outputs = self.activation_fn(self.fc1(attention2))
        outputs = self.fc2(outputs)

        # 3-2. 드롭아웃 + 잔차 연결과 층 정규화
        outputs = self.dropout(outputs)
        hidden_states = self.final_layer_norm(outputs + attention2)

        ############################################## EDIT ################################################
        
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, config, embed_tokens, embed_positions):
        super().__init__()
        self.d_model = config.d_model

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.embedding_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, enc_ids, enc_mask):
        '''
            Inputs
                enc_ids.shape = (B, E_L)
                enc_mask.shape = (B, E_L)
            Outputs
                output.shape = (B, E_L, H)
        '''
        token_embedding = self.embed_tokens(enc_ids)
        pos_embedding = self.embed_positions(enc_ids)
        hidden_states = token_embedding + pos_embedding
        hidden_states = self.embedding_layer_norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p = 0.1, training = self.training)

        enc_self_mask = _expand_mask(enc_mask)
        # enc_self_mask.shape = (B, 1, E_L, E_L)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, enc_self_mask)

        enc_hidden_states = hidden_states

        return enc_hidden_states

class Decoder(nn.Module):
    def __init__(self, config, embed_tokens, embed_positions):
        super().__init__()
        self.d_model = config.d_model

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        #layers는 N번 레이어 쌓은 디코더 
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.embedding_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, dec_ids, dec_mask = None, enc_hidden_states = None, enc_mask = None):
        '''
            Inputs
                dec_ids.shape = (B, D_L)
                dec_mask.shape = (B, D_L)
                enc_hidden_states.shape = (B, E_L, H)
                enc_mask.shape = (B, E_L)
            Outputs
                hidden_states.shape = (B, D_L, H)
        '''
        token_embedding = self.embed_tokens(dec_ids)
        pos_embedding = self.embed_positions(dec_ids)
        hidden_states = token_embedding + pos_embedding
        hidden_states = self.embedding_layer_norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p = 0.1, training = self.training)

        #디코더 첫번째 서브층 룩어헤드 마스크
        dec_self_mask = _make_causal_mask(dec_ids) + _expand_mask(dec_mask)

        enc_dec_mask = _expand_mask(enc_mask, dec_ids.shape[-1])

        #디코더 층 N개 통과
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, dec_self_mask, enc_hidden_states, enc_dec_mask)

        dec_hidden_states = hidden_states

        return dec_hidden_states

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.embed_positions = PositionalEncoding(config.d_model)

        self.encoder = Encoder(config, self.embed_tokens, self.embed_positions)
        self.decoder = Decoder(config, self.embed_tokens, self.embed_positions)

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask = None, enc_hidden_states = None):
        '''
            Inputs
                enc_ids.shape = (B, E_L)
                enc_mask.shape = (B, E_L)
                dec_ids.shape = (B, D_L)
                dec_mask.shape = (B, D_L)
                enc_hidden_states.shape = (B, E_L, H) batch, enc_len, hidden
            Outputs
                enc_hidden_states.shape = (B, E_L, H)
                dec_hidden_states.shape = (B, D_L, H) batch, dec_len, hidden
        '''
        #encoder
        if enc_hidden_states is None:
            enc_hidden_states = self.encoder(enc_ids, enc_mask)

        #decoder
        dec_hidden_states = self.decoder(dec_ids, dec_mask, enc_hidden_states, enc_mask)

        return enc_hidden_states, dec_hidden_states

class TransformerForConditionalGeneration(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = Transformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias = False)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask, label_ids):
        '''
            Inputs
                enc_ids.shape = (B, E_L) batch, enc_len
                enc_mask.shape = (B, E_L)
                dec_ids.shape = (B, D_L) batch, dec_len
                dec_mask.shape = (B, D_L)
                label_ids.shape = (B, D_L)
            Outputs
                lm_loss
        '''
        lm_loss = None
        ############################################## EDIT ################################################
        #Transformer학습에 사용되는 cross-entropy loss값을 계산하는 코드 구현

        enc_hidden_states, dec_hidden_states = self.model(enc_ids, enc_mask, dec_ids, dec_mask)
        #logits
        out =  self.lm_head(dec_hidden_states)
        output = out.contiguous().view([1,-1, self.config.vocab_size]).squeeze()
        label_ids = label_ids.contiguous().view([1,-1]).squeeze()
        
        #loss
        lm_loss = self.criterion(output, label_ids)

        ############################################## EDIT ################################################
        return lm_loss

    def generate(self, enc_ids, enc_mask) -> str:
        '''
            Inputs
                enc_ids.shape = (1, E_L)
                enc_mask.shape = (1, E_L)
            Outputs
                hypothesis : str

            Guide
                1. 학습된 Transformer 모델을 바탕으로 enc_ids와 enc_mask를 입력으로 받아 요약문 생성.
                2. eos_token이 생성될 때까지 반복문을 이용해 token 생성
        '''
        hypothesis = []
        # hypothesis type : List[int]

        dec_ids = torch.tensor([self.tokenizer.bos_token_id] * 1, dtype = torch.long, device = enc_ids.device).unsqueeze(-1)
        dec_mask = torch.ones_like(dec_ids)
        enc_hidden_states = None

        ############################################## EDIT ################################################
        #학습된 모델을 이용해 enc_ids 와 enc_mask를 입력하는 해당 입력에 맞는 summary를 생성하는 코드 구현
        
        #eos token이 나올 때까지.
        enc_hidden_states = self.model.encoder(enc_ids, enc_mask)

        for i in range(self.config.dec_len-1):
          dec_hidden_states = self.model.decoder(dec_ids, dec_mask, enc_hidden_states, enc_mask)
          #logit
          out =  self.lm_head(dec_hidden_states)
          out = F.softmax(out, dim=-1)
          output = torch.argmax(out, dim = -1) #output은 batch size * (output sequence length)
          #print(output, i)

          next_token = output[:,-1].view([1,1])
          dec_ids = torch.cat([dec_ids, next_token], dim=1)
          dec_mask = torch.ones_like(dec_ids)
          
          if next_token.item() == 2:
            hypothesis = dec_ids.view([-1])
            break

        if i == self.config.dec_len-2:
          dec_ids = torch.cat([dec_ids, torch.tensor([self.tokenizer.eos_token_id] * 1, dtype = torch.long, device = enc_ids.device).unsqueeze(-1)], dim=1)
          hypothesis = dec_ids.view([-1])

        #print(self.tokenizer.decode(hypothesis))

        ############################################## EDIT ################################################

        return self.tokenizer.decode(hypothesis)    
            
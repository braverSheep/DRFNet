import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import SFSWTS_Swin

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

class ScaledDotProductAttention(nn.Module):


    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))  # 注意力
        output = torch.matmul(attn, v)  # 注意力分数乘以v

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        q += residual
        q = self.layer_norm(q)
        return q, attn

class PositionwiseFeedForward(nn.Module):
    

    def __init__(self, d_in, d_hid, dropout=0.1):  # d_in=d_model=310, d_inner=d_ff= args.ffn_hidden=512
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.w_2(self.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):   #
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.self_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
       

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):
   

    def __init__(self, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, n_position=200):

        super().__init__()

        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []
        # -- Forward
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class CorruptionLayer(nn.Module):
    def __init__(self, device, corrupt_probability=0.1):
        super(CorruptionLayer, self).__init__()
        self.corrupt_p = corrupt_probability
        self.device = device

    def forward(self, feature):
        bitmask = torch.FloatTensor(feature.shape).uniform_() > self.corrupt_p
        return torch.mul(feature, bitmask)

class TransformerEncoder(nn.Module):

    def __init__(self, sentence_len, d_feature, n_layers=6, n_heads=8, p_drop=0.5, d_ff=2048):
        super(TransformerEncoder, self).__init__()
        d_k = d_v = d_feature // n_heads
        self.swin = SFSWTS_Swin.SFSWTS()
        self.sentence_len = sentence_len
        self.encoder = Encoder(n_position=sentence_len,
                               d_word_vec=d_feature, d_model=d_feature, d_inner=d_ff,
                               n_layers=n_layers, n_head=n_heads, d_k=d_k, d_v=d_v,
                               dropout=p_drop)

        # self.linear = nn.Linear(d_feature, 3)  # SEED
        # self.linear = nn.Linear(d_feature, 4)  #SEED-IV
        # self.linear = nn.Linear(d_feature, 5)  #SEED-V
        self.linear = nn.Linear(d_feature, 7)    #SEED-VII
        self.softmax = nn.Softmax(dim=-1)
        self.Linear1 = nn.Linear(64 * self.sentence_len, 405 * self.sentence_len)

    def forward(self, src_seq):
        src_mask = None
        Src_seq = None
        Attention = None
        shape = src_seq.shape
        for batch in range(src_seq.shape[0]):
            tt = src_seq[batch].reshape([-1,5, 9, 9])  # 2，5，9，9
            att = self.swin(tt)
            att = att.reshape(shape[1], -1, 1, 1)
            att = torch.unsqueeze(att, 1)
            out_put = torch.reshape(att, (-1,))
            if Src_seq == None:
                Src_seq = out_put
            else:
                Src_seq = torch.vstack((Src_seq, out_put))

        if src_seq.shape[0] == 1:
            SRC_seq = torch.unsqueeze(Src_seq, 0)
            SRC_seq = torch.unsqueeze(SRC_seq, 0)
        else:
            SRC_seq = Src_seq.reshape(src_seq.shape[0],src_seq.shape[1],-1)


        # print(SRC_seq.shape)

        outputs_feature, *_ = self.encoder(SRC_seq, src_mask)
        outputs, _ = torch.max(outputs_feature, dim=1)
        outputs_classification = self.softmax(self.linear(outputs))
        return outputs_feature, outputs_classification, Attention



if __name__ == "__main__":
    x = torch.randn(16, 5, 5, 9, 9).cuda()

    net = TransformerEncoder(sentence_len=62, d_feature=384).cuda()

    outputs_feature, outputs_classification, Attention =  net(x)

    print(outputs_classification.shape)
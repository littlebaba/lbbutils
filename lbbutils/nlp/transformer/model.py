import numpy as np
import torch
import torch.nn as nn

from .layer import EncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec, n_layer, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_postion=200, scale_emb=False):
        super().__init__()
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_postion)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self):
        pass


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forword(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output

class Transformer(nn.Module):
    def __init__(self,n_src_vocab,n_trg_vocab,src_pad_idx,trg_pad_idx,
                 d_word_vec=512,d_model=512,d_inner=2048,n_layers=6,
                 n_head=8,d_k=64,d_v=64,dropout=0.1,
                 n_position=200,trg_emb_prj_weight_sharing=True,emb_src_trg_weight_sharing=True,scale_emb_or_prj='prj'):
        super().__init__()
        self.src_pad_idx,self.trg_pad_idx = src_pad_idx,trg_pad_idx
        assert scale_emb_or_prj in ['emb','prj','none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import entmax

# Adapted from
# https://github.com/tensorflow/tensor2tensor/blob/0b156ac533ab53f65f44966381f6e147c7371eee/tensor2tensor/layers/common_attention.py
from seq2struct.utils import registry
from seq2struct.utils.vocab import Vocab

from seq2struct.commands import training_logger_global


def relative_attention_logits(query, key, relation):
    # We can't reuse the same logic as tensor2tensor because we don't share relation vectors across the batch.
    # In this version, relation vectors are shared across heads.
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # qk_matmul is [batch, heads, num queries, num kvs]
    qk_matmul = torch.matmul(query, key.transpose(-2, -1))

    # q_t is [batch, num queries, heads, depth]
    q_t = query.permute(0, 2, 1, 3)

    # r_t is [batch, num queries, depth, num kvs]
    r_t = relation.transpose(-2, -1)

    #   [batch, num queries, heads, depth]
    # * [batch, num queries, depth, num kvs]
    # = [batch, num queries, heads, num kvs]
    # For each batch and query, we have a query vector per head.
    # We take its dot product with the relation vector for each kv.
    q_tr_t_matmul = torch.matmul(q_t, r_t)

    # qtr_t_matmul_t is [batch, heads, num queries, num kvs]
    q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)

    # [batch, heads, num queries, num kvs]
    return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])

    # Sharing relation vectors across batch and heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [num queries, num kvs, depth].
    #
    # Then take
    # key reshaped
    #   [num queries, batch * heads, depth]
    # relation.transpose(-2, -1)
    #   [num queries, depth, num kvs]
    # and multiply them together.
    #
    # Without sharing relation vectors across heads:
    # query: [batch, heads, num queries, depth].
    # key: [batch, heads, num kvs, depth].
    # relation: [batch, heads, num queries, num kvs, depth].
    #
    # Then take
    # key.unsqueeze(3)
    #   [batch, heads, num queries, 1, depth]
    # relation.transpose(-2, -1)
    #   [batch, heads, num queries, depth, num kvs]
    # and multiply them together:
    #   [batch, heads, num queries, 1, depth]
    # * [batch, heads, num queries, depth, num kvs]
    # = [batch, heads, num queries, 1, num kvs]
    # and squeeze
    # [batch, heads, num queries, num kvs]


def relative_attention_values(weight, value, relation):
    # In this version, relation vectors are shared across heads.
    # weight: [batch, heads, num queries, num kvs].
    # value: [batch, heads, num kvs, depth].
    # relation: [batch, num queries, num kvs, depth].

    # wv_matmul is [batch, heads, num queries, depth]
    wv_matmul = torch.matmul(weight, value)

    # w_t is [batch, num queries, heads, num kvs]
    w_t = weight.permute(0, 2, 1, 3)

    #   [batch, num queries, heads, num kvs]
    # * [batch, num queries, num kvs, depth]
    # = [batch, num queries, heads, depth]
    w_tr_matmul = torch.matmul(w_t, relation)

    # w_tr_matmul_t is [batch, heads, num queries, depth]
    w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)

    return wv_matmul + w_tr_matmul_t


# Adapted from The Annotated Transformer
def clones(module_fn, N):
    return nn.ModuleList([module_fn() for _ in range(N)])


# liyutian
# Adapted from The Annotated Transformer
def clones2(module_fn_1, module_fn_2):
    return nn.ModuleList([module_fn_1(), module_fn_2()])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
    return torch.matmul(p_attn, value), p_attn


def sparse_attention(query, key, value, alpha, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    if alpha == 2:
        p_attn = entmax.sparsemax(scores, -1)
    elif alpha == 1.5:
        p_attn = entmax.entmax15(scores, -1)
    else:
        raise NotImplementedError
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
    return torch.matmul(p_attn, value), p_attn


# Adapted from The Annotated Transformers
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(lambda: nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if query.dim() == 3:
            x = x.squeeze(1)
        return self.linears[-1](x)


# Adapted from The Annotated Transformer
def attention_with_relations(query, key, value, relation_k, relation_v, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = relative_attention_logits(query, key, relation_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn_orig = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn_orig)
    return relative_attention_values(p_attn, value, relation_v), p_attn_orig


# liyutian
# Adapted from The Annotated Transformer
def attention_with_history_relations(query, key, value, relation_k, relation_v, history_weight=None, mask=None, dropout=None, sep_id=None,history_reg=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = relative_attention_logits(query, key, relation_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # max
    # sep_select = torch.index_select(scores, -2, sep_id)
    # sep_pool = torch.max(sep_select,-1)[0]

    history_weight = history_weight.view(1,history_weight.shape[0],history_weight.shape[1],1)
    sep_select = torch.index_select(query, -2, sep_id)
    sep_pool = torch.matmul(sep_select,history_weight)
    sep_pool = sep_pool.view(sep_pool.shape[0],sep_pool.shape[1],sep_pool.shape[2])
    sep_pred = F.softmax(sep_pool,dim=-1)

    zeros = scores-scores
    for i in range(sep_pred.shape[-1]):
        _sep_pred = sep_pred[:,:,i].view(sep_pred.shape[0],sep_pred.shape[1],1,1)
        _sep_pred = _sep_pred.expand(zeros[:,:,sep_id[i]:,sep_id[i]:].shape)
        zeros[:,:,sep_id[i]:,sep_id[i]:] += scores[:,:,sep_id[i]:,sep_id[i]:]*_sep_pred
    scores = zeros

    # reg loss 1
    loss = torch.mean(1 - torch.norm(sep_pred, float('inf'), dim=-1))

    # calculate reg loss 2
    sep_pred = torch.log(sep_pred.view(-1, sep_pred.shape[-1]))
    history_reg = history_reg.expand(sep_pred.shape)
    loss += F.kl_div(sep_pred, history_reg)

    p_attn_orig = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn_orig)
    return relative_attention_values(p_attn, value, relation_v), p_attn_orig, loss


class PointerWithRelations(nn.Module):
    def __init__(self, hidden_size, num_relation_kinds, dropout=0.2):
        super(PointerWithRelations, self).__init__()
        self.hidden_size = hidden_size
        self.linears = clones(lambda: nn.Linear(hidden_size, hidden_size), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.relation_k_emb = nn.Embedding(num_relation_kinds, self.hidden_size)
        self.relation_v_emb = nn.Embedding(num_relation_kinds, self.hidden_size)

    def forward(self, query, key, value, relation, mask=None):
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)

        if mask is not None:
            mask = mask.unsqueeze(0)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, 1, self.hidden_size).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        _, self.attn = attention_with_relations(
            query,
            key,
            value,
            relation_k,
            relation_v,
            mask=mask,
            dropout=self.dropout)

        return self.attn[0, 0]


# Adapted from The Annotated Transformer
class MultiHeadedAttentionWithRelations(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithRelations, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(lambda: nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, relation_k, relation_v, mask=None):
        # query shape: [batch, num queries, d_model]
        # key shape: [batch, num kv, d_model]
        # value shape: [batch, num kv, d_model]
        # relations_k shape: [batch, num queries, num kv, (d_model // h)]
        # relations_v shape: [batch, num queries, num kv, (d_model // h)]
        # mask shape: [batch, num queries, num kv]
        if mask is not None:
            # Same mask applied to all h heads.
            # mask shape: [batch, 1, num queries, num kv]
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: [batch, heads, num queries, depth]
        x, self.attn = attention_with_relations(
            query,
            key,
            value,
            relation_k,
            relation_v,
            mask=mask,
            dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# liyutian
# Adapted from The Annotated Transformer
class MultiHeadedAttentionWithHistoryRelations(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithHistoryRelations, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(lambda: nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        #liyutian
        self.history_weights = nn.Parameter(torch.Tensor(h,self.d_k))
        nn.init.kaiming_uniform_(self.history_weights, a=math.sqrt(5))

    def forward(self, query, key, value, relation_k, relation_v, mask=None, sep_id=None, history_reg=None):
        # query shape: [batch, num queries, d_model]
        # key shape: [batch, num kv, d_model]
        # value shape: [batch, num kv, d_model]
        # relations_k shape: [batch, num queries, num kv, (d_model // h)]
        # relations_v shape: [batch, num queries, num kv, (d_model // h)]
        # mask shape: [batch, num queries, num kv]
        if mask is not None:
            # Same mask applied to all h heads.
            # mask shape: [batch, 1, num queries, num kv]
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x shape: [batch, heads, num queries, depth]
        x, self.attn, loss = attention_with_history_relations(
            query,
            key,
            value,
            relation_k,
            relation_v,
            history_weight = self.history_weights,
            mask=mask,
            dropout=self.dropout,
            sep_id=sep_id,
            history_reg=history_reg)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), loss


# Adapted from The Annotated Transformer
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, layer_size, N, tie_layers=False):
        super(Encoder, self).__init__()
        if tie_layers:
            self.layer = layer()
            self.layers = [self.layer for _ in range(N)]
        else:
            self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer_size)

        # TODO initialize using xavier

    def forward(self, x, relation, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, relation, mask)
        return self.norm(x)


# liyutian
# Adapted from The Annotated Transformer
class HistoryEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, layer_size, N, tie_layers=False, device=None):
        super(HistoryEncoder, self).__init__()
        self._device = device
        if tie_layers:
            self.layer = layer()
            self.layers = [self.layer for _ in range(N)]
        else:
            self.layers = clones(layer, N)  # transformer.HistoryEncoderLayer
        self.norm = nn.LayerNorm(layer_size)

        # TODO initialize using xavier

    def forward(self, x, relation, mask, sep_id, history_reg):
        "Pass the input (and mask) through each layer in turn."
        loss = torch.tensor(0.0).to(self._device)
        for layer in self.layers:
            x, _loss = layer(x, relation, mask, sep_id, history_reg)
            loss += _loss

        return self.norm(x), loss


# zhanghanchu
@registry.register('multitask', 'turn-switch-classifier')
class TurnSwitchClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, vocab_path, dropout=0.1, device=None):
        super(TurnSwitchClassifier, self).__init__()
        self.turn_len = 6
        self._device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self._loss = nn.BCEWithLogitsLoss()

        self.vocab = Vocab.load(vocab_path)
        self.switch_label_size = len(self.vocab)

        self.counter = 0

        # self.turn_switch_label_embeddings = torch.nn.Embedding(
        #     num_embeddings=self.switch_label_size,
        #     embedding_dim=self.input_dim
        # )

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self._classification_layer = torch.nn.Sequential(
            nn.Linear(self.input_dim, self.switch_label_size),
            nn.LeakyReLU(0.2)
        )

    def create_label_t(self, desc):
        turn_change_index = desc['turn_change_index']
        lable_tc = np.zeros((self.turn_len, self.switch_label_size))
        for i, tc in enumerate(turn_change_index):
            for item in tc:
                lable_tc[i, item] = 1.0

        lable_tc_t = torch.tensor(lable_tc).to(self._device)
        return lable_tc_t

    def create_mask_t(self, turn_change_index):
        tc_mask = np.zeros(self.turn_len)
        tc_mask[1:len(turn_change_index)] = 1.0
        tc_mask_t = torch.tensor(tc_mask).to(self._device)
        return tc_mask_t

    def forward(self, embedded_text_input, sep_id, mask, label):
        # bs=1 (bs,sep_len,dim)
        sep_select_embedded = torch.index_select(embedded_text_input, -2, sep_id.long())  # .int()

        if self.dropout:
            sep_select_embedded = self.dropout(sep_select_embedded)

        # (bs, sep_len, dim) -->  (sep_len, switch_label_size)
        logits = torch.squeeze(self._classification_layer(sep_select_embedded))

        # apply sep_select_embedded_mask=(bs,sep_len)
        mask = torch.unsqueeze(mask > 0, -1).expand_as(logits)

        logits_m = torch.masked_select(logits, mask)
        label_m = torch.masked_select(label, mask)

        output_dict = {"logits": logits}

        if label is not None and logits_m.shape[0] != 0:
            loss = self._loss(logits_m, label_m)
            output_dict['loss'] = loss

            if (torch.isnan(loss).sum() > 0):
                print("here!")
                print(
                    "A lap in this 'sky pool' may have you holding your breath, and not just because you're underwater."
                    " With the streets of London looming 10 stories down, "
                    "the view through the pool's clear bottom is a bit freaky to all but the fearless. ")
                print(logits_m)
                print("=====label_m=====")
                print(label_m)
                print("=====logits_m.sigmoid()=====")
                print(logits_m.sigmoid())
                print("==========")

            self.counter += 1
            if self.counter % 10 == 0:
                print(output_dict['loss'])
        else:
            output_dict['loss'] = 0.0

        return output_dict
        # label_emb = self.turn_switch_label_embeddings(label)  # (bs,switch_label_size,dim)
        # (bs,sep_len,dim) * (bs,switch_label_size,dim) --> (bs,sep_len, switch_label_size)
        # scores = torch.matmul(sep_select_embedded, label_emb.transpose(2, 3))
        # logits = torch.sigmoid(scores.float(), dim=-1).type_as(scores) #(bs,sep_len, switch_label_size)

        # similarity_multi = F.cosine_similarity(vector1, vector2, dim=3)
        # logits = self._classification_layer(sep_select_embedded)
        #
        # probs = torch.nn.functional.softmax(logits, dim=-1)

@registry.register('multitask', 'dynamic_loss_weight')
class TurnSwitchDynamicLossWeight(nn.Module):
    def __init__(self, input_dim, hidden_dim,  device=None):
        super(TurnSwitchDynamicLossWeight, self).__init__()
        self._device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        weight_layer = torch.nn.Sequential(
            nn.Linear(self.input_dim * 4, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),      #====>输出的一个分数
            nn.Sigmoid()    #====>分数压缩到(0,1)之间
        )
        self.layers = clones(lambda: weight_layer, 2)

    def forward(self, embedded_input):
        scores = [torch.squeeze(layer(embedded_input)) for layer in self.layers]
        return scores

@registry.register('multitask', 'turn-switch-classifier-interact')
class TurnSwitchClassifierInteract(nn.Module):

    def __init__(self, input_dim, hidden_dim, vocab_path, dropout=0.1, device=None, leaky_rate=0.2, loss_scalar=4, mid_layer_activator='relu',max_turn_len=6, report_loss_every_n=20):
        super(TurnSwitchClassifierInteract, self).__init__()
        self.turn_len = max_turn_len
        self._device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  #用不着

        self._loss = nn.BCELoss() #替换掉 nn.BCEWithLogitsLoss()

        self.vocab = Vocab.load(vocab_path)
        self.switch_label_size = len(self.vocab)

        self.counter = 0

        self.loss_scalar = loss_scalar

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.report_loss_every_n = report_loss_every_n
        print("mid_layer_activator:{}".format(mid_layer_activator))
        self._classification_layer = torch.nn.Sequential(
            nn.Linear(self.input_dim * 4, self.hidden_dim),
            nn.LeakyReLU(leaky_rate) if mid_layer_activator == 'relu' else nn.Tanh(),
            nn.Linear(self.hidden_dim, self.switch_label_size),
            nn.Sigmoid()
        )

    def create_label_t(self, desc):
        turn_change_index = desc['turn_change_index']
        lable_tc = np.zeros((self.turn_len, self.switch_label_size))
        for i, tc in enumerate(turn_change_index):
            for item in tc:
                lable_tc[i, item] = 1.0

        lable_tc_t = torch.tensor(lable_tc).to(self._device)
        return lable_tc_t

    def create_mask_t(self, desc):
        turn_change_index = desc['turn_change_index']
        tc_mask = np.zeros(self.turn_len)
        tc_mask[1:len(turn_change_index)] = 1.0
        tc_mask_t = torch.tensor(tc_mask).to(self._device)
        return tc_mask_t

    def forward(self, embedded_text_input, sep_id, mask, label):
        # bs=1 (bs,sep_len,dim)
        sep_select_embedded_ = torch.index_select(embedded_text_input, -2, sep_id.long())  # .int()

        pre_turn_sep_id = torch.cat((torch.tensor([0]).to(self._device), sep_id[:-1]), 0).long()
        pre_turn_select_embedded = torch.index_select(embedded_text_input, -2, pre_turn_sep_id)  # .int()
        pre_turn_select_embedded[:, 0, :] = 0.0  #对第一轮来说,前一轮为0

        #[sep_len,  sep_prev_len, label_size]


        diff_embed = sep_select_embedded_ - pre_turn_select_embedded
        pointwise_embed = sep_select_embedded_ * pre_turn_select_embedded

        comprehensive_embed = torch.cat((sep_select_embedded_, pre_turn_select_embedded, diff_embed, pointwise_embed),
                                        -1)

        if self.dropout:
            comprehensive_embed = self.dropout(comprehensive_embed)

        # (bs, sep_len, dim) -->  (sep_len, switch_label_size)
        logits = torch.squeeze(self._classification_layer(comprehensive_embed))

        # apply sep_select_embedded_mask=(bs,sep_len)
        mask_logits = torch.unsqueeze(mask > 0, -1).expand_as(logits)

        logits_m = torch.masked_select(logits, mask_logits)
        label_m = torch.masked_select(label, mask_logits)

        logits_m = torch.clamp(logits_m, min=1e-4, max=1 - 1e-4)

        output_dict = {"comprehensive_sep_embed": comprehensive_embed, "turn_sep_mask": mask,"diff_embed":diff_embed,"pointwise_embed":pointwise_embed}

        if label is not None and logits_m.shape[0] != 0:
            loss = self._loss(logits_m, label_m.float())
            output_dict['loss'] = loss * self.loss_scalar

            # self.counter += 1
            # if self.counter % self.report_loss_every_n == 0:
            #     print("loss **********************************AAAA***:"+str(output_dict['loss']))
                #print("logits turn switch: " + str(logits))

        else:
            output_dict['loss'] = 0.0

        return output_dict

@registry.register('multitask', 'turn-switch-classifier-interact-skip')
class TurnSwitchClassifierInteractSkip(nn.Module):

    def __init__(self, input_dim, hidden_dim, vocab_path, dropout=0.1, device=None, leaky_rate=0.2, loss_scalar=4, mid_layer_activator='relu',max_turn_len=6, report_loss_every_n=2):
        super(TurnSwitchClassifierInteractSkip, self).__init__()
        self.turn_len = max_turn_len
        self._device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  #用不着

        self._loss = nn.BCELoss() #替换掉 nn.BCEWithLogitsLoss()

        self.vocab = Vocab.load(vocab_path)
        self.switch_label_size = len(self.vocab)

        self.counter = 0

        self.loss_scalar = loss_scalar

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.report_loss_every_n = report_loss_every_n
        print("mid_layer_activator:{}".format(mid_layer_activator))
        self._classification_layer = torch.nn.Sequential(
            nn.Linear(self.input_dim * 4, self.hidden_dim),
            nn.LeakyReLU(leaky_rate) if mid_layer_activator == 'relu' else nn.Tanh(),
            nn.Linear(self.hidden_dim, self.switch_label_size),
            nn.Sigmoid()
        )

    def create_label_t(self, desc):
        turn_change_index = desc['turn_change_index']
        prev_sep_id = desc['prev_sep_id']

        lable_tc = np.zeros((self.turn_len, self.switch_label_size))

        for sep_idx, index_list in enumerate(turn_change_index):
            lable_tc[sep_idx+1, index_list] = 1.0

        lable_tc_t = torch.tensor(lable_tc).to(self._device)
        return lable_tc_t

    def create_mask_t(self, desc):
        # prev_sep_id = desc['prev_sep_id']
        # tc_mask = np.zeros(self.turn_len)
        # tc_mask[prev_sep_id] = 1.0
        # tc_mask_t = torch.tensor(tc_mask).to(self._device)
        turn_change_index = desc['turn_change_index']
        tc_mask = np.zeros(self.turn_len)
        tc_mask[1:len(turn_change_index)+1] = 1.0
        for sep_idx, index_list in enumerate(turn_change_index):
            if len(index_list) == 0:
                tc_mask[sep_idx+1] = 0.0

        tc_mask_t = torch.tensor(tc_mask).to(self._device)
        return tc_mask_t

    def forward(self, embedded_text_input, sep_id, mask, label):
        # bs=1 (bs,sep_len,dim)
        sep_select_embedded_ = torch.index_select(embedded_text_input, -2, sep_id.long())  # .int()

        pre_turn_sep_id = torch.cat((torch.tensor([0]).to(self._device), sep_id[:-1]), 0).long()
        pre_turn_select_embedded = torch.index_select(embedded_text_input, -2, pre_turn_sep_id)  # .int()
        pre_turn_select_embedded[:, 0, :] = 0.0  #对第一轮来说,前一轮为0

        #[sep_len,  sep_prev_len, label_size]

        diff_embed = sep_select_embedded_ - pre_turn_select_embedded
        pointwise_embed = sep_select_embedded_ * pre_turn_select_embedded

        comprehensive_embed = torch.cat((sep_select_embedded_, pre_turn_select_embedded, diff_embed, pointwise_embed),
                                        -1)

        if self.dropout:
            comprehensive_embed = self.dropout(comprehensive_embed)

        # (bs, sep_len, dim) -->  (sep_len, switch_label_size)
        logits = torch.squeeze(self._classification_layer(comprehensive_embed))

        # apply sep_select_embedded_mask=(bs,sep_len)
        mask_logits = torch.unsqueeze(mask > 0, -1).expand_as(logits)

        logits_m = torch.masked_select(logits, mask_logits)
        label_m = torch.masked_select(label, mask_logits)

        logits_m = torch.clamp(logits_m, min=1e-4, max=1 - 1e-4)

        output_dict = {"comprehensive_sep_embed": comprehensive_embed, "turn_sep_mask": mask,"diff_embed":diff_embed,"pointwise_embed":pointwise_embed}

        if label is not None and logits_m.shape[0] != 0:
            loss = self._loss(logits_m, label_m.float())
            output_dict['loss'] = loss * self.loss_scalar

            # self.counter += 1
            # if self.counter % self.report_loss_every_n == 0:
            #     print("loss **********************************AAAA***:"+str(output_dict['loss']))
                #logger.log('loss **********************************AAAA***: {}'.format(str(output_dict['loss'])))
                #print("logits turn switch: " + str(logits))

        else:
            output_dict['loss'] = 0.0

        return output_dict

@registry.register('multitask', 'turn-switch-classifier-free-interact')
class TurnSwitchClassifierFreeInteract(nn.Module):

    def __init__(self, input_dim, hidden_dim, vocab_path, dropout=0.1, device=None, leaky_rate=0.2, loss_scalar=4,
                 mid_layer_activator='relu', max_turn_len=6, report_loss_every_n=2):
        super(TurnSwitchClassifierFreeInteract, self).__init__()
        self.turn_len = max_turn_len
        self._device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # 用不着

        self._loss = nn.BCELoss()  # 替换掉 nn.BCEWithLogitsLoss()

        self.vocab = Vocab.load(vocab_path)
        self.switch_label_size = len(self.vocab)

        self.counter = 0

        self.loss_scalar = loss_scalar

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.report_loss_every_n = report_loss_every_n
        print("mid_layer_activator:{}".format(mid_layer_activator))
        self._classification_layer = torch.nn.Sequential(
            nn.Linear(self.input_dim * 4, self.hidden_dim),
            nn.LeakyReLU(leaky_rate) if mid_layer_activator == 'relu' else nn.Tanh(),
            nn.Linear(self.hidden_dim, self.switch_label_size),
            nn.Sigmoid()
        )

    def create_label_t(self, desc):
        turn_change_index = desc['turn_change_index']
        lable_tc = np.zeros((self.turn_len, self.switch_label_size))
        for i, tc in enumerate(turn_change_index):
            for item in tc:
                lable_tc[i, item] = 1.0

        lable_tc_t = torch.tensor(lable_tc).to(self._device)
        return lable_tc_t

    def create_mask_t(self, turn_change_index):
        tc_mask = np.zeros(self.turn_len)
        tc_mask[1:len(turn_change_index)] = 1.0
        tc_mask_t = torch.tensor(tc_mask).to(self._device)
        return tc_mask_t

    def forward(self, embedded_text_input, sep_id, prev_sep_id, mask, label):
        # prev_sep_id=[sep_prev_len]  label=[sep_prev_len,label_size] mask = [sep_prev_len, label_size]
        sep_select_embedded_ = torch.index_select(embedded_text_input, -2, sep_id.long())  # .int()

        pre_turn_sep_id = torch.cat((torch.tensor([0]).to(self._device), sep_id[:-1]), 0).long()
        pre_turn_select_embedded = torch.index_select(embedded_text_input, -2, pre_turn_sep_id)  # .int()
        pre_turn_select_embedded[:, 0, :] = 0.0  # 对第一轮来说,前一轮为0

        # [sep_len,  sep_prev_len, label_size]

        diff_embed = sep_select_embedded_ - pre_turn_select_embedded
        pointwise_embed = sep_select_embedded_ * pre_turn_select_embedded

        comprehensive_embed = torch.cat((sep_select_embedded_, pre_turn_select_embedded, diff_embed, pointwise_embed),
                                        -1)

        if self.dropout:
            comprehensive_embed = self.dropout(comprehensive_embed)

        # (bs, sep_len, dim) -->  (sep_len, switch_label_size)
        logits = torch.squeeze(self._classification_layer(comprehensive_embed))

        # apply sep_select_embedded_mask=(bs,sep_len)
        mask_logits = torch.unsqueeze(mask > 0, -1).expand_as(logits)

        logits_m = torch.masked_select(logits, mask_logits)
        label_m = torch.masked_select(label, mask_logits)

        logits_m = torch.clamp(logits_m, min=1e-4, max=1 - 1e-4)

        output_dict = {"comprehensive_sep_embed": comprehensive_embed, "turn_sep_mask": mask, "diff_embed": diff_embed,
                       "pointwise_embed": pointwise_embed}

        if label is not None and logits_m.shape[0] != 0:
            loss = self._loss(logits_m, label_m.float())
            output_dict['loss'] = loss * self.loss_scalar

            # if torch.isnan(loss):
            #     print("here!")
            #     print(
            #         "A lap in this 'sky pool' may have you holding your breath, and not just because you're underwater."
            #         " With the streets of London looming 10 stories down, "
            #         "the view through the pool's clear bottom is a bit freaky to all but the fearless. ")
            #     print(logits_m)
            #     print("=====label_m=====")
            #     print(label_m)
            #     print("=====logits_m.sigmoid()=====")
            #     print(logits_m.sigmoid())
            #     print("==========")

            # self.counter += 1
            # if self.counter % self.report_loss_every_n == 0:
            #     print("loss **********************************AAAA***:" + str(output_dict['loss']))
                # print("logits turn switch: " + str(logits))

        else:
            output_dict['loss'] = 0.0

        return output_dict


# zhanghanchu
@registry.register('multitask', 'turn-switch-col-classifier')
class TurnSwitchColClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_path, dropout=0.1, device=None, leaky_rate=0.2, loss_scalar=8, report_loss_every_n=20):
        super(TurnSwitchColClassifier, self).__init__()
        self._device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self._loss = nn.BCELoss()  # 替换掉 nn.BCEWithLogitsLoss()

        self.vocab = Vocab.load(vocab_path)
        self.switch_label_size = len(self.vocab)

        self.counter = 0

        self.loss_scalar = loss_scalar

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self.report_loss_every_n = report_loss_every_n

        self._classification_layer = torch.nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.switch_label_size),
            nn.Sigmoid()
        )

    def create_label_t(self, turn_change_col_index, col_tab_len):
        # id 是第几个列
        # "dict{1:[2,4,7],2:[],3:[1,5,9]}"
        # [[1,0],[ 2,2]
        lable_tc = np.zeros((col_tab_len, self.switch_label_size))

        for id_, label_list in turn_change_col_index.items():
            lable_tc[[int(id_) for _ in label_list], label_list] = 1.0

        lable_tc_t = torch.tensor(lable_tc).to(self._device)
        return lable_tc_t

    def forward(self, embedded_col_tab_input, label):
        # bs=1 (bs,sep_len,dim)

        if self.dropout:
            embedded_col_tab_input = self.dropout(embedded_col_tab_input)

        # (bs, sep_len, dim) -->  (sep_len, switch_label_size)
        logits = torch.squeeze(self._classification_layer(embedded_col_tab_input))

        logits_m = torch.clamp(logits, min=1e-4, max=1 - 1e-4)

        output_dict = {}

        if label is not None:
            loss = self._loss(logits_m, label.float())
            output_dict['loss'] = loss * self.loss_scalar

            # self.counter += 1
            # if self.counter % self.report_loss_every_n == 0:
            #     print("loss BBBB: " + str(output_dict['loss']))
                #training_logger_global.training_logger_global_transformers.logger.log('loss BBBB: {}'.format(str(output_dict['loss'])))

        else:
            output_dict['loss'] = 0.0

        return output_dict
# Adapted from The Annotated Transformer
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# liyutian
# Adapted from The Annotated Transformer
class HistorySublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(HistorySublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        y, loss = sublayer(self.norm(x))
        return x + self.dropout(y), loss


# Adapted from The Annotated Transformer
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, num_relation_kinds, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(lambda: SublayerConnection(size, dropout), 2)
        self.size = size

        self.relation_k_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)
        self.relation_v_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)

    def forward(self, x, relation, mask):
        "Follow Figure 1 (left) for connections."
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask))
        return self.sublayer[1](x, self.feed_forward)


# Adapted from The Annotated Transformer
class HistoryEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, num_relation_kinds, dropout):
        super(HistoryEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones2(lambda: HistorySublayerConnection(size, dropout),
                                lambda: SublayerConnection(size, dropout))
        self.size = size

        self.relation_k_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)
        self.relation_v_emb = nn.Embedding(num_relation_kinds, self.self_attn.d_k)

    def forward(self, x, relation, mask, sep_id, history_reg):
        "Follow Figure 1 (left) for connections."
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)

        x, loss = self.sublayer[0](x,
                                   lambda x: self.self_attn(x, x, x, relation_k, relation_v, mask, sep_id, history_reg))
        return self.sublayer[1](x, self.feed_forward), loss


# Adapted from The Annotated Transformer
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

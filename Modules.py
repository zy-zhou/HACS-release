# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:14:09 2019

@author: Zhou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from Utils import sequence_mask, tuple_map
from Data import pad

word_embed = 256
lstm_units = 256
num_layers = 2
bidirectional = True
dropout = 0.3
glob_attn = 'mul'

class IntraAttention(nn.Module):
    ''' Intra-statement attention to obtain statement vectors. '''
    def __init__(self, units):
        super(IntraAttention, self).__init__()
        self.W = nn.Linear(units, units)
        self.u = nn.Parameter(torch.rand(1, 1, units))
    
    def score(self, value):
        key = torch.tanh(self.W(value)) # num_statm_total * max_steps * units
        u = self.u.expand(value.shape[0], 1, value.shape[2])
        output = torch.bmm(u, key.transpose(1, 2)).squeeze(1)
        return output
    
    def forward(self, value, memory_lengths):
        score = self.score(value)
        if memory_lengths is not None:
            score_mask = sequence_mask(memory_lengths, value.shape[1])
            score.masked_fill_(~score_mask, float('-inf'))
        
        alignments = F.softmax(score, 1)
        context = torch.bmm(alignments.unsqueeze(1), value) # batch_size * 1 * units
        context = context.squeeze(1)
        return context, alignments

class GlobalAttention(nn.Module):
    ''' Multiplicative and additive global attention.
        shape of query: [batch_size, units]
        shape of key: [batch_size, max_steps, key_dim]
        shape of context: [batch_size, units]
        shape of alignments: [batch_size, max_steps]
        style should be either "add" or "mul"'''
    def __init__(self, units, key_dim=None, style='mul', scale=True):
        super(GlobalAttention, self).__init__()
        self.style = style
        self.scale = scale
        key_dim = key_dim or units
            
        self.Wk = nn.Linear(key_dim, units, bias=False)
        if self.style == 'mul':
            if self.scale:
                self.v = nn.Parameter(torch.tensor(1.))
        elif self.style == 'add':
            self.Wq = nn.Linear(units, units)
            self.v = nn.Parameter(torch.ones(units))
        else:
            raise ValueError(str(style) + ' is not an appropriate attention style.')
            
    def score(self, query, key):
        query = query.unsqueeze(1) # batch_size * 1 * units
        key = self.Wk(key)
        
        if self.style == 'mul':
            output = torch.bmm(query, key.transpose(1, 2))
            output = output.squeeze(1)
            if self.scale:
                output = self.v * output
        else:
            output = torch.sum(self.v * torch.tanh(self.Wq(query) + key), 2)
        return output
    
    def forward(self, query, key, memory_lengths=None, custom_mask=None):
        score = self.score(query, key) # batch_size * max_steps
        if memory_lengths is not None:
            if type(memory_lengths) in {list, tuple}:
                memory_lengths = memory_lengths[0]
            score_mask = sequence_mask(memory_lengths, key.shape[1])
            score.masked_fill_(~score_mask, float('-inf'))
        elif custom_mask is not None:
            score.masked_fill_(~custom_mask, float('-inf'))
        
        alignments = F.softmax(score, 1)
        context = torch.bmm(alignments.unsqueeze(1), key) # batch_size * 1 * units
        context = context.squeeze(1)
        return context, alignments

class RNNEncoder(nn.Module):
    def __init__(self, field=None, embed_dim=word_embed, units=lstm_units, in_dim=None,
                 num_layers=num_layers, bidirectional=bidirectional, dropout=dropout):
        super(RNNEncoder, self).__init__()
        if field is not None:
            vocab_size = len(field.vocab)
            pad_id = field.vocab.stoi[pad]
            self.field = field
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
            self.lstm = nn.LSTM(embed_dim, units, num_layers, batch_first=True,
                                bidirectional=bidirectional)
        else:
            self.field = None
            self.lstm = nn.LSTM(in_dim, units, num_layers, batch_first=True,
                                dropout=dropout, bidirectional=bidirectional)
        
        self.units = units
        self.num_layers = num_layers
        self.directions = int(bidirectional) + 1
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, src_lengths=None):
        if self.field is not None:
            inputs = self.embedding(inputs)
        x = self.dropout(inputs)
        if src_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                    x, src_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        output_seqs, states = self.lstm(x)
        
        if self.directions > 1:
            final_idx = torch.arange(self.num_layers, device=inputs.device) * self.directions + 1
            states = tuple_map(lambda x: x.index_select(0, final_idx), states)
        
        if src_lengths is not None:
            output_seqs, _ = nn.utils.rnn.pad_packed_sequence(output_seqs, batch_first=True)
        return output_seqs, states

class DecoderCellState(
        namedtuple('DecoderCellState',
                   ('context', 'state', 'alignments'), defaults=[None])):
    def batch_select(self, indices):
        select = lambda x, dim=0: x.index_select(dim, indices)
        return self._replace(context = select(self.context),
                             state = tuple_map(select, self.state, dim=1),
                             alignments = tuple_map(select, self.alignments))

class DecoderCell(nn.Module):
    def __init__(self, embed_dim, units=lstm_units, num_layers=num_layers, dropout=dropout,
                 glob_attn=glob_attn, memory_dim=None, input_feed=True, use_attn_layer=True):
        super(DecoderCell, self).__init__()
        self.glob_attn = glob_attn
        self.input_feed = glob_attn and input_feed
        self.use_attn_layer = glob_attn and use_attn_layer
        self.dropout = nn.Dropout(dropout)
        
        if memory_dim is None:
            memory_dim = units
        cell_in_dim, context_dim = embed_dim, memory_dim
        if glob_attn is not None:
            self.attention = GlobalAttention(units, memory_dim, glob_attn)
            
            if use_attn_layer:
                self.attn_layer = nn.Linear(context_dim + units, units)
                context_dim = units
            if input_feed:
                cell_in_dim += context_dim
        self.cell = nn.LSTM(cell_in_dim, units, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=False)
            
    def forward(self, tgt_embed, prev_state, memory=None, src_lengths=None):
#        perform one decoding step
        cell_input = self.dropout(tgt_embed) # batch_size * embed_size
        
        if self.glob_attn is not None:
            if self.input_feed:
                cell_input = torch.cat([cell_input, prev_state.context], 1).unsqueeze(1)
                # batch_size * 1 * (embed_size + units)
            output, state = self.cell(cell_input, prev_state.state)
            output = output.squeeze(1) # batch_size * units
            context, alignments = self.attention(output, memory, src_lengths)
            if self.use_attn_layer:
                context = torch.cat([context, output], 1)
                context = torch.tanh(self.attn_layer(context))
            context = self.dropout(context)
            return DecoderCellState(context, state, alignments)
        else:
            output, state = self.cell(cell_input, prev_state.state)
            output = output.squeeze(1)
            return DecoderCellState(output, state)

class BasicDecoder(nn.Module):
    def __init__(self, field, embed_dim=word_embed, units=lstm_units, num_layers=num_layers,
                 dropout=dropout, glob_attn=glob_attn, **kwargs):
        ' If hybrid, memory, memory_dim and src_lengths should be tuples. '
        super(BasicDecoder, self).__init__()
        vocab_size = len(field.vocab)
        pad_id = field.vocab.stoi[pad]
        self.field = field
        self.units = units
        self.glob_attn = glob_attn
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.out_layer = nn.Linear(units, vocab_size, bias=False)
        self.cell = DecoderCell(embed_dim, units, num_layers, dropout,
                                glob_attn, **kwargs)
        
    @property
    def attn_history(self):
        if self.cell.hybrid_mode:
            return self.cell.attention.attn_history
        elif self.glob_attn:
            return 'std'
    
    def initialize(self, enc_final):
        init_context = torch.zeros(enc_final[0].shape[1], self.units,
                                   device=enc_final[0].device)
        return DecoderCellState(init_context, enc_final)
    
    def forward(self, tgt_inputs, enc_final, memory=None, src_lengths=None, return_contxt=False):
        tgt_embeds = self.embedding(tgt_inputs) # batch_size * max_steps * units
        prev_state = self.initialize(enc_final)
        output_seqs, attn_history = [], []
        
        for tgt_embed in tgt_embeds.split(1, 1):
            state = self.cell(tgt_embed.squeeze(1), prev_state, memory, src_lengths)
            output_seqs.append(state.context)
            if self.glob_attn is not None:
                attn_history.append(state.alignments)
            prev_state = state
        
        output_seqs = torch.stack(output_seqs, 1)
        logits = self.out_layer(output_seqs)
        if self.glob_attn is not None:
            attn_history = torch.stack(attn_history, 1)
        if return_contxt:
            return logits, output_seqs, attn_history
        else:
            return logits, attn_history
        
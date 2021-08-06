# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:35:43 2019

@author: Zhou
"""

from Modules import IntraAttention
from DecoderWrappers import SampleDecodingWrapper, BeamSearchWrapper,\
                            EnsembleSamplDecWrapper, EnsembleBmSrchWrapper
from Utils import save, sequence_loss, batch_bleu, batch_meteor, batch_rouge,\
                  perplexity
from Data import id2word
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

beam_width = 4
n_best = 1
sampling_temp = 1.
sampling_topk = -1
max_iter = 30
length_penalty = 0.4
coverage_penalty = 0.

class HierarchialEncoder(nn.Module):
    def __init__(self, statm_encoder, token_encoder, intra_attn='token', cat_contxt=True):
        super(HierarchialEncoder, self).__init__()
        self.statm_encoder = statm_encoder
        self.token_encoder = token_encoder
        self.cat_contxt = cat_contxt
        self.intra_attn = intra_attn
        if intra_attn:
            units = token_encoder.units * token_encoder.directions
            self.attention = IntraAttention(units)
            if intra_attn == 'both':
                units = statm_encoder.units * statm_encoder.directions
                self.statm_attention = IntraAttention(units)
    
    def forward(self, batches, src_lengths):
        num_statm, statm_lengths = src_lengths
        token_memory, token_final = self.token_encoder(batches, statm_lengths)
        if self.intra_attn:
            token_final, attn_history = self.attention(token_memory, statm_lengths)
            # self.attn_history = attn_history
        # num_statm_total * token_units
        chunks = torch.split(token_final, num_statm.tolist())
        token_final = pad_sequence(chunks, batch_first=True) # batch_size * max_statm * token_units
        statm_memory, statm_final = self.statm_encoder(token_final, num_statm)
        
        if self.intra_attn == 'both':
            c_final = statm_final[-1]
            h_final, _ = self.statm_attention(statm_memory, num_statm)
            h_final = h_final[:, -self.statm_encoder.units:]
            statm_final = (h_final, c_final)
        if self.cat_contxt:
            statm_memory = torch.cat([statm_memory, token_final], -1)
        return statm_memory, statm_final

class EnsembleEncoder(nn.Module):
    def __init__(self, encoders):
        super(EnsembleEncoder, self).__init__()
        self.encoders = nn.ModuleList(encoders)
    
    def forward(self, batches, src_lengths):
        outputs = [encoder(batch, length) for encoder, batch, length in zip(
                                    self.encoders, batches, src_lengths)]
        memories, final_states = zip(*outputs)
        return memories, final_states

class EnsembleDecoder(nn.Module):
    def __init__(self, decoders):
        super(EnsembleDecoder, self).__init__()
        self.decoders = nn.ModuleList(decoders)
        self.field = decoders[0].field
        self.glob_attn = decoders[0].glob_attn
    
    def forward(self, tgt_batch, final_states, memories, src_lengths):
        outputs = [decoder(tgt_batch, final_state, memory, length) for \
                   decoder, final_state, memory, length in zip(
                           self.decoders, final_states, memories, src_lengths)]
        logits, attn_history = zip(*outputs)
        probs = torch.softmax(torch.stack(logits), -1) # num_models * batch_size * num_words
        probs = probs.mean(0)
        return probs, attn_history

class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        if type(encoder) in {list, tuple}:
            assert type(decoder) in {list, tuple} and len(encoder) == len(decoder)
            encoder = EnsembleEncoder(encoder)
            decoder = EnsembleDecoder(decoder)
            self.is_ensemble = True
        else:
            assert type(decoder) not in {list, tuple}
            self.is_ensemble = False
        self.encoder = encoder
        self.decoder = decoder
    
    @classmethod
    def ensemble(cls, *models):
        encoders = [model.encoder for model in models]
        decoders = [model.decoder for model in models]
        return cls(encoders, decoders)
    
    def forward(self, src_batch, src_lengths=None, tgt_batch=None):
        tgt_batch = tgt_batch[:,:-1]
        memory, final_state = self.encoder(src_batch, src_lengths)
        logits, attn_history = self.decoder(tgt_batch, final_state, memory, src_lengths)
        return logits, attn_history

class Translator(object):
    def __init__(self, model, sampling_temp=sampling_temp, sampling_topk=sampling_topk,
                 beam_width=beam_width, n_best=n_best, max_iter=max_iter,
                 length_penalty=length_penalty, coverage_penalty=coverage_penalty,
                 metrics=['loss', 'bleu'], unk_replace=False, smooth=3):
        self.model = model
        self.metrics = metrics
        self.unk_replace = unk_replace
        self.smooth = smooth
        
        if not beam_width or beam_width == 1:
            if model.is_ensemble:
                self.wrapped_decoder = EnsembleSamplDecWrapper(
                        model.decoder, sampling_topk, max_iter)
            else:
                self.wrapped_decoder = SampleDecodingWrapper(
                        model.decoder, sampling_temp, sampling_topk, max_iter)
        else:
            if model.is_ensemble:
                self.wrapped_decoder = EnsembleBmSrchWrapper(
                        model.decoder, beam_width, n_best, max_iter, length_penalty, coverage_penalty)
            else:
                self.wrapped_decoder = BeamSearchWrapper(
                        model.decoder, beam_width, n_best, max_iter, length_penalty, coverage_penalty)

    @property
    def metrics(self):
        return self._metrics
    
    @metrics.setter
    def metrics(self, metrics):
        metrics = set(metrics)
        all_metrics = {'loss', 'bleu', 'rouge', 'meteor'}
        if not metrics.issubset(all_metrics):
            raise ValueError('Unkown metric(s): ' + str(metrics.difference(all_metrics)))
        self._metrics = metrics
    
    def val_loss(self, final_state, memory, src_lengths, tgt_batch):
        outputs, _ = self.model.decoder(tgt_batch[:,:-1], final_state, memory, src_lengths)
        loss = sequence_loss(outputs, tgt_batch[:,1:], is_probs=self.model.is_ensemble,
                             pad_id=self.wrapped_decoder.pad_id)
        return loss.item()
    
    def translate_batch(self, src_batch, src_lengths=None, tgt_batch=None, raw_batches=[None]):
        reports = dict(scores=None, attn_history=None)
        with torch.no_grad():
            memory, final_state = self.model.encoder(src_batch, src_lengths)
            if 'loss' in self._metrics:
                reports['loss'] = self.val_loss(final_state, memory, src_lengths, tgt_batch)
            predicts, reports['scores'], reports['attn_history'] = \
            self.wrapped_decoder(final_state, memory, src_lengths)
        
        if type(self.wrapped_decoder) in {BeamSearchWrapper, EnsembleBmSrchWrapper}:
            predicts = [b[0] for b in predicts]
            reports['scores'] = [b[0] for b in reports['scores']]
            if reports['attn_history'][0]:
                reports['attn_history'] = [b[0] for b in reports['attn_history']]
        
        predicts = id2word(predicts, self.model.decoder.field,
                           (raw_batches[0], reports['attn_history']), 
                           replace_unk=self.unk_replace)
        if 'bleu' in self._metrics:
            reports['bleu'] = batch_bleu(predicts, raw_batches[-1], self.smooth) * 100
        predicts = [' '.join(s) for s in predicts]
        
        if not self._metrics.isdisjoint({'rouge', 'meteor'}):
            targets = [' '.join(s) for s in raw_batches[-1]]
            if 'rouge' in self._metrics:
                rouge = batch_rouge(predicts, targets)
                reports['rouge'] = rouge['rouge-l']['f'] * 100
            if 'meteor' in self._metrics:
                reports['meteor'] = batch_meteor(predicts, targets) * 100
        
        return predicts, reports
    
    def init_generator(self, data_gen):
        data_gen.raw_data = self.unk_replace or self._metrics.difference({'loss'})
        self.raw_data = data_gen.raw_data
    
    def __call__(self, batches, save_path=None):
        self.init_generator(batches)
        self.model.eval()
        results = []
        reports = defaultdict(float, scores=[], attn_history=[])
        
        pbar = tqdm(batches, desc='Translating...')
        for batch in pbar:
            predicts, reports_ = self.translate_batch(*batch)
            pbar.set_postfix({metric: reports_[metric] for metric in self._metrics})
            results.extend(predicts)
            for metric in self._metrics:
                reports[metric] += reports_[metric]
#            reports['scores'].extend(reports_['scores'])
            # reports['attn_history'].extend(reports_['attn_history'])
        
        for metric in self._metrics:
            reports[metric] /= len(batches)
            print('total {}: {:.2f}'.format(metric, reports[metric]))
            if metric == 'loss':
                reports['ppl'] = perplexity(reports[metric])
                print('total ppl: {:.2f}'.format(reports['ppl']))
        if save_path is not None:
            save(results, save_path)
        return results, reports

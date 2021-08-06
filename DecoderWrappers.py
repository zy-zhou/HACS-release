# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:55:33 2020

@author: Zhou
"""

import torch
import torch.nn.functional as F
from Utils import sequence_mask, sample_with_temp, sample_with_probs, tile, tuple_map
from BeamSearch import BeamSearch
from Data import pad, bos, eos

class SampleDecodingWrapper(object):
    def __init__(self, decoder, sampling_temp, sampling_topk, max_iter, trunc_output=True):
        self.decoder = decoder
        self.pad_id = decoder.field.vocab.stoi[pad]
        self.bos_id = decoder.field.vocab.stoi[bos]
        self.eos_id = decoder.field.vocab.stoi[eos]
        self.sampling_temp = sampling_temp
        self.sampling_topk = sampling_topk
        self.max_iter = max_iter
        self.trunc_output = trunc_output
    
    def initialize(self, enc_final):
        init_state = self.decoder.initialize(enc_final)
        bos_ids = torch.full([enc_final[0].shape[1]], self.bos_id, dtype=torch.long,
                             device=enc_final[0].device)
        self.finished = torch.zeros_like(bos_ids, dtype=torch.bool)
        self.lengths = torch.ones_like(bos_ids, dtype=torch.int) # include EOS
        return bos_ids, init_state
    
    def update(self, step_logits):
        predicts, scores = sample_with_temp(step_logits, self.sampling_temp, self.sampling_topk)
        predicts, scores = predicts.squeeze(1), scores.squeeze(1)
        self.finished |= predicts.eq(self.eos_id)
        self.lengths += (~self.finished).int()
        return predicts, scores
    
    def finalize(self, outputs, scores, attn_history):
        outputs, scores = torch.stack(outputs, 1), torch.stack(scores, 1)
        if self.trunc_output:
            indices = range(outputs.shape[0])
            outputs = [outputs[i,:self.lengths[i]] for i in indices]
#            scores = [scores[i,:self.lengths[i]] for i in indices]
            if self.decoder.glob_attn is not None:
                attn_history = torch.stack(attn_history, 1)
                attn_history = [attn_history[i,:self.lengths[i],...] for i in indices]
        else:
            mask = sequence_mask(self.lengths, outputs.shape[1])
            outputs = outputs.masked_fill(~mask, self.pad_id)
        return outputs, scores, attn_history
    
    def __call__(self, enc_final, memory=None, src_lengths=None):
        prev_predicts, prev_state = self.initialize(enc_final)
        outputs, scores, attn_history = [], [], []
        
        for step in range(self.max_iter):
            embed = self.decoder.embedding(prev_predicts)
            state = self.decoder.cell(embed, prev_state, memory, src_lengths)
            logits = self.decoder.out_layer(state.context)
            predicts, sampl_scores = self.update(logits)
            outputs.append(predicts)
            scores.append(sampl_scores)
            if self.decoder.glob_attn is not None:
                attn_history.append(state.alignments)
            if self.finished.all():
                break
            prev_predicts, prev_state = predicts, state
        
        return self.finalize(outputs, scores, attn_history)

class BeamSearchWrapper(object):
    def __init__(self, decoder, beam_width, n_best, max_iter, length_penalty,
                 coverage_penalty):
        self.decoder = decoder
        self.pad_id = decoder.field.vocab.stoi[pad]
        self.bos_id = decoder.field.vocab.stoi[bos]
        self.eos_id = decoder.field.vocab.stoi[eos]
        self.vocab_size = len(decoder.field.vocab)
        self.beam_width = beam_width
        self.n_best = n_best
        self.max_iter = max_iter
        self.length_penalty = length_penalty
        self.coverage_penalty = decoder.glob_attn and coverage_penalty
        
    def initialize(self, enc_final, memory, src_lengths):
        if type(src_lengths) in {tuple, list}:
            memory_lengths = src_lengths[0]
        else:
            memory_lengths = src_lengths
        self.beam = BeamSearch(
                beam_size=self.beam_width,
                batch_size=enc_final[0].shape[1],
                bos=self.bos_id,
                eos=self.eos_id,
                n_best=self.n_best,
                device=enc_final[0].device,
                max_length=self.max_iter,
                return_attention=bool(self.decoder.glob_attn),
                memory_lengths=memory_lengths,
                stepwise_penalty=self.coverage_penalty,
                length_penalty=self.length_penalty,
                ratio=0.)
        
        tile_ = lambda x, dim=0: tile(x, self.beam_width, dim)
        enc_final = tuple_map(tile_, enc_final, dim=1)
        if self.decoder.glob_attn is not None:
            memory = tuple_map(tile_, memory)
            src_lengths = tuple_map(tile_, src_lengths)
        init_state = self.decoder.initialize(enc_final)
        return init_state, memory, src_lengths
    
    def __call__(self, enc_final, memory=None, src_lengths=None):
        prev_state, memory, src_lengths = self.initialize(enc_final, memory, src_lengths)
        
        for step in range(self.max_iter):
            embed = self.decoder.embedding(self.beam.current_predictions)
            state = self.decoder.cell(embed, prev_state, memory, src_lengths)
            log_probs = F.log_softmax(self.decoder.out_layer(state.context), -1)
            alignments = state.alignments.unsqueeze(0) if self.decoder.glob_attn else None
            self.beam.advance(log_probs, alignments)
            
            any_beam_finished = self.beam.is_finished.any()
            if any_beam_finished:
                self.beam.update_finished()
                if self.beam.done:
                    break
            select = lambda x: x.index_select(0, self.beam.current_origin)
            if any_beam_finished and self.decoder.glob_attn is not None:
                memory = tuple_map(select, memory)
                src_lengths = tuple_map(select, src_lengths)
            prev_state = state.batch_select(self.beam.current_origin)
        
        return self.beam.predictions, self.beam.scores, self.beam.attention

class EnsembleSamplDecWrapper(object):
    def __init__(self, ensemble_decoder, sampling_topk, max_iter, trunc_output=True):
        self.decoders = ensemble_decoder.decoders
        self.pad_id = ensemble_decoder.field.vocab.stoi[pad]
        self.bos_id = ensemble_decoder.field.vocab.stoi[bos]
        self.eos_id = ensemble_decoder.field.vocab.stoi[eos]
        self.sampling_topk = sampling_topk
        self.max_iter = max_iter
        self.trunc_output = trunc_output
    
    def initialize(self, enc_finals):
        init_states = [decoder.initialize(enc_final)
                      for decoder, enc_final in zip(self.decoders, enc_finals)]
        bos_ids = torch.full([enc_finals[0][0].shape[1]], self.bos_id, dtype=torch.long,
                             device=enc_finals[0][0].device)
        self.finished = torch.zeros_like(bos_ids, dtype=torch.bool)
        self.lengths = torch.ones_like(bos_ids, dtype=torch.int) # include EOS
        return bos_ids, init_states
    
    def update(self, step_probs, states):
        step_probs = torch.stack(step_probs).mean(0)
        predicts, scores = sample_with_probs(step_probs, self.sampling_topk)
        predicts, scores = predicts.squeeze(1), scores.squeeze(1)
        self.finished |= predicts.eq(self.eos_id)
        self.lengths += (~self.finished).int()
        return predicts, scores
    
    def finalize(self, outputs, scores, attn_history):
        outputs, scores = torch.stack(outputs, 1), torch.stack(scores, 1)
        if self.trunc_output:
            indices = range(outputs.shape[0])
            outputs = [outputs[i,:self.lengths[i]] for i in indices]
#            scores = [scores[i,:self.lengths[i]] for i in indices]
            if self.decoders[0].glob_attn is not None:
                attn_history = zip(*attn_history)
                attn_history = [torch.stack(x, 1) for x in attn_history]
                attn_history = [[x[i,:self.lengths[i],...] for i in indices] for x in attn_history]
        else:
            mask = sequence_mask(self.lengths, outputs.shape[1])
            outputs = outputs.masked_fill(~mask, self.pad_id)
        return outputs, scores, attn_history
    
    def __call__(self, enc_finals, memories=None, src_lengths=None):
        prev_predicts, prev_states = self.initialize(enc_finals)
        outputs, scores, attn_history = [], [], []
        
        for step in range(self.max_iter):
            probs, states = [], []
            for decoder, prev_state, memory, length in zip(
                    self.decoders, prev_states, memories, src_lengths):
                embed = decoder.embedding(prev_predicts)
                state = decoder.cell(embed, prev_state, memory, length)
                probs.append(F.softmax(decoder.out_layer(state.context), -1))
                states.append(state)
            predicts, sampl_scores = self.update(probs, states)
            outputs.append(predicts)
            scores.append(sampl_scores)
            if self.decoders[0].glob_attn is not None:
                attn_history.append([state.alignments for state in states])
            if self.finished.all():
                break
            prev_predicts, prev_states = predicts, states
        
        return self.finalize(outputs, scores, attn_history)

class EnsembleBmSrchWrapper(object):
    def __init__(self, ensemble_decoder, beam_width, n_best, max_iter, length_penalty,
                 coverage_penalty):
        self.decoders = ensemble_decoder.decoders
        self.pad_id = ensemble_decoder.field.vocab.stoi[pad]
        self.bos_id = ensemble_decoder.field.vocab.stoi[bos]
        self.eos_id = ensemble_decoder.field.vocab.stoi[eos]
        self.vocab_size = len(ensemble_decoder.field.vocab)
        self.beam_width = beam_width
        self.n_best = n_best
        self.max_iter = max_iter
        self.length_penalty = length_penalty
        self.coverage_penalty = ensemble_decoder.glob_attn and coverage_penalty
        
    def initialize(self, enc_finals, memories, src_lengths):
        if type(src_lengths[0]) in {tuple, list}:
            memory_lengths = src_lengths[0][0]
        else:
            memory_lengths = src_lengths[0]
        self.beam = BeamSearch(
                beam_size=self.beam_width,
                batch_size=enc_finals[0][0].shape[1],
                bos=self.bos_id,
                eos=self.eos_id,
                n_best=self.n_best,
                device=enc_finals[0][0].device,
                max_length=self.max_iter,
                return_attention=bool(self.decoders[0].glob_attn),
                memory_lengths=memory_lengths,
                stepwise_penalty=self.coverage_penalty,
                length_penalty=self.length_penalty,
                ratio=0.)
        
        tile_ = lambda x, dim=0: tile(x, self.beam_width, dim)
        enc_finals = tuple_map(tile_, enc_finals, dim=1)
        if self.decoders[0].glob_attn is not None:
            memories = tuple_map(tile_, memories)
            src_lengths = tuple_map(tile_, src_lengths)
        init_states = [decoder.initialize(enc_final)
                      for decoder, enc_final in zip(self.decoders, enc_finals)]
        return init_states, memories, src_lengths
    
    def __call__(self, enc_finals, memories=None, src_lengths=None):
        prev_states, memories, src_lengths = self.initialize(
            enc_finals, memories, src_lengths)
        
        for step in range(self.max_iter):
            probs, states = [], []
            for decoder, prev_state, memory, length in zip(
                    self.decoders, prev_states, memories, src_lengths):
                embed = decoder.embedding(self.beam.current_predictions)
                state = decoder.cell(embed, prev_state, memory, length)
                probs.append(F.softmax(decoder.out_layer(state.context), -1))
                states.append(state)
            
            log_probs = torch.stack(probs).mean(0).log()
            alignments = states[0].alignments.unsqueeze(0) if self.decoders[0].glob_attn else None
            self.beam.advance(log_probs, alignments)
            
            any_beam_finished = self.beam.is_finished.any()
            if any_beam_finished:
                self.beam.update_finished()
                if self.beam.done:
                    break
            select = lambda x: x.index_select(0, self.beam.current_origin)
            if any_beam_finished and self.decoders[0].glob_attn is not None:
                memories = tuple_map(select, memories)
                src_lengths = tuple_map(select, src_lengths)
            prev_states = [state.batch_select(self.beam.current_origin) for state in states]
        
        return self.beam.predictions, self.beam.scores, self.beam.attention
    
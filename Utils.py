# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 18:51:41 2019

@author: Zhou
"""
import json
from math import exp
import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from multiprocessing import Pool
from time import time
from rouge import Rouge

def parallel(func, data, workers=5, chunksize=None, **kwargs):
    print('Initializing multi-process...')
    begin = time()
    pool = Pool(workers, **kwargs)
    results = pool.map(func, data, chunksize=chunksize)
    pool.close()
    pool.join()
    gap = time() - begin
    print('Done.')
    print('Elapsed time: {} min {:.2f} sec'.format(int(gap // 60), gap % 60))
    return results

def load(path, is_json=False, key=None, drop_list=()):
    print('Loading...')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not is_json:
        if not drop_list:
            return lines
        else:
            return [line for i, line in enumerate(lines) if not i in drop_list]
    
    if key is None:
        return [json.loads(line) for i, line in enumerate(lines) if not i in drop_list]
    else:
        return [json.loads(line)[key] for i, line in enumerate(lines) if not i in drop_list]

def save(data, path, is_json=False):
    print('Saving...')
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            if is_json:
                line = '' if not line else json.dumps(line)
            f.write(line + '\n')

def sequence_mask(lengths, maxlen=None, dtype=None):
    maxlen = maxlen or lengths.max()
    row = torch.arange(maxlen, dtype=lengths.dtype, device=lengths.device)
    col = lengths.unsqueeze(-1)
    result = torch.lt(row, col)
    if dtype is not None:
        result = result.type(dtype)
    return result

def sequence_loss(logits_or_probs=None, targets=None, is_probs=False, pad_id=1, reduction='mean'):
    ''' shape of logits or probs: [batch_size, max_steps, vocab_size]
        shape of targets: [batch_size, max_steps] '''
    targets = targets.reshape(-1)
    outputs = logits_or_probs.view(-1, logits_or_probs.shape[-1])
    if is_probs:
        loss = F.nll_loss(outputs.log(), targets, ignore_index=pad_id, reduction=reduction)
    else:
        loss = F.cross_entropy(outputs, targets, ignore_index=pad_id, reduction=reduction)
    return loss

def sample_with_temp(logits, sampling_temp=1.0, keep_topk=-1):
    ''' Select next tokens randomly from the top k possible next tokens.
        shape of logits: [batch_size, vocab_size]
        shape of returned tokens and scores: [batch_size, 1]'''
    if sampling_temp == 0.0 or keep_topk == 1:
        topk_scores, topk_ids = logits.topk(1, dim=-1)
        if sampling_temp > 0:
            topk_scores /= sampling_temp
    else:
        logits = torch.div(logits, sampling_temp)

        if keep_topk > 0:
            top_values, top_indices = torch.topk(logits, keep_topk, dim=1)
            kth_best = top_values[:, -1].view([-1, 1])
            kth_best = kth_best.repeat([1, logits.shape[1]]).float()
            ignore = torch.lt(logits, kth_best)
            logits = logits.masked_fill(ignore, float('-inf'))

        probs = F.softmax(logits, -1)
        dist = torch.distributions.Multinomial(probs=probs, total_count=1)
#        dist = torch.distributions.Multinomial(logits=logits, total_count=1)
        topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
        topk_scores = probs.gather(dim=1, index=topk_ids)
#        topk_scores = logits.gather(dim=1, index=topk_ids)
    return topk_ids, topk_scores

def sample_with_probs(probs, keep_topk=-1):
    ''' Select next tokens randomly from the top k possible next tokens.
        shape of probs: [batch_size, vocab_size]
        shape of returned tokens and scores: [batch_size, 1]'''
    if keep_topk == 1:
        topk_scores, topk_ids = probs.topk(1, dim=-1)
    else:
        if keep_topk > 0:
            top_values, top_indices = torch.topk(probs, keep_topk, dim=1)
            kth_best = top_values[:, -1].view([-1, 1])
            kth_best = kth_best.repeat([1, probs.shape[1]]).float()
            ignore = torch.lt(probs, kth_best)
            probs = probs.masked_fill(ignore, float('-inf'))

        dist = torch.distributions.Multinomial(probs=probs, total_count=1)
        topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
        topk_scores = probs.gather(dim=1, index=topk_ids)
    return topk_ids, topk_scores

def tile(x, count, dim=0):
    perm = list(range(x.dim()))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.shape)
    out_size[0] *= count
    batch = x.shape[0]
    x = x.reshape(batch, -1).transpose(0, 1) \
         .repeat(count, 1).transpose(0, 1) \
         .reshape(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def tuple_map(fn, t, **kwargs):
    if t is None:
        return None
    if type(t) not in {list, tuple}:
        return fn(t, **kwargs)
    for s in t:
        return tuple(tuple_map(fn, s, **kwargs) for s in t)

def batch_bleu(hypotheses, references, smooth_method=3, n=4, average=True):
    ' expect tokenized inputs '
    assert len(hypotheses) == len(references)
    cc = SmoothingFunction()
    smooth = getattr(cc, 'method' + str(smooth_method))
    weights = [1. / n] * n
    scores = [sentence_bleu([ref], hyp, weights, smoothing_function=smooth) \
              for hyp, ref in zip(hypotheses, references)]
    return np.mean(scores) if average else scores

def batch_meteor(hypotheses, references, alpha=0.85, beta=0.2, gamma=0.6, average=True):
    assert len(hypotheses) == len(references)
    scores = [single_meteor_score(ref, hyp, alpha=alpha, beta=beta, gamma=gamma) \
              for hyp, ref in zip(hypotheses, references)]
    return np.mean(scores) if average else scores

def batch_rouge(hypotheses, references, metrics=['rouge-l'], average=True):
    assert len(hypotheses) == len(references)
    rouge = Rouge(metrics=metrics, max_n=4)
    if average:
        scores = rouge.get_scores(hypotheses, references)
    else:
        scores = [rouge.get_scores(hyp, ref) for hyp, ref in zip(hypotheses, references)]
    return scores

def perplexity(loss):
    return exp(min(loss, 100))

def group_score(ids, scores, groups):
    u = max(ids)
    assert u >= groups
    n = int(np.round(u / groups))
    result = [[] for _ in range(groups)]
    for i, score in zip(ids, scores):
        m = i // n
        if m >= groups:
            m = -1
        result[m].append(score)
    result = [np.mean(r) for r in result]
    return result

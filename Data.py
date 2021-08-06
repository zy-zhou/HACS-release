# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:51:52 2019

@author: Zhou
"""

from javalang import tokenizer
from Tree import node_filter
from Utils import save
from math import ceil
from itertools import chain
import pyastyle as style
import re
import random
import torch
from torchtext.data import Field
from tqdm import tqdm

subtoken = True
node_maxlen = 15
text_minlen = 1
text_maxlen = 30
code_maxlen = 400
max_statm = 60
statm_maxlen = 100
code_vocab_size = 35000
nl_vocab_size = 25000
batch_size = 64
epoches = 15
# special tokens
bos = '<s>'
eos = '</s>'
pad = '<pad>'
unk = '<unk>'

_NUM = {tokenizer.Integer,
        tokenizer.BinaryInteger,
        tokenizer.DecimalInteger,
        tokenizer.DecimalFloatingPoint,
        tokenizer.FloatingPoint,
        tokenizer.HexFloatingPoint,
        tokenizer.HexInteger,
        tokenizer.OctalInteger}

_LITERAL = {tokenizer.Character,
            tokenizer.Literal,
            tokenizer.String,
            tokenizer.Identifier}

def code_filter(code, subtoken=subtoken, max_tokens=code_maxlen):
    tokens = list(tokenizer.tokenize(code))
    result = []
    for token in tokens:
        if type(token) in _NUM:
            result.append('num')
        elif type(token) in _LITERAL:
            value = node_filter(token.value, subtoken)
            result.extend(value)
        else:
            result.append(token.value)
    return result[:max_tokens]

def tokenize_code(codes, subtoken=subtoken, max_tokens=code_maxlen, save_path=None):
    results = []
    for code in tqdm(codes, desc='Tokenizing codes...'):
        result = code_filter(code, subtoken, max_tokens)
        results.append(result)
    
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results

def split_code(codes, subtoken=subtoken, max_statm=max_statm, statm_maxlen=statm_maxlen,
               save_path=None):
    results = []
    for code in tqdm(codes, desc='Splitting codes into statements & tokenizing... '):
        code += '\n' # ensure that the formatter will work
        code = style.format(code, '--style=allman --delete-empty-lines --add-brackets')
        statements = code.strip().split('\n')
        
        result = []
        for statement in statements:
            statement = statement.strip()
            if statement == 'else {': # a bug of old AStyle
                result.extend([['else'], ['{']])
            else:
                tokens = code_filter(statement, subtoken, statm_maxlen)
                result.append(tokens)
        results.append(result[:max_statm])
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results

def nl_filter(s, subtoken=subtoken, min_tokens=text_minlen, max_tokens=text_maxlen):
    s = re.sub(r"\([^\)]*\)|(([eE]\.[gG])|([iI]\.[eE]))\..+|<\S[^>]*>", " ", s)
#    brackets; html labels; e.g.; i.e.
    s = re.sub(r"\d+\.\d+\S*|0[box]\w*|\b\d+[lLfF]\b", " num ", s)
    first_p = re.search(r"[\.\?\!]+(\s|$)", s)
    if first_p is not None:
        s = s[:first_p.start()]
    s = re.sub(r"https:\S*|http:\S*|www\.\S*", " url ", s)
    s = re.sub(r"\b(todo|TODO)\b.*|[^A-Za-z0-9\.,\s]|\.{2,}", " ", s)
    s = re.sub(r"\b\d+\b", " num ", s)
    s = re.sub(r"([\.,]\s*)+", lambda x: " " + x.group()[0] + " ", s)
    
    if subtoken:
        s = re.sub(r"[a-z][A-Z]", lambda x: x.group()[0] + " " + x.group()[1], s)
        s = re.sub(r"[A-Z]{2}[a-z]", lambda x: x.group()[0] + " " + x.group()[1:], s)
        s = re.sub(r"\w{32,}", " ", s) # MD5
        s = re.sub(r"[A-Za-z]\d+", lambda x: x.group()[0] + " ", s)
    s = re.sub(r"\s(num\s+){2,}", " num ", s)
    s = s.lower().split()
    return 0 if len(s) < min_tokens else s[:max_tokens]

def tokenize_nl(texts, subtoken=subtoken, min_tokens=text_minlen,
                max_tokens=text_maxlen, save_path=None):
    drop_list = set()
    results = []
    for idx, text in enumerate(tqdm(texts, desc='Tokenizing texts...')):
        tok_text = nl_filter(text, subtoken, min_tokens, max_tokens)
        if tok_text:
            results.append(tok_text)
        else:
            results.append([])
            drop_list.add(idx)
    
    if save_path is not None:
        save(results, save_path, is_json=True)
    print('number of dropped texts:', len(drop_list))
    return results, drop_list

def get_txt_vocab(texts, max_vocab_size=nl_vocab_size, is_tgt=True, save_path=None):
    if is_tgt:
        field = Field(init_token=bos, eos_token=eos, batch_first=True, pad_token=pad, unk_token=unk)
    else:
        field = Field(batch_first=True, pad_token=pad, unk_token=unk)
    field.build_vocab(texts, max_size=max_vocab_size)
    if save_path:
        print('Saving...')
        torch.save(field, save_path)
    return field

def id2word(word_ids, field, source=None, remove_eos=True, remove_unk=False, replace_unk=False):
    if replace_unk:
        assert type(source) is tuple and not remove_unk
        raw_src, alignments = source
    eos_id = field.vocab.stoi[eos]
    unk_id = field.vocab.stoi[unk]
    
    if remove_eos:
        word_ids = [s[:-1] if s[-1] == eos_id else s for s in word_ids]
    if remove_unk:
        word_ids = [filter(lambda x: x != unk_id, s) for s in word_ids]
    if not replace_unk:
        return [[field.vocab.itos[w] for w in s] for s in word_ids]
    else:
        return [[field.vocab.itos[w] if w != unk_id else rs[a[i].argmax()] \
                 for i, w in enumerate(s)] for s, rs, a in zip(word_ids, raw_src, alignments)]

class DataGenerator(object):
    def __init__(self, data, fields, batch_size=batch_size, sort='src_len', bucket_size=100,
                 shuffle=True, cycle=False, raw_data=False, device=None):
        ''' If both sort and shuffle, then sorts within buckets and shuffle the batches.
            Data and fields should be tuples of same lengths. '''
#        assert len(data) == len(fields)
        self.data = list(zip(*data))
        self.fields = fields
        self.batch_size = batch_size
        if sort == 'src_len':
            self.sort_key = lambda x: len(x[0])
        elif sort == 'tgt_len':
            self.sort_key = lambda x: len(x[-1])
        else:
            self.sort_key = None
        self.bucket_size = bucket_size
        self.shuffle =  shuffle
        self.cycle = cycle
        self.raw_data = raw_data
        self.device = device
    
    @classmethod
    def batch(cls, data, batch_size):
        low, high = 0, batch_size
        while True:
            if high >= len(data):
                yield data[low:]
                return
            else:
                yield data[low:high]
                low, high = high, high + batch_size
    
    @classmethod
    def pool(cls, data, batch_size, bucket_size, sort_key, shuffle_bathes=True):
        for bucket in cls.batch(data, batch_size * bucket_size):
            batches = cls.batch(sorted(bucket, key=sort_key), batch_size)
            if shuffle_bathes:
                batches = list(batches)
                random.shuffle(batches)
            for batch in batches:
                yield batch
    
    def process(self, raw_batch, field):
        raise NotImplementedError()
    
    def init_epoch(self):
        if self.shuffle:
            data = random.sample(self.data, len(self.data))
        else:
            data = self.data
        if self.sort_key is not None:
            self.batches = self.pool(data, self.batch_size, self.bucket_size,
                                     self.sort_key, self.shuffle)
        else:
            self.batches = self.batch(data, self.batch_size)
    
    def __len__(self):
        return ceil(len(self.data) / self.batch_size)
    
class TxtDataGenerator(DataGenerator):
    def process(self, raw_batch, field):
        lengths = list(map(len, raw_batch))
        lengths = torch.tensor(lengths, device=self.device)
        batch = field.process(raw_batch, device=self.device)
        return batch, lengths
    
    def __iter__(self):
        while True:
            self.init_epoch()
            for raw_batch in self.batches:
                src, tgt = zip(*raw_batch)
                src_batch, src_lengths = self.process(src, self.fields[0])
                tgt_batch = self.fields[1].process(tgt, device=self.device)
                if self.raw_data:
                    yield src_batch, src_lengths, tgt_batch, (src, tgt)
                else:
                    yield src_batch, src_lengths, tgt_batch
            if not self.cycle:
                return

class HANDataGenerator(TxtDataGenerator):
    def process(self, raw_batch, field):
        num_statm = list(map(len, raw_batch))
        batch = list(chain(*raw_batch))
        statm_lengths = list(map(len, batch))
        num_statm = torch.tensor(num_statm, device=self.device)
        statm_lengths = torch.tensor(statm_lengths, device=self.device)
        batch = field.process(batch, device=self.device)
        return batch, (num_statm, statm_lengths)

class EnsembleHANDataGen(HANDataGenerator):
    def __iter__(self):
        while True:
            self.init_epoch()
            for raw_batch in self.batches:
                codes, nodes, texts = zip(*raw_batch)
                code_batch, code_lengths = self.process(codes, self.fields[0])
                node_batch, num_nodes = self.process(nodes, self.fields[1])
                src_batch = (code_batch, node_batch)
                src_lengths = (code_lengths, num_nodes)
                tgt_batch = self.fields[2].process(texts, device=self.device)
                if self.raw_data:
                    yield src_batch, src_lengths, tgt_batch, (codes, texts)
                else:
                    yield src_batch, src_lengths, tgt_batch
            if not self.cycle:
                return

if __name__ == '__main__':
    from Utils import load
    
    for name in ('nl', 'codes', 'statms', 'nodes', 'split_ast'):
        if name == 'nl':
            raw = load('data/raw_data.json', is_json=True, key='nl')
            data, dropped = tokenize_nl(raw)
            dropped.update(torch.load('data/dropped.pkl'))
            torch.save(dropped, 'data/dropped.pkl')
            data = [t for i, t in enumerate(data) if i not in dropped]
            N = len(data)
            i, j = int(N * 0.8), int(N * 0.9)
        elif name in {'codes', 'statms'}:
            raw = load('data/raw_data.json', is_json=True, key='code', drop_list=dropped)
            if name == 'codes':
                data = tokenize_code(raw)
            else:
                data = split_code(raw)
        else:
            data = load('data/' + name + '.json', is_json=True, drop_list=dropped)
        
        train, val, test = data[:i], data[i:j], data[j:]
        if name in {'nl', 'codes', 'nodes'}:
            field = get_txt_vocab(train, max_vocab_size=nl_vocab_size,
                                  save_path='data/preprocessed/' + name + '_field.pkl')
        save(train, 'data/preprocessed/train.' + name + '.json', is_json=True)
        save(val, 'data/preprocessed/valid.' + name + '.json', is_json=True)
        save(test, 'data/preprocessed/test.' + name + '.json', is_json=True)
            
    
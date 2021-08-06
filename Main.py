# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:47:52 2019

@author: Zhou
"""
import torch
from Utils import load, batch_bleu, batch_rouge, batch_meteor
from Data import TxtDataGenerator, HANDataGenerator, EnsembleHANDataGen
from Modules import BasicDecoder, RNNEncoder
from Models import Model, HierarchialEncoder, Translator
import warnings
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(name, mode='train', batch_size=64, **kwargs):
    if mode == 'train':
        kwargs.update(dict(batch_size=batch_size, device=device))
    else:
        kwargs.update(dict(batch_size=batch_size, device=device, shuffle=False, sort=False))
    tgt_field = torch.load('data/preprocessed/nl_field.pkl')
    tgt = load('data/preprocessed/' + mode + '.nl.json', is_json=True)
    
    if name != 'ensemble':
        if name in {'codes', 'statms'}:
            src_field = torch.load('data/preprocessed/codes_field.pkl')
        else:
            src_field = torch.load('data/preprocessed/nodes_field.pkl')
        src = load('data/preprocessed/' + mode + '.' + name + '.json', is_json=True)
        fields = src_field, tgt_field
        if name in {'statms', 'split_ast'}:
            data_gen = HANDataGenerator((src, tgt), fields, **kwargs)
        else:
            data_gen = TxtDataGenerator((src, tgt), fields, **kwargs)
    else:
        codes_field = torch.load('data/preprocessed/codes_field.pkl')
        nodes_field = torch.load('data/preprocessed/nodes_field.pkl')
        statms = load('data/preprocessed/' + mode + '.statms.json', is_json=True)
        split_ast = load('data/preprocessed/' + mode + '.split_ast.json', is_json=True)
        fields = codes_field, nodes_field, tgt_field
        data_gen = EnsembleHANDataGen((statms, split_ast, tgt), fields, **kwargs)
    return data_gen, fields

def build_model(name, fields, pretrn_encoder=None):
    src_field, tgt_field = fields
    if name == 'bilstm':
        e = RNNEncoder(src_field)
        memory_dim = e.units * e.directions
        d = BasicDecoder(tgt_field, memory_dim=memory_dim)
    else:
        if pretrn_encoder is None:
            token_e = RNNEncoder(src_field)
        else:
            token_e = pretrn_encoder
        memory_dim = token_e.units * token_e.directions
        statm_e = RNNEncoder(field=None, in_dim=memory_dim)
        e = HierarchialEncoder(statm_e, token_e, intra_attn='token', cat_contxt=True)
        d = BasicDecoder(tgt_field, memory_dim=memory_dim * 2)
    
    model = Model(e, d)
    model = model.to(device)
    return model

if __name__ == '__main__':
    test_gen, fields = load_data('ensemble', 'test', batch_size=100)
    HACS_token = build_model('hacs', (fields[0], fields[-1]))
    checkpoint = torch.load('checkpoints/HACS-token.pt', map_location=device)
    HACS_token.load_state_dict(checkpoint['model'])
    
    HACS_AST = build_model('hacs', fields[1:])
    checkpoint = torch.load('checkpoints/HACS-AST.pt', map_location=device)
    HACS_AST.load_state_dict(checkpoint['model'])

    model = Model.ensemble(HACS_token, HACS_AST)
    evaluator = Translator(model, metrics=[])
    predicts, reports = evaluator(test_gen, save_path='predicts/predict_HACS.txt')
    
    hyp = [s.split() for s in predicts]
    ref = load('data/preprocessed/test.nl.json', is_json=True)
    bleu_4 = batch_bleu(hyp, ref, smooth_method=0)
    print('BLEU-4 score: {:.2f}'.format(bleu_4 * 100))
    bleu_s = batch_bleu(hyp, ref, smooth_method=3)
    print('Smoothed BLEU-4 score: {:.2f}'.format(bleu_s * 100))
    hyp = predicts
    ref = [' '.join(s) for s in ref]
    rouge = batch_rouge(hyp, ref)
    print('ROUGE-L score: {:.2f}'.format(rouge['rouge-l']['f'] * 100))
    meteor = batch_meteor(hyp, ref)
    print('METEOR score: {:.2f}'.format(meteor * 100))
    
    

    
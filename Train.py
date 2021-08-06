# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:35:27 2019

@author: Zhou
"""
from Utils import sequence_loss, perplexity
from Models import Translator
from Data import pad
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod

pretrn_epoches = 55
epoches = 65
lr = 0.001
adapt_lr = 0.005
optimizer = 'adam'
max_grad_norm = 5
lr_decay = None
patience = 4
val_metric = 'bleu'

def get_optimizer(optimizer, lr, params):
    params = filter(lambda p: p.requires_grad, params)
    if optimizer == 'sgd':
        return optim.SGD(params, lr=lr)
    elif optimizer == 'nag':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'adagrad':
        return optim.Adagrad(params, lr=lr)
    elif optimizer == 'adadelta':
        return optim.Adadelta(params, lr=lr)
    elif optimizer == 'adam':
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError('Invalid optimizer type: ' + optimizer)

class Trainer(object):
    def __init__(self, model, epoches=epoches, optimizer=optimizer, lr=lr,
                 max_grad_norm=max_grad_norm, lr_decay=lr_decay, metrics=['loss'],
                 val_metric=val_metric, save_path=None, load_path=None, patience=patience,
                 save_per_epoch=True, **kwargs):
        self.model = model
        self.pad_id = model.decoder.field.vocab.stoi[pad]
        self.epoches = epoches
        self.save_path = save_path
        self.save_per_epoch = save_per_epoch
        self.patience = patience if save_path else float('inf')
        self.optimizer = get_optimizer(optimizer, lr, model.parameters())
        if lr_decay:
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)
        
        self.metrics = set(metrics)
        assert val_metric in metrics
        self.val_metric = val_metric
        if self.metrics.difference({'loss'}):
            kwargs['metrics'] = self.metrics
            self.evaluator = Translator(model, **kwargs)
        
        self.device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        if load_path:
            self.load_states(load_path)
        else:
            self.max_grad_norm = max_grad_norm
            self.curr_epoch = self.curr_iter = self.best_epoch = 0
            self.best_score = float('-inf')
            self.log = defaultdict(list)
        
    def save_states(self):
        print('Saving model and settings...')
        checkpoint = dict(model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict(),
                          max_grad_norm = self.max_grad_norm,
                          epoch=self.curr_epoch,
                          iteration=self.curr_iter,
                          log=self.log,
                          best_epoch=self.best_epoch,
                          best_score=self.best_score)
        if hasattr(self, 'scheduler'):
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        new_path = self.save_path
        if self.save_per_epoch:
            new_path = new_path.split('.')
            new_path[0] = new_path[0] + '_epoch' + str(self.curr_epoch)
            new_path = '.'.join(new_path)
        torch.save(checkpoint, new_path)
    
    def load_states(self, load_path):
        print('Loading model and settings from checkpoint...')
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.max_grad_norm = checkpoint['max_grad_norm']
        self.curr_epoch = checkpoint['epoch']
        self.curr_iter = checkpoint['iteration']
        self.log = checkpoint['log']
        self.best_epoch = checkpoint['best_epoch']
        self.best_score = checkpoint['best_score']
        if 'scheduler' in checkpoint.keys() and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])
    
    def validate_epoch(self, val_iter):
        if self.metrics.difference({'loss'}):
            _, reports = self.evaluator(val_iter)
            del reports['scores'], reports['attn_history']
            return reports
            
        self.model.eval()
        reports = dict(loss=0)
        pbar = tqdm(val_iter, desc='Validating epoch ' + str(self.curr_epoch))
        with torch.no_grad():
            for batch in pbar:
                outputs, _ = self.model(*batch)
                loss = sequence_loss(outputs, batch[-1][:,1:], is_probs=self.model.is_ensemble,
                                     pad_id=self.pad_id)
                reports['loss'] += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        reports['loss'] /= len(val_iter)
        print('val loss: {:.2f}'.format(reports['loss']))
        reports['ppl'] = perplexity(reports['loss'])
        print('val ppl: {:.2f}'.format(reports['ppl']))
        torch.cuda.empty_cache()
        return reports
    
    @abstractmethod
    def train_epoch(self, train_iter):
        pass
    
    @abstractmethod
    def init_generator(self, train_iter, val_iter=None):
        pass
    
    def __call__(self, train_iter, val_iter=None):
        self.init_generator(train_iter, val_iter)
        
        for epoch in range(self.epoches):
            reports = self.train_epoch(train_iter)
            for key, value in reports.items():
                self.log[key].append(value)
            self.curr_epoch += 1
            
            if val_iter is not None:
                reports = self.validate_epoch(val_iter)
                for key, value in reports.items():
                    self.log['val_' + key].append(value)
                
                score = - reports['ppl'] if self.val_metric == 'loss' else reports[self.val_metric]
                if score > self.best_score:
                    self.best_epoch = self.curr_epoch
                    self.best_score = score
                    if self.save_path:
                        self.save_states()
                elif self.curr_epoch - self.best_epoch >= self.patience:
                    print('Early stopped at epoch ' + str(self.curr_epoch))
                    print('Best validating score reached at epoch ' + str(self.best_epoch))
                    break
            
        return self.log
    
class TeacherForcing(Trainer):
    def train_step(self, src_batch, src_lengths, tgt_batch):
        self.optimizer.zero_grad()
        outputs, _ = self.model(src_batch, src_lengths, tgt_batch)
        loss = sequence_loss(outputs, tgt_batch[:,1:], is_probs=self.model.is_ensemble,
                             pad_id=self.pad_id)
        
        loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
    
    def train_epoch(self, train_iter):
        self.model.train()
        reports = dict(loss=0)
        
        pbar = tqdm(train_iter, desc='Training epoch ' + str(self.curr_epoch + 1))
        for batch in pbar:
            loss = self.train_step(*batch)
            self.curr_iter += 1
            reports['loss'] += loss
            pbar.set_postfix(loss=loss)
#            torch.cuda.empty_cache()
        
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        reports['loss'] /= len(train_iter)
        print('train loss: {:.2f}'.format(reports['loss']))
        reports['ppl'] = perplexity(reports['loss'])
        print('train ppl: {:.2f}'.format(reports['ppl']))
        reports['lr'] = self.optimizer.param_groups[0]['lr']
        print('learning rate: {:.5f}'.format(reports['lr']))
        return reports
    
    def init_generator(self, train_iter, val_iter=None):
        train_iter.raw_data = False
        if val_iter is not None:
            assert len(self.metrics) > 0
            val_iter.raw_data = True if self.metrics.difference({'loss'}) else False
    
if __name__ == '__main__':
    from Main import load_data, build_model
    
    print('Pretraining token-level biLSTM...')
    train_gen, fields = load_data('codes', 'train')
    val_gen, _ = load_data('codes', 'valid')
    model = build_model('bilstm', fields)
    
    trainer = TeacherForcing(model, epoches=pretrn_epoches, metrics=['loss', 'bleu'],
                             smooth=0, save_per_epoch=False,
                             save_path='checkpoints/bilstm_token.pt')
    reports = trainer(train_gen, val_gen)
    
    print('Training HACS-token...')
    train_gen, fields = load_data('statms', 'train')
    val_gen, _ = load_data('statms', 'valid')
    model = build_model('hacs', fields, pretrn_encoder=model.encoder)
    
    trainer = TeacherForcing(model, metrics=['loss', 'bleu'], smooth=0, save_per_epoch=False,
                             save_path='checkpoints/HACS-token.pt')
    reports = trainer(train_gen, val_gen)
    
    del trainer, model
    torch.cuda.empty_cache()
    print('Pretraining node-level biLSTM...')
    train_gen, fields = load_data('nodes', 'train')
    val_gen, _ = load_data('nodes', 'valid')
    model = build_model('bilstm', fields)
    
    trainer = TeacherForcing(model, epoches=pretrn_epoches, metrics=['loss', 'bleu'],
                             smooth=0, save_per_epoch=False,
                             save_path='checkpoints/bilstm_node.pt')
    reports = trainer(train_gen, val_gen)
    
    print('Training HACS-AST...')
    train_gen, fields = load_data('split_ast', 'train')
    val_gen, _ = load_data('split_ast', 'valid')
    model = build_model('hacs', fields, pretrn_encoder=model.encoder)
    
    trainer = TeacherForcing(model, metrics=['loss', 'bleu'], smooth=0, save_per_epoch=False,
                             save_path='checkpoints/HACS-AST.pt')
    reports = trainer(train_gen, val_gen)
    

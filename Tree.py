# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:37:17 2019

@author: Zhou
"""

from Utils import parallel, save
import javalang
import re
from tqdm import tqdm
from collections import deque
from functools import partial

node_maxlen = 15
max_nodes = 400
max_tokens = 450
max_statms = 60
min_statms = 2
subtoken = True
workers = 4

_REF = {javalang.tree.MemberReference,
        javalang.tree.ClassReference,
        javalang.tree.MethodInvocation}

_BLOCK = {'body',
          'block',
          'then_statement',
          'else_statement',
          'catches',
          'finally_block'}

_IGNORE = {'throws',
           'dimensions',
           'prefix_operators',
           'postfix_operators',
           'selectors',
           'types',
           'case'}

_LITERAL_NODE = {'Annotation',
                 'MethodDeclaration',
                 'ConstructorDeclaration',
                 'FormalParameter',
                 'ReferenceType',
                 'MemberReference',
                 'VariableDeclarator',
                 'MethodInvocation',
                 'Literal'}

def get_value(node, token_list):
    value = None
    length = len(token_list)
    if hasattr(node, 'name'):
        value = node.name
    elif hasattr(node, 'value'):
        value = node.value
    elif type(node) in _REF and node.position:
        for i, token in enumerate(token_list):
            if node.position == token.position:
                pos = i + 1
                value = str(token.value)
                while pos < length and token_list[pos].value == '.':
                    value = value + '.' + token_list[pos + 1].value
                    pos += 2
                break
    elif type(node) is javalang.tree.TypeArgument:
        value = str(node.pattern_type)
    elif type(node) is javalang.tree.SuperMethodInvocation \
            or type(node) is javalang.tree.SuperMemberReference:
        value = str(node.member)
    elif type(node) is javalang.tree.BinaryOperation:
        value = node.operator
    return value

def parse_single(code, max_nodes=max_nodes):
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except:
        return []
    
    result = []
    q = deque([tree])
    idx = 1 # index of the next child node (level traversal)
    while len(q) > 0 and len(result) <= max_nodes:
        node = q.popleft()
        if type(node) is dict:
            result.append(node)
            continue
        node_d = {'id': len(result), 'type': node.__class__.__name__, 'children': []}
        value = get_value(node, token_list)
        if value is not None and type(value) is str:
            node_d['value'] = value
        result.append(node_d)
        
        for attr, child in zip(node.attrs, node.children):
            if idx >= max_nodes:
                break
            if attr in _BLOCK and child:
                if type(child) is javalang.tree.BlockStatement:
                    child = child.statements
                block_d = {'id': idx, 'type': attr, 'children': []}
                node_d['children'].append(idx)
                idx += 1
                q.append(block_d)
                node_d = block_d
            if isinstance(child, javalang.ast.Node):
                node_d['children'].append(idx)
                idx += 1
                q.append(child)
            elif type(child) is list and child and attr not in _IGNORE:
                child = [c[0] if type(c) is list else c for c in child[:max_nodes - idx]]
                child_idx = [idx + i for i in range(len(child))]
                node_d['children'].extend(child_idx)
                idx += len(child)
                q.extend(child)
    return result

def get_ast(codes, max_nodes=max_nodes, workers=workers, save_path=None):
    desc = 'Building ASTs...'
    if workers > 1 and len(codes) > 2000:
        print(desc)
        func = partial(parse_single, max_nodes=max_nodes)
        results = parallel(func, codes, workers=workers)
    else:
        results = []
        for code in tqdm(codes, desc=desc):
            results.append(parse_single(code, max_nodes))
    
    dropped = set(i for i, tree in enumerate(results) if len(tree) == 0)
    print('Number of parse failures:', len(dropped))
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results, dropped

def node_filter(s, subtoken=subtoken):
    s = re.sub(r"\d+\.\d+\S*|0[box]\w*|\b\d+[lLfF]\b", " num ", s)
    s = re.sub(r"%\S*|[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\b\d+\b", " num ", s)
    if subtoken:
        s = re.sub(r"[a-z][A-Z]", lambda x: x.group()[0] + " " + x.group()[1], s)
        s = re.sub(r"[A-Z]{2}[a-z]", lambda x: x.group()[0] + " " + x.group()[1:], s)
        s = re.sub(r"\w{32,}", " ", s) # MD5, hash
        s = re.sub(r"[A-Za-z]\d+", lambda x: x.group()[0] + " ", s)
    s = re.sub(r"\s(num\s+){2,}", " num ", s)
    return s.lower().split()

def pre_traverse(tree, idx, node_maxlen, subtoken):
    node = tree[idx]
    result = []
    result.append(node['type'])
    if node['type'] in _LITERAL_NODE:
        value = node_filter(node['value'], subtoken)
        result.extend(value[:node_maxlen])
    elif node.get('value'):
        result.append(node['value'].lower())
    
    if node['children']:
        for child in node['children']:
            result.extend(pre_traverse(tree, child, node_maxlen, subtoken))
    return result

def get_node_seq(trees, node_maxlen=node_maxlen, subtoken=subtoken,
                 max_tokens=max_tokens, save_path=None):
    results = []
    for tree in tqdm(trees, desc='Obtaining node seqs...'):
        results.append(pre_traverse(tree, 0, node_maxlen, subtoken)[:max_tokens])
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results

def split_ast(trees, node_maxlen=node_maxlen, max_statms=max_statms,
              min_statms=min_statms, subtoken=subtoken, save_path=None):
    def traverse(tree, idx):
        node = tree[idx]
        subTrees = []
        blocks = []
        for i, child in enumerate(node['children']):
            if tree[child]['type'] in _BLOCK:
                blocks.append(child)
                del node['children'][i]
        subTrees.append(pre_traverse(tree, idx, node_maxlen, subtoken)[:max_tokens])
        for block in blocks:
            for child in tree[block]['children']:
                subTrees.extend(traverse(tree, child))
        return subTrees
    
    results = []
    dropped = set()
    for idx, tree in enumerate(tqdm(trees, desc='Splitting ASTs...')):
        result = traverse(tree, 0)[:max_statms]
        results.append(result)
        if len(result) < min_statms:
            dropped.add(idx)
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results, dropped

if __name__ == '__main__':
    from Utils import load
    import torch
    
    codes = load('data/raw_data.json', is_json=True, key='code')
    trees, dropped = get_ast(codes, save_path='data/node_seqs.json')
    node_seqs = get_node_seq(trees, save_path='data/nodes.json')
    subTree_seqs, dropped_s = split_ast(trees, save_path='data/split_ast.json')
    torch.save(dropped.union(dropped_s), 'data/dropped.pkl')

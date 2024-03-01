# Compute and merge CA PA DA MI VI TA scores

import pandas as pd
import os, sys
import torch
import math
import json
from textdistance import levenshtein as ld

# Utils

def read(i, r):
    '''read json
    '''
    if i != 'winogrande':
        jpath = os.path.join(PARA['path'], str(r), 'data', i, 'val.jsonl')
    else:
        jpath = os.path.join(PARA['path'], str(r), 'data', i, 'winogrande_1.1', 'dev.jsonl')
    
    with open(jpath, 'r') as f: 
        data = [json.loads(line) for line in f]

    return data

def score(sent1, sent2):
    ''' score two strings with levenshtein distance
    '''
    score = ld.normalized_similarity(sent1, sent2)

    return score


# Mains

def ca_score(df):
    '''
    pmt file as a csv
    '''
    df['ca'] = df.apply(lambda x: score(x['org'], x['rnd']), axis=1)

    return df


def pa_score(df):
    '''
    pmt file as a csv
    '''
    df['pa'] = df.apply(lambda x: score(x['org'], x['gnt']), axis=1)

    return df

def da_score(df):
    '''
    pmt file as a csv
    '''
    df['da'] = df.apply(lambda x: x['pa']-x['ca'], axis=1)

    return df

def lt_score(pfile, df, inum):
    '''
    pfile file p file
    return a list
    '''
    # get pred
    predf = torch.load(pfile)
    idxs_list = []; pred_list = []

    for task, predt in predf.items():
        for guid, pred in zip(list(predt["guids"]), list(predt["preds"])): # format: boolq test-3237 1
            idxs_list.append(int(guid.split('-')[1])+1) # start from 1 not 0
            pred_list.append(pred)

    if max(pred_list) > 1:
        d = {0:0, 1:1, 2:1}
        pred_list = [d[int(i)] for i in pred_list]

    dt = pd.DataFrame({'inum':inum, 'lt':pred_list})
    dt = dt.groupby('inum', as_index=False).agg({'lt':'mean'})
    df = df.merge(dt, how='left', on='inum')

    return df


def pl_score(pfile, df, inum):
    '''
    pfile file as a json
    return a list
    '''
    # get pred
    predf = torch.load(pfile)
    idxs_list = []; pred_list = []

    for task, predt in predf.items():
        for guid, pred in zip(list(predt["guids"]), list(predt["preds"])): # format: boolq test-3237 1
            idxs_list.append(int(guid.split('-')[1])+1) # start from 1 not 0
            pred_list.append(pred)

    if max(pred_list) > 1:
        d = {0:0, 1:1, 2:1}
        pred_list = [d[int(i)] for i in pred_list]

    dt = pd.DataFrame({'inum':inum, 'pl':pred_list})
    dt = dt.groupby('inum', as_index=False).agg({'pl':'mean'})
    df = df.merge(dt, how='left', on='inum')

    return df


def vc_score(ppath, df, inum, labs):
    '''
    val pred from pred file
    '''
    predf = torch.load(ppath)
    idxs_list = []; pred_list = []

    for task, predt in predf.items():
        tl = labs
        predi = torch.tensor(predt["outputs"])
        softmax = torch.nn.Softmax(dim=1)
        predi = softmax(predi) # -> P()
        for guid, pred, idxn, ptkn in zip(list(predt["guids"]), list(predi), tl, list(predt["preds"])): # format: boolq test-3237 1
            idxs_list.append(int(guid.split('-')[1])+1) # start from 1 not 0
            if len(list(pred)) > 2:
                pred = list(pred)
                pred = [pred[0], pred[1]+pred[2]]
            pred_p = pred[idxn]
            log2_pred_p = round(math.log2(pred_p), 4)
            # print(idxn,ptkn,-log2_pred_p)
            pred_list.append(-log2_pred_p)
    assert(len(inum) == len(idxs_list))
    dt = pd.DataFrame({'inum':inum, 'vc':pred_list})
    dt = dt.groupby('inum', as_index=False).agg({'vc':'mean'})
    df = df.merge(dt, how='left', on='inum')

    return df


def vt_score(ppath, df, inum, labs):
    '''
    val pred from pred file
    '''
    predf = torch.load(ppath)
    idxs_list = []; pred_list = []

    for task, predt in predf.items():
        tl = labs
        predi = torch.tensor(predt["outputs"])
        softmax = torch.nn.Softmax(dim=1)
        predi = softmax(predi) # -> P()
        for guid, pred, idxn, ptkn in zip(list(predt["guids"]), list(predi), tl, list(predt["preds"])): # format: boolq test-3237 1
            idxs_list.append(int(guid.split('-')[1])+1) # start from 1 not 0
            if len(list(pred)) > 2:
                pred = list(pred)
                pred = [pred[0], pred[1]+pred[2]]
            pred_p = pred[idxn]
            log2_pred_p = round(math.log2(pred_p), 4)
            # print(idxn,ptkn,-log2_pred_p)
            pred_list.append(-log2_pred_p)
    assert(len(inum) == len(idxs_list))
    dt = pd.DataFrame({'inum':inum, 'vt':pred_list})
    dt = dt.groupby('inum', as_index=False).agg({'vt':'mean'})
    df = df.merge(dt, how='left', on='inum')

    return df


def vs_score(ppath, df, inum):
    '''
    val pred from pred file
    '''
    predf = torch.load(ppath)
    idxs_list = []; pred_list = []

    for task, predt in predf.items():
        tl = list(df['tl'])
        predi = torch.tensor(predt["outputs"])
        softmax = torch.nn.Softmax(dim=1)
        predi = softmax(predi)
        for guid, pred, idxn in zip(list(predt["guids"]), list(predi), tl): # format: boolq test-3237 1
            idxs_list.append(int(guid.split('-')[1])+1) # start from 1 not 0
            if len(list(pred)) > 2:
                pred = list(pred)
                pred = [pred[0], pred[1]+pred[2]]
            pred_p = pred[idxn]
            log2_pred_p = round(math.log2(pred_p), 4)
            # print(idxn,ptkn,-log2_pred_p)
            pred_list.append(-log2_pred_p)
    assert(len(inum) == len(idxs_list))
    dt = pd.DataFrame({'inum':inum, 'vs':pred_list})
    df = df.merge(dt, how='left', on='inum')

    return df


def tl_score(jdata, ppath, tname):
    '''
    jdata: json_file file as a json containing true labels
    ppath: val pred file
    return a dataframe
    '''
    # create task-lab dict
    ldict = dict(zip(PARA['tasks'], PARA['labs']))
    with open(jdata, 'r') as f: 
        jdata = [json.loads(line) for line in f]
    # find true labels
    if tname != 'multirc':
        labs = [item[ldict[tname]] for item in jdata] 
        labs = [PARA['labf'][tname][i] for i in labs]
        df = pd.DataFrame({'inum':range(1,len(labs)+1), 'tl':labs})
        inum = range(1,len(labs)+1)
    else:
        labs = [[rx['label'], i+1] for i, item in enumerate(jdata) for text in item['passage']['questions'] for rx in text['answers']]
        inum = [i[1] for i in labs]
        labs = [PARA['labf'][tname][i[0]] for i in labs]
        df = pd.DataFrame({'inum':inum, 'tl':labs})
    df = vs_score(ppath, df, inum)
    df = df.groupby('inum', as_index=False).agg({'tl':'mean', 'vs':'mean'})

    return df, inum, labs


def reduce(df):
    '''
    reduce a df based on inum 
    '''
    df = df[df['slen']>3]
    rd = df.groupby('inum', as_index=False).agg(
                                                {'ca':'mean',
                                                'pa':'mean',
                                                'da':'mean',
                                                'slen':'mean',
                                                # {'mp':'mean'},
                                                # {'cp':'mean'},
                                                'mi':'mean'},
                                                )
    #rd = rd.to_frame()

    return rd


def avg_rnd_prd(pfile, df, rnum):
    '''
    pfile file as a json
    return a list
    '''
    # get pred
    predf = torch.load(pfile)
    idxs_list = []; pred_list = []

    for task, predt in predf.items():
        for guid, pred in zip(list(predt["guids"]), list(predt["preds"])): # format: boolq test-3237 1
            idxs_list.append(int(guid.split('-')[1])+1) # start from 1 not 0
            pred_list.append(pred)

    if max(pred_list) > 1:
        d = {0:0, 1:1, 2:1}
        pred_list = [d[int(i)] for i in pred_list]

    dt = pd.DataFrame({'inum':idxs_list, 'pl-{}'.format(rnum):pred_list})
    # dt = dt.groupby('inum', as_index=False).agg({'pl':'mean'})
    df = df.merge(dt, how='left', on='inum')

    return df


def tl_pl_init(jdata, pfile, tname):
    '''
    pfile file as a json
    return a list
    '''
    # create task-lab dict
    ldict = dict(zip(PARA['tasks'], PARA['labs']))
    with open(jdata, 'r') as f: 
        jdata = [json.loads(line) for line in f]
    # find true labels
    if tname != 'multirc':
        labs = [item[ldict[tname]] for item in jdata] 
        labs = [PARA['labf'][tname][i] for i in labs]
        df = pd.DataFrame({'inum':range(1,len(labs)+1), 'tl':labs})
        # inum = range(1,len(labs)+1)
    else:
        labs = [[rx['label'], i+1] for i, item in enumerate(jdata) for text in item['passage']['questions'] for rx in text['answers']]
        # inum = [i[1] for i in labs]
        labs = [PARA['labf'][tname][i[0]] for i in labs]
        df = pd.DataFrame({'inum':range(1,len(labs)+1), 'tl':labs})
    df['dataset'] = tname

    '''get pred file'''
    predf = torch.load(pfile)
    idxs_list = []; pred_list = []

    for task, predt in predf.items():
        for guid, pred in zip(list(predt["guids"]), list(predt["preds"])): # format: boolq test-3237 1
            idxs_list.append(int(guid.split('-')[1])+1) # start from 1 not 0
            pred_list.append(pred)

    if max(pred_list) > 1:
        d = {0:0, 1:1, 2:1}
        pred_list = [d[int(i)] for i in pred_list]

    dt = pd.DataFrame({'inum':idxs_list, 'pl':pred_list})

    dt = dt.merge(df, how='left', on='inum')

    return dt


PARA = {
    'tasks': ['boolq', 'cb', 'copa', 'mrpc', 'multirc', 'qqp', 'rte', 'winogrande', 'sst'],
    'labs': ['label', 'label', 'label', 'label', 'passage', 'label', 'label','answer', 'label'],
    'labf':  {
            'boolq': {True: 1, False: 0},
            'cb': {'neutral': 1, 'contradiction': 1, 'entailment': 0},
            'copa': {0:0, 1:1},
            'mrpc': {'0':0, '1':1},
            'multirc': {0:0, 1:1},
            'qqp': {'0':0, '1':1},
            'rte': {'not_entailment': 1, 'entailment': 0},
            'sst': {'0':0, '1':1},
            'winogrande': {'1':0, '2':1},
            },
    'rnds': [1, 2, 3, 4, 5, 6],
    'path': '/network/scratch/x/xuanda.chen/rnd/',
}

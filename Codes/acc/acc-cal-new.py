# 

import pandas as pd
import numpy as np
import os, sys
import torch
import json
from textdistance import levenshtein as ld
from acc import *

PARA = {
    'tasks': ['boolq', 'copa', 'mrpc', 'qqp', 'rte', 'winogrande', 'sst'],
    'labs': ['label', 'label', 'label', 'label', 'label','answer', 'label'],
    'labf':  {
            'boolq': {True: 1, False: 0},
            'copa': {0:0, 1:1},
            'mrpc': {'0':0, '1':1},
            'qqp': {'0':0, '1':1},
            'rte': {'not_entailment': 1, 'entailment': 0},
            'sst': {'0':0, '1':1},
            'winogrande': {'1':0, '2':1},
            },
    'rnds': ['rnd1', 'rnd2', 'rnd3', 'rnd4', 'rnd5', 'rnd6'],
    'path': '/Users/xdchen/Downloads/eval2/data/data-out',
}

def get_pred_label(pfile):
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
    dt = pd.DataFrame({'inum':idxs_list, 'pl':pred_list})

    return dt

def get_true_label(jdata, tname):
    '''
    pfile file as a json
    return a list
    '''
    # create task-lab dict
    ldict = dict(zip(PARA['tasks'], PARA['labs']))
    with open(jdata, 'r') as f: 
        jdata = [json.loads(line) for line in f]
    # find true labels
    labs = [item[ldict[tname]] for item in jdata] 
    labs = [PARA['labf'][tname][i] for i in labs]
    df = pd.DataFrame({'inum':range(1,len(labs)+1), 'tl':labs})

    return df

def main(rnum, tname):

    # true labs
    jpath = os.path.join(PARA['path'], '{}-{}-{}'.format('run', tname, 'val.jsonl'))
    # pred labs with order
    rpath = os.path.join(PARA['path'], '{}-{}-{}'.format('run', tname, 'val_preds.p'))
    # pred labs with perturbation
    ppath = os.path.join(PARA['path'], '{}-{}-{}'.format(rnum, tname, 'val_preds.p'))
    # sent info
    spath = os.path.join(PARA['path'], '{}-{}-{}'.format(rnum, tname, 'score.csv'))
    # output
    opath = os.path.join(PARA['path'], '{}-{}-{}'.format(rnum, tname, 'rdc.csv')) 
    # read in a 
    daf = pd.read_csv(spath, sep='|', on_bad_lines='skip', engine='python',quoting=3)
    # inum|slen|cp|mp|mi|ca|pa|da
    # daf['mi2'] = daf.apply(lambda x: np.exp(x['cp']) / np.exp(x['mp']), axis=1)
    daf['cp'] = daf.apply(lambda x: x['cp'] / x['slen'], axis=1)
    daf['mp'] = daf.apply(lambda x: x['mp'] / x['slen'], axis=1)
    daf = daf[['inum', 'slen', 'da', 'cp', 'mp']]
    daf = daf[pd.to_numeric(daf['inum'], downcast="integer", errors='coerce').notnull()]
    daf['inum'] = daf['inum'].astype('int64')
    # daf = daf[[isinstance(value, int) for value in daf['inum']]]
    # get labels
    dtl = get_true_label(jpath, tname)
    dpl = get_pred_label(ppath)
    prl = get_pred_label(rpath)
    # change name
    prl = prl.rename(columns = {'pl':'rl'})
    daf = daf.merge(dtl, how='left', on='inum')
    daf = daf.merge(dpl, how='left', on='inum')
    daf = daf.merge(prl, how='left', on='inum')
    # get correct samples
    # comment if not 
    calmean = 1
    if calmean == 1:
        daf = daf[daf['pl']==daf['tl']]
    # daf['mi'] = 1/daf['mi']
    # sep by same or diff pred
    daf_same = daf[daf['rl']==daf['pl']]
    daf_diff = daf[daf['rl']!=daf['pl']]
    # same max; diff min
    daf_same = daf_same.groupby(['inum'], as_index=False).agg({'slen':'mean', 
                                                               'cp': 'mean',
                                                               'mp': 'mean',
                                                               'da':'mean', 
                                                               'tl':'mean', 
                                                               'rl':'mean'})
    daf_diff = daf_diff.groupby(['inum'], as_index=False).agg({'slen':'mean', 
                                                               'cp': 'min',
                                                               'mp': 'min',
                                                               'da':'min', 
                                                               'tl':'mean', 
                                                               'rl':'mean'})
    # concat
    dof = pd.concat([daf_same, daf_diff])
    # add dataset name and rnd num
    dof['dataset'] = tname
    dof['rndnum'] = rnum
    
    return dof

    
opath = os.path.join(PARA['path'], 'selected_score6.csv') 
fof = pd.DataFrame()
for tname in PARA['tasks']:
    # tname = 'rte'
    print('now processing {} ...'.format(tname))
    # loop over multiple rounds
    for rnum in PARA['rnds']:
        print('now at {}'.format(rnum))
        dof = main(rnum, tname)
        try:
            fof = pd.concat([fof, dof])
        except:
            print('Return NULL Values')
    # break
fof['lmi'] = fof.apply(lambda x: x['cp'] - x['mp'], axis=1)
fof['mi'] = fof.apply(lambda x: np.exp(x['lmi']), axis=1)
fof = fof.round(4)
print(fof.groupby(['dataset'], as_index=False).agg({'lmi':'mean'}))
fof.to_csv(opath, sep='|', index=False)
print(fof)

# Compute and merge CA PA DA MI VI TA scores

import pandas as pd
import os, sys
import torch
import json
from textdistance import levenshtein as ld
from acc import *


def main1():
    '''
    gen ca pa da
    '''

    for i in PARA['tasks']:
        for r in PARA['rnds']:

            # if i != 'winogrande':
            #     dpath = os.path.join(PARA['path'], 'rnd{}'.format(str(r)), 'data', i, '{}-pmt.csv'.format(i))
            # else:
            #     dpath = os.path.join(PARA['path'], 'rnd{}'.format(str(r)), 'data', i, 'winogrande_1.1', '{}-pmt.csv'.format(i))
            dpath = os.path.join(PARA['path'], 'rnd{}-{}-{}'.format(r, i, 'pmt.csv'))
            opath = os.path.join(PARA['path'], 'rnd{}-{}-{}'.format(r, i, 'score.csv'))

            df = pd.read_csv(dpath, sep='|',
                     dtype={
                     "org": "string", 
                     "rnd": "string", 
                     "gnt": "string",
                     },
                     on_bad_lines='skip',
                     engine='python',
                     quoting=3,
                     )
            # print(i, r, 'before', df.shape[0], df.shape[1])
            df = df.dropna()
            df = ca_score(df)
            df = pa_score(df)
            df = da_score(df)
            df = df.round(4)
            # print(i, r, 'after', df.shape[0], df.shape[1])
            # df.to_csv(os.path.join(os.path.dirname(dpath), 'score.csv'), sep='|', index=False)
            # opath
            df.to_csv(opath, sep='|', index=False)


def main2():
    '''
    reduce and include 
    new pred true label'''

    for i in PARA['tasks']:
        # find json file
        #if not i == 'sst':
        #    continue
        jpath = os.path.join(PARA['path'], '{}-{}-{}'.format('run', i, 'val.jsonl'))
        # find pred.p
        ppath = os.path.join(PARA['path'], '{}-{}-{}'.format('run', i, 'val_preds.p'))
        vpath = os.path.join(PARA['path'], '{}-{}-{}'.format('bin', i, 'val_preds.p'))
        # create dataframe containing vs and s-pred
        dt, inum, labs = tl_score(jpath, ppath, i) # inum|tl
        dt = pl_score(ppath, dt, inum)
        dt = vc_score(vpath, dt, inum, labs)
        # loop over multiple rounds
        for r in PARA['rnds']:
            # find csv file
            dpath = os.path.join(PARA['path'], 'rnd{}-{}-{}'.format(str(r), i, 'score.csv'))
            # find pred.p of each permutation
            rpath = os.path.join(PARA['path'], 'rnd{}-{}-{}'.format(str(r), i, 'val_preds.p')) 
            opath = os.path.join(PARA['path'], 'rnd{}-{}-{}'.format(str(r), i, 'rdc.csv')) 
            df = pd.read_csv(
                    dpath, 
                    sep='|',
                    on_bad_lines='skip',
                    # error_bad_lines=False,
                    engine='python',
                    quoting=3,
                    )
            print(i,df.shape)
            df = reduce(df)
            # print(i, r)
            df = df[pd.to_numeric(df['inum'], downcast="integer", errors='coerce').notnull()]
            dt['inum']=dt['inum'].astype(int)
            df['inum']=df['inum'].astype(int)
            df = dt.merge(df, how='left', on='inum')
            df = lt_score(rpath, df, inum)
            df = vt_score(rpath, df, inum, labs)
            df = df.round(4)
            df.to_csv(opath, sep='|', index=False)



def main3():
    '''
    Take average
    inum; slen; ca; pa; da; mi; vs; vt; vc; ta; tl
    no text
    '''
    mode = 'avg-all'
    rf = pd.DataFrame()
    opath = os.path.join(PARA['path'], '{}'.format('all-rdc.csv'))
    # Get avg from round 1 to round 6
    if mode == 'avg-all':
        for i in PARA['tasks']:
            # loop over multiple rounds
            of = pd.DataFrame()
            for r in PARA['rnds']:
                # find csv file
                dpath = os.path.join(PARA['path'], 'rnd{}-{}-{}'.format(str(r), i, 'rdc.csv'))
                # df = pd.read_csv(dpath, sep='|', on_bad_lines='skip', engine='python', quoting=3)
                df = pd.read_csv(dpath, sep='|')
                df['rnd'] = str(r)
                # print(df.head())
                if of.shape[0] == 0:
                    of = df
                else:
                    of = pd.concat((of, df))
            of = of.apply(lambda x: x.fillna(x.mean()), axis=0)

            print(of.head())
            
            # inum; slen; ca; pa; da; mi; vs; vt; vc; ta; tl
            rd = of.groupby('inum', as_index=False).agg({
                                                'slen':'mean',
                                                'ca':'mean',
                                                'pa':'mean',
                                                'da':'mean',
                                                'mi':'mean',
                                                'vs':'mean',
                                                'vt':'mean',
                                                'vc':'mean',
                                                'tl':'mean',
                                                'pl':'mean',
                                                'lt':'mean',}
                                                )
            rd['task'] = i
            if rf.shape[0] == 0:
                rf = rd
            else:
                rf = pd.concat((rf, rd))

    if mode != 'avg-all':
        pass

    rf = rf.round(4)
    rf.to_csv(opath, sep='|', index=False)


def main4():
    '''
    stat'''
    dpath = os.path.join(PARA['path'], '{}'.format('all-rdc.csv'))
    df = pd.read_csv(dpath, sep='|')
    rf = df.groupby('task', as_index=False).agg({
                                            'slen':'mean',
                                            'ca':'mean',
                                            'pa':'mean',
                                            'da':'mean',
                                            'mi':'mean',
                                            'vs':'mean',
                                            'vt':'mean',
                                            'vc':'mean',
                                            'tl':'mean',
                                            'pl':'mean',
                                            'lt':'mean'}
                                            )
    rf['vd'] = rf['vt']-rf['vs']
    rf['vc'] = rf['vt']-rf['vc']
    print(rf)


def main5():

    od = pd.DataFrame()

    for i in PARA['tasks']:

        jpath = os.path.join(PARA['path'], '{}-{}-{}'.format('run', i, 'val.jsonl'))
        ppath = os.path.join(PARA['path'], '{}-{}-{}'.format('run', i, 'val_preds.p'))
        dt = tl_pl_init(jpath, ppath, i)
        opath = os.path.join(PARA['path'], 'pred.csv') 
        # print(dt.head())
        # loop over multiple rounds
        for r in PARA['rnds']:
            # find pred.p of each permutation
            rpath = os.path.join(PARA['path'], 'rnd{}-{}-{}'.format(str(r), i, 'val_preds.p')) 
            df = avg_rnd_prd(rpath, dt, r)
            dt['prd-{}'.format(r)] = df['pl-{}'.format(r)]
            # df = df.merge(dt, how='left', on='inum')

        od = dt if od.shape[0] == 0 else pd.concat([od, dt])

    od.to_csv(opath, sep='|', index=False)


            

if __name__ == '__main__':

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
        'path': '/Users/xdchen/Downloads/eval2/data/data-out',
        'procnum1': 0, 
        'procnum2': 0, 
        'procnum3': 0, 
        'procnum4': 1, 
        'procnum5': 1, 
    }

    if PARA['procnum1'] == 1:
        main1()
    if PARA['procnum2'] == 1:
        main2()
    if PARA['procnum3'] == 1:
        main3()
    if PARA['procnum4'] == 1:
        main4()
    if PARA['procnum5'] == 1:
        main5()



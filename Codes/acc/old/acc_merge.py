import pandas as pd
import os
import numpy as np

def probe_clean(df, name):
    '''
    Formatting probe df'''

    dn = df[['item_num', 'probe-acc', 'sentlen', 'st5c', 'st5', 'sgpt2']]
    dn['idx'] = dn['item_num'].astype(int) - 1
    dn['dataset'] = name
    dn['mi'] = dn['st5c'] - dn['st5']
    dn['milm'] = dn['sgpt2'] - dn['st5']

    return dn

def merge():
    '''
    Two directories (probe files and control files)
    merge files by idx
    '''

    final = pd.DataFrame()
    
    for task in PARAS['TASKS']:

        print('Process Currect TASK {} ...'. format(task))

        cpath = os.path.join(PARAS['CPATH'], task+'.csv')
        ppath = os.path.join(PARAS['PPATH'], task+'.csv')

        cdf = pd.read_csv(cpath, header=0, index_col=0)
        pdf = pd.read_csv(ppath, header=0, index_col=0)

        pdf = probe_clean(pdf, task)

        # pdf = pdf[pdf['sentlen'] > 2]

        mdf = pd.merge(pdf, cdf, how='left', on='idx')
        mdf = mdf.drop(['item_num'], axis=1)
        mdf['dataset'] = task

        if len(final) == 0:
            final = mdf
        else:
            final = pd.concat([final, mdf])

        print('TASK Dataframe has {} items ...'. format(len(final)))

    print('SAVING TASK Dataframe ...')
    final.to_csv(PARAS['OPATH'], index=None)


if __name__ == '__main__':

    PARAS = {
        'PPATH': '/Users/xdchen/Downloads/data/ValidData/mi', # path to probe files
        'CPATH': '/Users/xdchen/Downloads/data/ValidData/control-acc', # path to control files
        'OPATH': '/Users/xdchen/Downloads/data/acc.csv', # path to outputs
        'TASKS':  ['boolq' , 'cb' , 'copa' , 'mrpc' , 'multirc' , 'rte' , 'wic' , 'winogrande' , 'wnli' , 'qqp' , 'wsc'],
    }

    merge()
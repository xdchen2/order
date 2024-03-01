
import os
import sys
import pandas as pd
from minicons import scorer

from probe_scorer import Seq2SeqScorer

def load_model(probe, device):
    '''
    Evalute on GPT2; T5; Probe
    '''

    loc_model = Seq2SeqScorer(model_name=probe, device=device)
    ilm_model = scorer.IncrementalLMScorer('distilgpt2', device)
    s2s_model = Seq2SeqScorer('t5-base', device)

    return loc_model, ilm_model, s2s_model

def conT5(model, sti, src):
    # p(s|t)
    return model.sequence_score(sti, source=src, reduction = lambda x: x.sum(0).item())[0]

def T5(model, sti): 
    # p(s)
    return model.sequence_score(sti, source_format = 'blank', reduction = lambda x: x.sum(0).item())[0]

def GPT2(model, sti):
    # p(s)
    return model.sequence_score(sti, reduction = lambda x: -x.sum(0).item())[0]

def scoring(df):
    '''
    Eval the p of sents with models
    '''

    df['slen'] = df['org'].apply(lambda x: len(x.strip().split(' ')))
    df['cp'] = df.apply(lambda x: conT5(loc_model, x['org'], x['rnd']), axis=1)
    df['mp'] = df['org'].apply(lambda x: T5(s2s_model, x))
    # df['mp2'] = df['org'].apply(lambda x: GPT2(ilm_model, x))
    df['mi'] = df.apply(lambda x: x['cp']/x['mp'], axis=1)

    return df


def path_eval():
    '''
    Test whether the file exists
    '''

    if not os.path.exists(PARAS['FPATH']):
        os.mkdir(PARAS['FPATH'])

    if not os.path.exists(PARAS['OPATH']):
        os.mkdir(PARAS['OPATH'])



if __name__ == '__main__':

    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    PARAS = {
        'FPATH': sys.argv[1], # path to probe files
        'MPATH': './model', # path to model files
        'OPATH': './out', # path to outputs
        'TASKS':  ['boolq' , 'cb' , 'copa' , 'mrpc' , 'multirc' , 'rte', 'winogrande' , 'qqp' , 'sst'],
    }

    loc_model, ilm_model, s2s_model = load_model(PARAS['MPATH'], device)

    for tname in PARAS['TASKS']:


        rpath = os.path.join(PARAS['FPATH'], 'data', tname, '{}-pmt.csv'.format(tname))
        # print(rpath)

        inpath = os.path.join(rpath)
        df = pd.read_csv(inpath, sep='|', header=0, warn_bad_lines=True, error_bad_lines=False, nrows=100000)
        df = scoring(df)
        df.to_csv(rpath, sep='|', index=False)